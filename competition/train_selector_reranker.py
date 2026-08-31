from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SelectorExample:
    query: str
    document: str
    label: int


def parse_selector_record(record: dict) -> SelectorExample:
    """Convert one official PaSa selector conversation into a ranking pair."""

    messages = record.get("messages") or []
    if len(messages) < 2:
        raise ValueError("selector record must contain user and assistant messages")
    prompt = str(messages[0].get("content") or "")
    answer = str(messages[-1].get("content") or "").strip().lower()

    title = _capture(prompt, r"Searched Paper:\s*\nTitle:\s*(.*?)\nAbstract:")
    abstract = _capture(prompt, r"\nAbstract:\s*(.*?)\n\nUser Query:")
    query = _capture(prompt, r"\nUser Query:\s*(.*?)\n\nOutput format:")
    if answer.startswith("true"):
        label = 1
    elif answer.startswith("false"):
        label = 0
    else:
        raise ValueError("selector decision must begin with True or False")

    document = f"{title}\n{abstract}".strip()
    if not query or not document:
        raise ValueError("selector query/title/abstract could not be parsed")
    return SelectorExample(query=query.strip(), document=document, label=label)


def load_selector_jsonl(
    path: str | Path,
    *,
    limit: int = 0,
    seed: int = 42,
) -> list[SelectorExample]:
    rows: list[SelectorExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(parse_selector_record(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid selector row {line_number}: {exc}") from exc
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the lightweight reranker on the official PaSa selector data."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--base_model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output_dir", default="models/wenyan-selector-reranker")
    parser.add_argument("--base_config", default="config.v12.yaml")
    parser.add_argument("--output_config", default="config.v12.selector.yaml")
    parser.add_argument("--max_train_samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_auc_gain", type=float, default=0.005)
    args = parser.parse_args()

    import numpy as np
    import torch
    import yaml
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for selector fine-tuning")
    _set_seed(args.seed, torch, np)

    train_rows = load_selector_jsonl(
        args.train,
        limit=args.max_train_samples,
        seed=args.seed,
    )
    dev_rows = load_selector_jsonl(args.dev, seed=args.seed)
    _validate_balance(train_rows, "train")
    _validate_balance(dev_rows, "dev")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    device = torch.device("cuda")
    model.to(device)

    collate = _collator(tokenizer, args.max_length, torch)
    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_rows,
        batch_size=max(args.batch_size, 16),
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    baseline = _evaluate(model, dev_loader, device, torch, np, _metric_functions())
    print("Baseline selector metrics:", json.dumps(baseline, ensure_ascii=False))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation))
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_metrics: dict[str, float] | None = None
    best_state: dict[str, object] | None = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits.reshape(-1)
                loss = loss_fn(logits, labels) / max(1, args.gradient_accumulation)
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * max(1, args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1} batch={step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}"
                )

        metrics = _evaluate(model, dev_loader, device, torch, np, _metric_functions())
        metrics["train_loss"] = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} metrics:", json.dumps(metrics, ensure_ascii=False))
        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_metrics is not None and best_state is not None
    accepted = (
        best_metrics["roc_auc"] >= baseline["roc_auc"] + args.min_auc_gain
        and best_metrics["f1"] >= baseline["f1"] - 0.01
    )
    summary = {
        "accepted": accepted,
        "selection_rule": (
            f"roc_auc gain >= {args.min_auc_gain:.4f} and f1 drop <= 0.01 on PaSa selector test"
        ),
        "base_model": args.base_model,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "baseline": baseline,
        "trained": best_metrics,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not accepted:
        print("REJECTED: trained selector did not pass the independent quality gate.")
        return

    model.load_state_dict(best_state)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8")) or {}
    config.setdefault("small_models", {})["reranker_model"] = str(output_dir)
    Path(args.output_config).write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"ACCEPTED: model saved to {output_dir}")
    print(f"Evaluation config written to {args.output_config}")


def _capture(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _validate_balance(rows: Sequence[SelectorExample], name: str) -> None:
    positives = sum(row.label for row in rows)
    negatives = len(rows) - positives
    if not rows or positives == 0 or negatives == 0:
        raise ValueError(f"{name} selector data must contain positive and negative examples")
    print(f"{name}: total={len(rows)} positive={positives} negative={negatives}")


def _set_seed(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collator(tokenizer, max_length: int, torch):
    def collate(rows: Sequence[SelectorExample]):
        encoded = tokenizer(
            [row.query for row in rows],
            [row.document for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row.label for row in rows], dtype=torch.float32)
        return encoded

    return collate


def _metric_functions():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _evaluate(model, loader, device, torch, np, metrics) -> dict[str, float]:
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score = metrics
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits.reshape(-1)
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
            labels.extend(batch_labels.tolist())
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(probabilities, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


if __name__ == "__main__":
    main()
