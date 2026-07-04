from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from wenyan_competition.agent import AcademicSearchAgent
from wenyan_competition.config import AppConfig, load_config
from wenyan_competition.dataset import load_jsonl
from wenyan_competition.metrics import aggregate, f1_at, precision_at, recall_at


DEFAULT_QUERY = ""


class DemoState:
    """Shared server state.

    The UI and the evaluation endpoint call the same AcademicSearchAgent used by
    the competition scripts, so the demo reflects the real ranking pipeline.
    """

    def __init__(
        self,
        config: AppConfig,
        config_path: str,
        use_llm: bool,
        fallback_models: bool,
    ):
        self.config = config
        self.config_path = config_path
        self.use_llm = use_llm
        self.fallback_models = fallback_models
        self.started_at = time.time()
        self.agent = AcademicSearchAgent(
            config,
            use_llm=use_llm,
            force_fallback_models=fallback_models,
        )
        self.lock = threading.Lock()


class DemoHandler(BaseHTTPRequestHandler):
    state: DemoState

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML)
            return
        if parsed.path == "/api/health":
            self._send_json(self._health())
            return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = (params.get("query") or [""])[0]
            top_k = _int((params.get("top_k") or ["10"])[0], 10)
            self._search(query, top_k)
            return
        if parsed.path == "/api/evaluate_smoke":
            self._evaluate_smoke()
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            data = self._read_json()
            self._search(str(data.get("query") or ""), _int(data.get("top_k"), 10))
            return
        self._send_json({"error": "not found"}, status=404)

    def _search(self, query: str, top_k: int) -> None:
        query = query.strip()
        if not query:
            self._send_json({"error": "query is required"}, status=400)
            return
        top_k = max(1, min(50, top_k))
        started = time.time()
        try:
            with self.state.lock:
                result = self.state.agent.search(query, top_k=top_k)
            payload = result.to_dict()
            payload["server_latency_seconds"] = time.time() - started
            payload["retrieval_mode"] = self._retrieval_mode()
            payload["demo_notice"] = self._search_notice(len(result.papers))
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _evaluate_smoke(self) -> None:
        eval_path = Path("data/sample_eval.jsonl")
        if not eval_path.exists():
            self._send_json({"error": "data/sample_eval.jsonl not found"}, status=404)
            return
        rows = []
        examples = load_jsonl(eval_path)
        try:
            with self.state.lock:
                for ex in examples:
                    result = self.state.agent.search(ex.query, top_k=100)
                    pred_ids = [p.paper_id or p.doi or p.title for p in result.papers]
                    rows.append(
                        {
                            "query": ex.query,
                            "gold_ids": sorted(ex.gold_ids),
                            "pred_ids": pred_ids[:20],
                            "precision@20": precision_at(pred_ids, ex.gold_ids, 20),
                            "recall@20": recall_at(pred_ids, ex.gold_ids, 20),
                            "recall@50": recall_at(pred_ids, ex.gold_ids, 50),
                            "recall@100": recall_at(pred_ids, ex.gold_ids, 100),
                            "f1@20": f1_at(pred_ids, ex.gold_ids, 20),
                            "latency_seconds": result.stats.latency_seconds,
                            "api_calls": float(result.stats.api_calls),
                            "llm_calls": float(result.stats.llm_calls),
                        }
                    )
            metrics = aggregate(
                [
                    {
                        "precision@20": r["precision@20"],
                        "recall@20": r["recall@20"],
                        "recall@50": r["recall@50"],
                        "recall@100": r["recall@100"],
                        "f1@20": r["f1@20"],
                        "api_calls": r["api_calls"],
                        "llm_calls": r["llm_calls"],
                        "latency_seconds": r["latency_seconds"],
                    }
                    for r in rows
                ]
            )
            self._send_json({"examples": len(rows), "metrics": metrics, "rows": rows})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _health(self) -> Dict[str, Any]:
        retrieval = self.state.config.retrieval
        small = self.state.config.small_models
        return {
            "status": "ok",
            "config_path": self.state.config_path,
            "use_llm": self.state.use_llm,
            "llm_base_url": self.state.config.llm.base_url,
            "llm_model": self.state.config.llm.model,
            "llm_api_key_configured": bool(self.state.config.llm.api_key),
            "fallback_models": self.state.fallback_models,
            "local_corpus_path": retrieval.local_corpus_path,
            "use_openalex": retrieval.use_openalex,
            "use_semantic_scholar": retrieval.use_semantic_scholar,
            "use_arxiv": retrieval.use_arxiv,
            "use_serper": retrieval.use_serper,
            "serper_api_key_configured": bool(retrieval.serper_api_key),
            "pasa_id2paper_path": retrieval.pasa_id2paper_path,
            "academic_only": retrieval.academic_only,
            "embedding_model": small.embedding_model,
            "reranker_model": small.reranker_model,
            "retrieval_mode": self._retrieval_mode(),
            "uptime_seconds": time.time() - self.state.started_at,
        }

    def _retrieval_mode(self) -> str:
        retrieval = self.state.config.retrieval
        if (
            retrieval.local_corpus_path
            and not retrieval.use_openalex
            and not retrieval.use_semantic_scholar
            and not retrieval.use_arxiv
            and not (retrieval.use_serper and retrieval.serper_api_key)
        ):
            return "smoke_local_only"
        if (
            retrieval.use_openalex
            or retrieval.use_semantic_scholar
            or retrieval.use_arxiv
            or (retrieval.use_serper and retrieval.serper_api_key)
        ):
            return "online_api"
        return "custom"

    def _search_notice(self, paper_count: int) -> str:
        if self._retrieval_mode() == "smoke_local_only":
            if paper_count == 0:
                return (
                    "当前使用的是离线 smoke 样例库，只包含少量 AI/软件工程示例论文。"
                    "这个查询没有命中样例库，请切换 config.online.json 或 config.yaml 进行真实学术检索。"
                )
            return "当前使用的是离线 smoke 样例库，结果只用于验证流程，不代表真实论文检索结果。"
        return ""

    def _read_json(self) -> Dict[str, Any]:
        length = _int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Wenyan competition pipeline web demo")
    parser.add_argument("--config", default="config.smoke.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--fallback_models", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    DemoHandler.state = DemoState(
        config=config,
        config_path=args.config,
        use_llm=not args.no_llm,
        fallback_models=args.fallback_models,
    )
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Web demo: http://{args.host}:{args.port}/")
    print(f"Config: {args.config}; use_llm={not args.no_llm}; fallback_models={args.fallback_models}")
    server.serve_forever()


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>问研智搜</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #627084;
      --line: #dfe6ee;
      --blue: #246bfe;
      --green: #16845b;
      --red: #c2413b;
      --amber: #a46300;
      --chip: #eef3ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .wrap {
      max-width: 1260px;
      margin: 0 auto;
      padding: 18px 20px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    main.wrap {
      display: grid;
      grid-template-columns: minmax(360px, 430px) 1fr;
      gap: 16px;
      align-items: start;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .panel-body { padding: 16px; }
    textarea {
      width: 100%;
      min-height: 128px;
      resize: vertical;
      border: 1px solid #cfd8e3;
      border-radius: 6px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.5;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    label { color: var(--muted); font-size: 13px; }
    input[type="number"] {
      width: 80px;
      border: 1px solid #cfd8e3;
      border-radius: 6px;
      padding: 8px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      font-weight: 700;
      padding: 9px 13px;
      cursor: pointer;
    }
    button.secondary {
      background: #e8eef8;
      color: #1c304f;
    }
    button:disabled {
      opacity: .62;
      cursor: progress;
    }
    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 66px;
      background: #fbfdff;
    }
    .metric b {
      display: block;
      font-size: 20px;
      margin-top: 4px;
    }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .plan {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .chip {
      background: var(--chip);
      color: #17345d;
      border: 1px solid #d5e1ff;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
    }
    .results {
      display: grid;
      gap: 10px;
    }
    .paper {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .paper-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }
    .paper h3 {
      margin: 0;
      font-size: 16px;
      line-height: 1.35;
    }
    .score {
      min-width: 76px;
      text-align: right;
      color: var(--green);
      font-weight: 800;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .abstract {
      color: #344054;
      font-size: 13px;
      line-height: 1.5;
      margin-top: 9px;
    }
    .bars {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .bar-label {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .bar {
      height: 6px;
      background: #e8edf4;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar > i {
      display: block;
      height: 100%;
      background: var(--blue);
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
    }
    .tab {
      background: #eef2f7;
      color: #26384d;
    }
    .tab.active {
      background: var(--ink);
      color: white;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      color: #dbeafe;
      padding: 12px;
      border-radius: 6px;
      max-height: 420px;
      overflow: auto;
    }
    svg {
      width: 100%;
      min-height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }
    .notice {
      border-left: 4px solid var(--amber);
      background: #fff9ec;
      padding: 10px 12px;
      color: #5b410c;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
    }
    .synthesis {
      display: grid;
      gap: 12px;
    }
    .synthesis-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .synthesis-box h3 {
      margin: 0 0 8px;
      font-size: 15px;
    }
    .synthesis-box p,
    .synthesis-box li {
      color: #344054;
      font-size: 13px;
      line-height: 1.55;
    }
    .synthesis-box ul {
      margin: 0;
      padding-left: 20px;
    }
    .synthesis-list {
      display: grid;
      gap: 8px;
    }
    .synthesis-item {
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px;
      background: #fbfdff;
      font-size: 13px;
      line-height: 1.5;
    }
    .synthesis-item b {
      color: var(--ink);
    }
    .trace {
      display: grid;
      gap: 10px;
    }
    .trace-step {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .trace-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 6px;
    }
    .trace-top b {
      color: var(--ink);
    }
    .trace-role {
      color: var(--green);
      font-weight: 800;
      font-size: 12px;
    }
    .trace-counts {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    @media (max-width: 980px) {
      main.wrap { grid-template-columns: 1fr; }
      .bars { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>问研智搜</h1>
      <div class="sub">江苏大学 汽车工程研究院 搜的都队</div>
    </div>
  </header>

  <main class="wrap">
    <aside>
      <div class="panel-head">搜索与评测</div>
      <div class="panel-body">
        <textarea id="query" placeholder="输入论文检索问题或关键词"></textarea>
        <div class="row">
          <label>Top K <input id="topK" type="number" value="8" min="1" max="50" /></label>
          <button id="searchBtn">搜索论文</button>
          <button class="secondary" id="evalBtn">跑 smoke 评测</button>
        </div>
        <div class="row">
          <button class="secondary" data-q="retrieval augmented generation evaluation evidence attribution">RAG 评测</button>
          <button class="secondary" data-q="software vulnerability detection graph neural network">漏洞检测</button>
        </div>
        <div id="notice" class="notice" style="margin-top:12px;">正在读取服务状态...</div>
        <div class="status-grid">
          <div class="metric"><span class="small">LLM Calls</span><b id="llmCalls">-</b></div>
          <div class="metric"><span class="small">API Calls</span><b id="apiCalls">-</b></div>
          <div class="metric"><span class="small">Latency</span><b id="latency">-</b></div>
          <div class="metric"><span class="small">Papers</span><b id="paperCount">-</b></div>
        </div>
      </div>
    </aside>

    <section>
      <div class="panel-head">结果展示</div>
      <div class="panel-body">
        <div class="tabs">
          <button class="tab active" data-tab="papers">论文排序</button>
          <button class="tab" data-tab="plan">查询拆解</button>
          <button class="tab" data-tab="trace">Agent轨迹</button>
          <button class="tab" data-tab="synthesis">结果归纳</button>
          <button class="tab" data-tab="graph">关系图</button>
          <button class="tab" data-tab="raw">JSON</button>
        </div>
        <div id="papers"></div>
        <div id="plan" style="display:none;"></div>
        <div id="trace" style="display:none;"></div>
        <div id="synthesis" style="display:none;"></div>
        <div id="graph" style="display:none;"></div>
        <pre id="raw" style="display:none;"></pre>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let current = null;

    async function health() {
      const res = await fetch('/api/health');
      const data = await res.json();
      const mode = data.retrieval_mode === 'smoke_local_only' ? '离线样例库' : '在线学术 API';
      const llmStatus = data.use_llm
        ? `启用；模型：${data.llm_model || '-'}；Key：${data.llm_api_key_configured ? '已配置' : '未配置'}`
        : '关闭';
      const retrievalSources = [
        data.use_serper && data.serper_api_key_configured ? 'Serper/arXiv' : '',
        data.use_arxiv ? 'arXiv' : '',
        data.use_openalex ? 'OpenAlex' : '',
        data.use_semantic_scholar ? 'Semantic Scholar' : ''
      ].filter(Boolean).join(' / ') || '自定义语料';
      $('notice').innerHTML = `配置：${data.config_path}<br>模式：${mode}<br>LLM：${llmStatus}<br>小模型：${data.fallback_models ? 'fallback' : '真实模型优先'}；检索：${data.local_corpus_path || retrievalSources}；学术过滤：${data.academic_only ? '开启' : '关闭'}`;
    }

    async function search() {
      const btn = $('searchBtn');
      if (!$('query').value.trim()) {
        current = null;
        $('papers').innerHTML = `<div class="notice">请输入检索问题后再搜索。</div>`;
        $('plan').innerHTML = '';
        $('trace').innerHTML = '';
        $('synthesis').innerHTML = '';
        $('graph').innerHTML = '';
        $('raw').textContent = '';
        $('llmCalls').textContent = '-';
        $('apiCalls').textContent = '-';
        $('latency').textContent = '-';
        $('paperCount').textContent = '-';
        showTab('papers');
        return;
      }
      btn.disabled = true;
      btn.textContent = '搜索中...';
      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: $('query').value, top_k: Number($('topK').value || 8)})
        });
        current = await res.json();
        if (current.error) throw new Error(current.error);
        render(current);
      } catch (err) {
        $('papers').innerHTML = `<div class="notice" style="border-left-color:var(--red);background:#fff1f1;color:#74211d;">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = '搜索论文';
      }
    }

    async function evalSmoke() {
      const btn = $('evalBtn');
      btn.disabled = true;
      btn.textContent = '评测中...';
      try {
        const res = await fetch('/api/evaluate_smoke');
        const data = await res.json();
        current = data;
        $('raw').textContent = JSON.stringify(data, null, 2);
        showTab('raw');
        const m = data.metrics || {};
        $('notice').innerHTML = `Smoke 评测完成：F1@20=${fmt(m['f1@20'])}，Recall@100=${fmt(m['recall@100'])}，样例数=${data.examples}`;
      } catch (err) {
        $('notice').textContent = err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = '跑 smoke 评测';
      }
    }

    function render(data) {
      const papers = data.papers || [];
      $('llmCalls').textContent = data.stats?.llm_calls ?? 0;
      $('apiCalls').textContent = data.stats?.api_calls ?? 0;
      $('latency').textContent = `${(data.stats?.latency_seconds || 0).toFixed(2)}s`;
      $('paperCount').textContent = papers.length;
      $('raw').textContent = JSON.stringify(data, null, 2);
      renderPlan(data.plan || {});
      renderPapers(papers);
      renderTrace(data.agent_trace || []);
      renderSynthesis(data.synthesis || {}, papers);
      renderGraph(papers);
      const warnings = data.stats?.warnings || [];
      if (warnings.length) {
        $('notice').innerHTML = `<b>检索告警</b><br>${warnings.map(escapeHtml).join('<br>')}`;
      } else if (data.demo_notice) {
        $('notice').innerHTML = escapeHtml(data.demo_notice);
      }
      showTab('papers');
    }

    function renderPlan(plan) {
      const sub = plan.sub_queries || [];
      $('plan').innerHTML = `
        <div class="small">Intent</div>
        <h3>${escapeHtml(plan.intent || plan.original_query || '')}</h3>
        <div class="small">Entities</div>
        <div class="plan">${(plan.entities || []).map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('')}</div>
        <div class="small" style="margin-top:14px;">Sub Queries</div>
        <div class="plan">${sub.map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('')}</div>
      `;
    }

    function renderPapers(papers) {
      if (!papers.length) {
        const msg = current?.demo_notice || '没有找到足够相关的论文。请换成在线配置，或扩大本地语料库。';
        $('papers').innerHTML = `<div class="notice">${escapeHtml(msg)}</div>`;
        return;
      }
      $('papers').className = 'results';
      $('papers').innerHTML = papers.map((p, i) => `
        <article class="paper">
          <div class="paper-top">
            <h3>${i + 1}. ${escapeHtml(p.title || '')}</h3>
            <div class="score">${fmt(p.final_score)}</div>
          </div>
          <div class="meta">${escapeHtml([p.year, p.venue, p.publication_type, p.source, p.paper_id].filter(Boolean).join(' · '))}</div>
          <div class="abstract">${escapeHtml((p.abstract || '').slice(0, 520))}</div>
          <div class="bars">
            ${bar('API', p.api_score)}
            ${bar('BM25', p.bm25_score)}
            ${bar('Embedding', p.embedding_score)}
            ${bar('Reranker', p.reranker_score)}
            ${bar('LLM', p.llm_score)}
            ${bar('Authority', p.authority_score)}
            ${bar('Recency', p.recency_score)}
            ${bar('Diversity', p.diversity_score)}
          </div>
        </article>
      `).join('');
    }

    function renderTrace(trace) {
      if (!trace.length) {
        $('trace').innerHTML = `<div class="notice">暂无 Agent 搜索轨迹。请先完成一次检索。</div>`;
        return;
      }
      $('trace').className = 'trace';
      $('trace').innerHTML = trace.map(step => `
        <div class="trace-step">
          <div class="trace-top">
            <b>${escapeHtml(step.step || '')}. ${escapeHtml(step.action || '')}</b>
            <span class="trace-role">${escapeHtml(step.role || '')}</span>
          </div>
          <div class="small">${escapeHtml(step.detail || '')}</div>
          <div class="plan">${(step.queries || []).map(q => `<span class="chip">${escapeHtml(q)}</span>`).join('')}</div>
          <div class="trace-counts">
            before ${Number(step.candidates_before || 0)} · after ${Number(step.candidates_after || 0)} · selected ${Number(step.selected_count || 0)}
          </div>
        </div>
      `).join('');
    }

    function renderSynthesis(synthesis, papers) {
      if (!papers.length) {
        $('synthesis').innerHTML = `<div class="notice">没有可归纳的候选论文。请先完成一次有效检索。</div>`;
        return;
      }
      const high = Array.isArray(synthesis.highly_relevant) ? synthesis.highly_relevant : [];
      const partial = Array.isArray(synthesis.partial_relevant) ? synthesis.partial_relevant : [];
      const themes = Array.isArray(synthesis.themes) ? synthesis.themes : [];
      const gaps = Array.isArray(synthesis.gaps) ? synthesis.gaps : [];
      const suggestions = Array.isArray(synthesis.next_search_suggestions) ? synthesis.next_search_suggestions : [];
      $('synthesis').className = 'synthesis';
      $('synthesis').innerHTML = `
        <div class="synthesis-box">
          <h3>整体结论</h3>
          <p>${escapeHtml(synthesis.overview || '当前结果已完成排序，但没有生成额外归纳。')}</p>
        </div>
        <div class="synthesis-box">
          <h3>主题线索</h3>
          <div class="plan">${themes.map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('') || '<span class="small">暂无明显主题词</span>'}</div>
        </div>
        <div class="synthesis-box">
          <h3>高度相关候选</h3>
          ${renderSynthesisItems(high)}
        </div>
        <div class="synthesis-box">
          <h3>部分相关候选</h3>
          ${renderSynthesisItems(partial)}
        </div>
        <div class="synthesis-box">
          <h3>证据缺口</h3>
          ${renderList(gaps, '暂无明显缺口提示')}
        </div>
        <div class="synthesis-box">
          <h3>下一轮检索建议</h3>
          ${renderList(suggestions, '暂无新的检索建议')}
        </div>
      `;
    }

    function renderSynthesisItems(items) {
      if (!items.length) return '<div class="small">暂无</div>';
      return `<div class="synthesis-list">${items.map(item => `
        <div class="synthesis-item">
          <b>${escapeHtml(item.rank ? `#${item.rank} ` : '')}${escapeHtml(item.title || '')}</b><br>
          ${escapeHtml(item.reason || '')}
        </div>
      `).join('')}</div>`;
    }

    function renderList(items, emptyText) {
      if (!items.length) return `<div class="small">${escapeHtml(emptyText)}</div>`;
      return `<ul>${items.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul>`;
    }

    function renderGraph(papers) {
      const w = 760, h = 380, cx = w / 2, cy = h / 2;
      const nodes = papers.slice(0, 10).map((p, i) => {
        const angle = (Math.PI * 2 * i) / Math.max(1, Math.min(10, papers.length));
        return {id: p.paper_id || String(i), title: p.title || '', x: cx + Math.cos(angle) * 260, y: cy + Math.sin(angle) * 130};
      });
      const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
      const edges = [];
      papers.slice(0, 10).forEach(p => {
        const from = byId[p.paper_id];
        (p.references || []).forEach(r => {
          if (from && byId[r]) edges.push([from, byId[r]]);
        });
      });
      $('graph').innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="paper relation graph">
          ${edges.map(([a,b]) => `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#94a3b8" stroke-width="1.5" />`).join('')}
          ${nodes.map((n, i) => `
            <g>
              <circle cx="${n.x}" cy="${n.y}" r="${i === 0 ? 24 : 18}" fill="${i === 0 ? '#246bfe' : '#16845b'}" />
              <text x="${n.x}" y="${n.y + 4}" text-anchor="middle" fill="white" font-size="12" font-weight="700">${i + 1}</text>
              <text x="${Math.min(w - 180, Math.max(10, n.x - 80))}" y="${n.y + 42}" fill="#334155" font-size="11">${escapeSvg(short(n.title, 30))}</text>
            </g>`).join('')}
        </svg>
      `;
    }

    function bar(label, value) {
      const v = Math.max(0, Math.min(1, Number(value || 0)));
      return `<div><div class="bar-label">${label} ${fmt(v)}</div><div class="bar"><i style="width:${v * 100}%"></i></div></div>`;
    }

    function showTab(name) {
      ['papers', 'plan', 'trace', 'synthesis', 'graph', 'raw'].forEach(id => $(id).style.display = id === name ? '' : 'none');
      document.querySelectorAll('.tab').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
    }

    function fmt(v) {
      return Number(v || 0).toFixed(3);
    }

    function short(s, n) {
      return s.length > n ? s.slice(0, n - 1) + '...' : s;
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function escapeSvg(s) {
      return escapeHtml(s);
    }

    document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));
    document.querySelectorAll('[data-q]').forEach(btn => btn.addEventListener('click', () => { $('query').value = btn.dataset.q; search(); }));
    $('searchBtn').addEventListener('click', search);
    $('evalBtn').addEventListener('click', evalSmoke);
    health().then(() => {
      $('papers').innerHTML = `<div class="notice">请输入检索问题后点击搜索。</div>`;
      showTab('papers');
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
