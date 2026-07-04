# Git 部署说明

## 适用场景

如果后续代码经常修改，推荐用 Git 推送到 GitHub、Gitee 或自建仓库，服务器只需要 `git pull`，不用每次重新压缩和上传 zip。

## 本地首次推送

在 Git Bash 中执行：

```bash
cd /c/Users/ak/Documents/Codex/2026-07-02/https-chatgpt-com-share-6a465475-b6c8/outputs/wenyan-zhisou-advanced
git init
git add .
git commit -m "init wenyan zhisou competition"
git branch -M main
git remote add origin 你的仓库地址
git push -u origin main
```

注意：`.gitignore` 已排除 `competition/data/pasa-dataset/`、`competition/runs/`、`.venv/` 和 `competition/config.yaml`，避免把大数据、运行结果和本地密钥配置推上去。

## 服务器首次拉取

```bash
cd ~
mv wenyan-zhisou-advanced wenyan-zhisou-advanced.bak.$(date +%Y%m%d_%H%M%S)
git clone 你的仓库地址 wenyan-zhisou-advanced
cp -r wenyan-zhisou-advanced.bak.*/competition/data/pasa-dataset wenyan-zhisou-advanced/competition/data/ 2>/dev/null || true
cp wenyan-zhisou-advanced.bak.*/competition/config.yaml wenyan-zhisou-advanced/competition/config.yaml 2>/dev/null || true
cd ~/wenyan-zhisou-advanced/competition
source .venv/bin/activate 2>/dev/null || python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python offline_quality_check.py
```

## 后续更新

本地：

```bash
cd /c/Users/ak/Documents/Codex/2026-07-02/https-chatgpt-com-share-6a465475-b6c8/outputs/wenyan-zhisou-advanced
git add .
git commit -m "update search agent"
git push
```

服务器：

```bash
cd ~/wenyan-zhisou-advanced
git pull
cd competition
source .venv/bin/activate
python offline_quality_check.py
```

## 服务器正式评测

新版 `evaluate_pasa.py` 默认启用正式评测增强：更大的候选池、更多 LLM selector 验证、PaSa-style selector-first 排序，并输出 `hit_report.json`。

```bash
cd ~/wenyan-zhisou-advanced/competition
source .venv/bin/activate
export DEEPSEEK_API_KEY="你的真实 DeepSeek Key"
python evaluate_pasa.py \
  --config config.yaml \
  --input data/pasa-dataset/RealScholarQuery/test.jsonl \
  --output_dir runs/pasa_rs_limit5 \
  --limit 5 \
  --top_k 20
cat runs/pasa_rs_limit5/metrics.json
python - <<'PY'
import json
rows=json.load(open("runs/pasa_rs_limit5/hit_report.json", encoding="utf-8"))
for i, row in enumerate(rows):
    print(i, row["hit_at_20"], row["hit_at_50"], row["hit_at_100"], row["query"][:80])
PY
```
