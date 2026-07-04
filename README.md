# 问研智搜：赛题三竞赛版

本项目只保留 2026 年中国研究生人工智能大赛华为赛题三需要使用的 Python 竞赛管线。早期演示版、旧运行结果、临时打包目录和重复说明文档已经清理。

## 当前目录

```text
wenyan-zhisou-advanced/
├── README.md
├── competition/
│   ├── run_agent.py              # 单条查询入口
│   ├── evaluate_pasa.py          # 官方/公开 JSONL 测试集评测入口
│   ├── web_demo.py               # 前端可视化演示
│   ├── requirements.txt          # 完整模型环境依赖
│   ├── requirements-lite.txt     # 轻量演示依赖
│   ├── config.example.yaml       # 正式配置模板
│   ├── config.llm.example.yaml   # API 大模型配置模板
│   ├── config.online.json        # 在线 API + fallback 小模型
│   ├── config.smoke.json         # 离线样例冒烟测试
│   ├── data/                     # 小样例数据
│   └── wenyan_competition/       # 核心源码
└── docs/
    ├── 01_项目完整报告.md
    ├── 02_技术方案说明.md
    ├── 03_实验结果与消融实验.md
    └── 04_通俗解释报告.md
```

## 先跑通

进入竞赛目录：

```bash
cd competition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lite.txt
```

Windows PowerShell 使用：

```powershell
cd competition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lite.txt
```

离线冒烟测试：

```bash
python run_agent.py --config config.smoke.json --query "large language model hallucination detection factuality evaluation" --output runs/smoke_single.json --no_llm --fallback_models --top_k 5
python evaluate_pasa.py --config config.smoke.json --input data/sample_eval.jsonl --output_dir runs/smoke_eval --no_llm --fallback_models --top_k 20
```

启动前端演示：

```bash
python web_demo.py --config config.online.json --no_llm --fallback_models --port 8091
```

浏览器打开：

```text
http://127.0.0.1:8091/
```

## 正式比赛模式

租 GPU 服务器后安装完整依赖：

```bash
cd competition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

配置 `config.yaml` 中的大模型、小模型、OpenAlex、Semantic Scholar、Serper 等参数，然后运行：

```bash
export DEEPSEEK_API_KEY="你的DeepSeek Key"
export SERPER_API_KEY="你的Serper Key"
python evaluate_pasa.py --config config.yaml --input data/pasa-dataset/RealScholarQuery/test.jsonl --output_dir runs/pasa_realscholar
```

如果只是用在线大模型 API，而不在服务器本地部署 LLM，可以把 `config.llm.example.yaml` 复制成 `config.yaml`，设置对应 API Key 后运行。

## 核心能力

- 大模型：负责查询理解、子查询拆解、二轮查询演化、候选论文相关性验证和结果归纳。
- 小模型：负责 embedding 语义召回和 cross-encoder reranker 精排。
- 学术 API：对接 OpenAlex、Semantic Scholar、arXiv，并可选接入 Serper/Google 风格搜索来补强 arXiv 召回。
- PaSaTitleDB：如果服务器存在 PaSa 数据集的 `paper_database/id2paper.json`，系统会自动作为本地高召回标题库使用；没有该文件也能正常跳过。
- Agent：采用轻量 PaSa-inspired Crawler/Selector 架构，自动组织多策略检索、候选合并、去重、一跳引文网络扩展、查询演化、批量相关性验证和综合排序。
- 效率优化：在线 API 检索支持并发与进程内缓存，小模型 Embedding/Reranker 分数支持缓存，正式评测时减少重复模型计算和重复网络请求。
- 全文扩展：本地语料支持 `full_text`/`text` 字段参与 BM25、Embedding 和 Reranker 打分；在线 API 仍以标题、摘要和元数据为主。
- 排序策略：显式融合相关性、权威性、时效性和多样性，缓解“高召回带来噪声”和“结果同质化”的问题。
- 评测：输出 Precision@20、Recall@20、Recall@50、Recall@100、F1@20、API 调用次数、LLM 调用次数、延迟和逐条命中报告。
- 前端：展示论文排序、查询拆解、Agent 搜索轨迹、结果归纳、关系图和 JSON 结果。

## 相比参考系统的轻量改进点

本项目没有复刻 PaSa 的训练框架和强化学习 checkpoint，而是实现了一个更容易部署的轻量版本：用 Crawler 负责多策略召回、查询演化和引文网络扩展，用 Selector 负责候选预筛和 LLM 相关性验证。针对赛题提到的四类挑战，当前版本做了如下工程化改进：

- 查询理解不充分：`LLMPlanner` 解析实体、方法、数据集、约束；解析失败时有本地领域词扩展兜底。
- 覆盖率与精确度平衡：Crawler 采用 semantic-core、constraint-focused、authority-oriented、recency-oriented、多 API、Serper/arXiv 和本地 PaSaTitleDB 多路召回；Selector 再过滤低质量候选并做 F1@20 导向排序。
- 权威性、时效性、相关性、多样性权衡：`ranker.py` 新增 authority、recency、diversity 分数，并与 BM25、Embedding、Reranker、LLM verifier 融合。
- 结构化归纳展示：前端提供“Agent轨迹”“结果归纳”“关系图”，便于答辩时解释系统如何从查询走到最终结果。

## 重要说明

`config.smoke.json` 只用于验证工程链路，里面只有少量样例论文，不能代表真实比赛效果。正式效果必须使用 `config.yaml`、真实学术 API、真实大模型/小模型和官方公开测试集验证。

当前版本已经补上更强的查询演化、可用全文字段检索、结果归纳展示和离线质量自检，但这些属于工程能力补强，不等于最终指标已经达标。比赛成绩仍需要接通 DeepSeek/其他 OpenAI-compatible LLM、稳定学术 API Key，并在官方或自建测试集上运行 `evaluate_pasa.py`。
