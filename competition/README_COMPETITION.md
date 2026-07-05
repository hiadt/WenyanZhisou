# Competition 运行说明

本目录是项目主线。比赛、评测、前端演示和服务器部署都围绕这里运行。

## 模块对应关系

| 赛题要求 | 本项目实现 |
|---|---|
| 查询理解与拆解 | `LLMPlanner` 生成 `QueryPlan`，解析意图、实体、方法、数据集和约束 |
| 自主搜索策略迭代 | 轻量 Crawler/Selector 架构，`LLMQueryEvolver` 根据候选结果生成下一轮英文检索式 |
| 学术 API 检索 | `retrievers.py` 对接 Serper/arXiv、arXiv、OpenAlex 与 Semantic Scholar |
| 引文网络探索 | Crawler 从高分种子论文的一跳 references/citations 拉取补充候选 |
| 标题/摘要/全文检索 | `LocalCorpusRetriever` 支持 `full_text`/`text` 字段，排序阶段统一进入文本打分 |
| 小模型排序 | `models.py` 中 embedding scorer 与 cross-encoder reranker |
| 综合排序 | `ranker.py` 融合 API、BM25、Embedding、Reranker、LLM verifier、权威性、时效性、多样性分数 |
| 搜索结果归纳 | `ResultSynthesizer` 输出整体结论、主题线索、相关候选、证据缺口和下一轮检索建议 |
| 官方格式评测 | `evaluate_pasa.py` 读取 JSONL 并输出指标、逐条命中报告和延迟统计 |
| 可视化演示 | `web_demo.py` 提供论文排序、查询拆解、Agent 轨迹、结果归纳、关系图和 JSON 界面 |

## 依赖选择

轻量演示环境：

```bash
pip install -r requirements-lite.txt
```

适合在线 API + fallback 小模型，不下载本地模型权重，主要用于前端展示、接口联调和工程链路验证。

完整比赛环境：

```bash
pip install -r requirements.txt
```

适合加载 `sentence-transformers`、`torch`、`transformers` 等模型依赖，用于正式小模型 embedding/reranker。

## 三种运行模式

### 1. 离线冒烟测试

```bash
python run_agent.py --config config.smoke.json --query "large language model hallucination detection factuality evaluation" --output runs/smoke_single.json --no_llm --fallback_models --top_k 5
python evaluate_pasa.py --config config.smoke.json --input data/sample_eval.jsonl --output_dir runs/smoke_eval --no_llm --fallback_models --top_k 20
```

用途：确认代码、配置、评测脚本都能跑通。

### 2. 在线检索演示

```bash
python web_demo.py --config config.online.json --no_llm --fallback_models --port 8091
```

用途：前端展示真实学术 API 搜索结果。该模式不需要 GPU。

### 3. 正式比赛评测

```bash
cp config.example.yaml config.yaml
python evaluate_pasa.py --config config.yaml --input data/RealScholarQuery/test.jsonl --output_dir runs/pasa_realscholar
```

用途：接入真实 LLM、真实 embedding/reranker、真实测试集后计算指标。正式评测默认启用更大的候选池、更大的 LLM selector 队列、API 并发和缓存，目标是在召回率与 F1@20 之间取得更稳的平衡。

## 配置项说明

`llm`：
- `base_url`：OpenAI-compatible 接口地址，可以是 DeepSeek、Qwen API、vLLM、本地 Ollama 等。
- `api_key`：API Key，建议通过环境变量传入。
- `model`：大模型名称。
- `temperature`：建议 0.1 到 0.3，保证查询拆解稳定。

`small_models`：
- `embedding_model`：双塔向量模型，用于语义相似度打分。
- `reranker_model`：交叉编码器，用于候选论文精排。
- `device`：`cuda`、`cpu` 或 `auto`。
- `embedding_batch_size` / `reranker_batch_size`：显存不足时调小。

`retrieval`：
- `use_openalex` / `use_semantic_scholar` / `use_arxiv`：是否启用对应学术 API。
- `use_serper` / `serper_api_key`：是否启用 PaSa 风格 Serper/arXiv 搜索。配置 `SERPER_API_KEY` 后可通过搜索引擎补强 arXiv 页面召回。
- `general_index_path`：通用公开论文元数据索引路径，可接 JSONL 或 arXiv id 到标题的 JSON 映射；未配置时会自动寻找 `data/general_academic_index/`，并可兼容 PaSa 公开 paper database。
- `general_index_limit` / `local_bm25_top_k` / `local_dense_top_k`：控制本地 BM25 风格召回和概念召回的候选规模。
- `per_query`：每个子查询从 API 拉取的论文数。
- `max_candidates`：候选池最大论文数。
- `max_rounds`：最多检索轮数；第二轮会根据当前高分候选继续扩展检索式。
- `citation_expand_seeds`：用于引用扩展的种子论文数量。
- `citation_expand_limit`：引用扩展最多补充多少论文。
- `local_corpus_path`：离线样例或私有语料路径。
- `academic_only`：开启后过滤新闻、书籍条目、词典页面等明显非论文结果。
- `api_parallelism`：在线 API 并发数。
- `enable_api_cache`：是否缓存同进程重复 API 查询。

`ranking`：
- `api_weight`：外部 API 返回顺序和可信度权重。
- `bm25_weight`：关键词匹配权重。
- `embedding_weight`：语义相似度权重。
- `reranker_weight`：精排模型权重。
- `llm_verifier_weight`：大模型相关性验证权重。
- `llm_verify_top_n`：送入 LLM verifier 的候选论文数量。
- `llm_verifier_batch_size`：每次 LLM verifier 判断的论文数量。
- `meta_ranker_enabled`：启用轻量 meta-ranker，将来源覆盖、标题命中、方法/数据集词重合、引用和年份等特征融合进最终排序。

## 输出文件

单条查询：

```text
runs/single_result.json
```

批量评测：

```text
runs/pasa_realscholar/
├── predictions.jsonl
├── metrics.json
└── report.md
```

## 服务器建议

正式跑本地小模型，建议优先租：

```text
GPU: RTX 4090 24GB / A10 24GB / L20 48GB / A100 40GB+
CPU: 8 vCPU 或以上
RAM: 64GB 推荐，32GB 可调试
Disk: 100GB SSD 起步
Python: 3.10 或 3.11
CUDA: 选择平台预装 PyTorch + CUDA 镜像
```

如果只使用 DeepSeek、Qwen API 这类在线大模型，本地服务器不需要跑 LLM，24GB 显存已经足够跑 embedding/reranker；如果还想本地部署 7B LLM，建议 L20 48GB 或 A100 40GB 以上。
