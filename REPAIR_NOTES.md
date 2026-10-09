# 学生代码修复说明

2026-10-09：将 `../code-rethinking-mas-defense-main/` 的完整核心执行链移植到此目录，保留学生的离线消息分类功能与百炼入口。没有修改主代码、历史数据或历史结果。

同日复查补充修正：实际异步直连请求与同步请求统一使用百炼参数适配，补充 SDK 请求及 HTTP 请求体测试；分析器要求 summary 身份和样本数量一致，含无效攻击的条件不会标为完成；所有输出须在本地 tokenizer 输入目录之外。百炼思考开关通过 Python SDK 的 `extra_body` 传入，格式核对自[官方文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)。HTTP 测试使用拦截传输，不访问真实服务。

## 已修改的行为

- 攻击提示恢复主代码逻辑，Overt 首轮包含原问题与目标答案。
- 使用生成、选择、提交分离的消息状态；默认生成四个候选，并用下一轮单攻击者及联合攻击者反事实结果选取候选。
- 加入 embedding 支撑集、攻击约束和 Wrapper payload token 预算检查。无可行候选或答案解析失败会记录到 checkpoint，并以不完整运行退出；不将其算作成功攻击样本。
- 拓扑采用主代码的有向开链、传播检查、逐题随机种子和攻击者位置策略。邻居消息不再包含隐式自身消息。
- 置信度使用实际生成文本的 logprobs；缺少必要数据时明确失败。`exact_full_vocab` 与 `top_logprobs_tail_bucket` 是不同协议，后者仅提供近似熵。
- 结果记录代码、数据、配置身份、候选诊断和每轮指标；`--resume` 校验身份。默认输出和 checkpoint 位于 `results/repaired/`，拒绝覆盖非空已有输出或输入文件。
- GSM8K 使用数值解析与归一化。BBH 支持显式两至四选项适配，不能直接评估任意自由文本答案；不支持的样本在模型调用前报错。
- 分析器将缺失预测算错，显示无效攻击与不完整状态，拒绝静默跳过损坏 JSON。没有配套 summary 的历史日志显示为未验证。
- 保留 `--classify_messages`，作为生成后的分类分析；它不参与消息过滤或投票。

主代码仍将邻居推理截取为 300 字符，本次保持该行为。实际攻击传播应结合 `agent_histories` 中的下一轮 prompt 检查。

## 安装与离线检查

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B evaluate.py --help
```

离线测试使用假模型和假 embedding，覆盖攻击选择、状态隔离、传播、置信度、解析、失败记录、身份恢复，以及百炼请求格式和分类兼容性；不访问推理服务、不下载模型。它们证明代码行为，不能证明真实模型上的准确率已复现。

## 百炼诊断

从本目录运行（Windows PowerShell）：

```powershell
.\run_bailian_reproduction.ps1 -PayloadTokenizerPath 'D:\models\requested-model-tokenizer' -NSamples 20 -Seed 2026
```

示例 tokenizer 路径需要替换为实际目录，其 tokenizer 应对应 `-Model` 指定的模型。脚本默认数据为 `datasets/raw_backup/mmlu_test_full.jsonl`，也可传 `-DataFile`；`-EmbeddingModel` 可指定已有本地 embedding 目录。API key 从 `OPENAI_API_KEY` 或隐藏输入读取，不写入命令参数或结果文件。每次创建带时间戳的独立结果目录，执行 11 个条件，失败即停止；候选和反事实调用会增加请求量。

该入口显式使用 `--top_logprobs 5`、近似熵、完整拓扑和固定攻击者。客户端不再全局截断 logprobs 数量。百炼请求的 `enable_thinking` 放在顶层，vLLM Qwen 请求使用 `chat_template_kwargs`。

Wrapper 默认通过服务 `/tokenize` 计数。上述云入口改用显式本地 tokenizer，不会自动下载，日志记录 tokenizer 文件哈希及 `local_diagnostic_unverified_serving_identity`。本地计数无法证明与云端服务 tokenizer 完全相同，因此拒绝与正式身份校验标志组合。正式复现应使用已验证的服务 tokenizer、embedding/backend manifests，以及论文对应的模型、数据、采样和熵协议。

先用相同模型、数据、题目顺序与配置对比 clean 和攻击条件，检查攻击可行率、良性答案翻转及实际传播，再扩大样本。准确率是否仍然偏高需要真实模型运行验证，不能预设修复必须降低准确率。

## 结果检查

```bash
python analyze.py results/repaired/<run>/mmlu_s2026_clean.jsonl
python analyze.py results/repaired/<run>/*.jsonl --summary
```

标准 `<结果文件名去掉 .jsonl>.summary.json` 提供 `complete`、`requested_samples`、`failed_samples`。若自定义 `--summary_file`，需另外核查该 summary；分析器自动发现标准命名的 summary。不完整运行的准确率只描述已完成子集，不能当作全条件结果。
