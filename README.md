# ZTF_prompt 使用教程

基于 LLM 的天文光变曲线分类工具。将 `npy`/`csv` 光变曲线数据转换为结构化 Markdown 分析报告，再通过 few-shot prompting 调用大模型进行分类。

---

## 目录结构

```
ZTF_prompt/
├── .env                 ← API 密钥配置
├── config.py            ← 全局配置
├── promt.py             ← 数据 → MD 分析报告
├── classify.py          ← MD → LLM → 分类结果
├── eval.py              ← 评估准确率
│
├── sources/             ← 生成的 MD 文件（每个源一个目录）
├── index.json           ← 所有源的标签索引
└── results/             ← 分类结果 JSON
```

---

## 第一步：配置

编辑 `.env` 文件，填入 API 密钥：

```
LLM_API_KEY=***  LLM_MODEL=deepseek-v4-pro
```

- `LLM_API_KEY`：API 密钥（必需）
- `LLM_MODEL`：模型名称（默认 `deepseek-v4-pro`）

---

## 第二步：生成标注数据（few-shot 池）

分类需要已知标签的源作为示例。先生成 TDE 和 SN 的 MD 文件：

```bash
# TDE（真实源，不含 synth mock）
python promt.py --batch /home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/TDE/ --label TDE

# SN
python promt.py --batch /home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/SN/ --label SN
```

> 注意：`--batch` 会处理目录下所有 `.npy` 和 `.csv` 文件。如果只想处理少量，用单文件模式。

---

## 第三步：生成待分类源

```bash
# 单个文件
python promt.py data/WFST_J101658.csv --label unknown

# 指定 source_id（用于后续引用）
python promt.py data/WFST_J143207.csv --label unknown --source-id WFST_J143207

# 强制覆盖已有文件
python promt.py data/xxx.npy --label unknown --force
```

---

## 第四步：分类

```bash
# 分类单个源
python classify.py WFST_J101658

# 分类所有 unknown 源
python classify.py --all-unlabeled

# 强制重新分类（覆盖已有结果）
python classify.py WFST_J101658 --force

# 调整 few-shot 数量（每类示例数）
python classify.py WFST_J101658 --n-shot 3

# 换模型
python classify.py WFST_J101658 --model qwen3.6-reasoner
```

---

## 第五步：查看结果

```bash
# 查看完整结果（分类 + 置信度 + 每个判断指标）
python classify.py --results WFST_J101658

# 直接读 JSON
cat results/WFST_J101658.json
```

结果 JSON 结构：

```json
{
  "classification": {
    "label": "SN",
    "confidence": "medium",
    "score": 0.60
  },
  "reasoning": {
    "summary": "...",
    "indicators": [
      {"name": "Color evolution", "weight": 0.4, "direction": "SN", "note": "..."},
      {"name": "Rise time", "weight": 0.2, "direction": "SN"}
    ]
  },
  "quality": {
    "overall": "medium",
    "flags": ["Rise phase sparsely sampled"]
  }
}
```

---

## 评估准确率

```bash
# 默认：每类抽 30 条测试，3-shot
python eval.py

# 自定义
python eval.py --test-size 50 --n-shot 3 --classes TDE,SN

# 详细输出（显示每条源的预测）
python eval.py --verbose
```

输出包括：混淆矩阵、每类 Precision/Recall/F1、错误案例、低置信度案例。

---

## 管理标签

```bash
# 查看统计
python promt.py --stats

# 列出某类所有源
python promt.py --list TDE

# 改标签
python promt.py --relabel WFST_J101658 TDE
```

---

## 完整工作流示例

```bash
# 1. 配置
vim .env    # 填入 API key

# 2. 生成 few-shot 池（只需做一次）
python promt.py --batch .../Flux/TDE/ --label TDE
python promt.py --batch .../Flux/SN/  --label SN

# 3. 生成新数据
python promt.py data/my_new_source.csv --label unknown

# 4. 分类
python classify.py --all-unlabeled

# 5. 查看
python classify.py --results my_new_source
```

---

## MD 分析报告结构

每个源生成的分析报告包含 5 个部分：

| 章节 | 内容 |
|------|------|
| §1 Source Metadata | 基本信息（点数、波段、峰值等） |
| §2 Derived Features | 计算特征（形态、颜色演化、分阶段统计、数据质量） |
| §3 Predictive Features | TDE vs SN 对比表 |
| §4 Raw Light Curve | 完整原始数据表 |
| §5 Classification Protocol | 给 LLM 的分类指令 |

> 注意：调用 API 时默认去掉 §4（原始数据表）以节省 token。

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `sources/{id}/analysis.md` | 完整分析报告 |
| `results/{id}.json` | 分类结果（含置信度和推理链） |
| `eval_report.json` | 评估报告（运行 `eval.py` 后生成） |
| `index.json` | 所有源的标签和元信息索引 |

---

## 注意事项

1. **API 调用较慢**：USTC 代理每次约 25-60 秒，分类一条源约 1 分钟。
2. **不要用后台模式**：`classify.py` 必须前台运行（后台进程 SSL 连接有问题）。
3. **mock 源不参与 few-shot**：`synth_flux_*` 是合成数据，已从 `index.json` 移除。
4. **多模态模式**：`--mode multimodal` 需要 `sources/{id}/lightcurve.png` 存在，否则自动降级为 text。