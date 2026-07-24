# ZTF_prompt 使用教程

基于 LLM 的天文光变曲线分类工具。将 `npy`/`csv` 光变曲线数据转换为结构化 Markdown 分析报告，再通过 few-shot prompting 调用大模型进行分类。

---

## 目录结构

```
ZTF_prompt/
├── .env                 ← API 密钥配置
├── config.py            ← 全局配置
├── promt.py             ← 数据 → MD 分析报告
├── plot.py              ← 数据 → 光变曲线 PNG（用于多模态分类）
├── classify.py          ← MD → LLM → 分类结果
├── eval.py              ← 评估准确率（主动跑 API）
├── summary.py           ← 汇总已有结果 + 出图（零 API 成本）
├── run.sh               ← 一键全流程脚本
├── diag.py              ← 网络诊断工具
│
├── sources/             ← 生成文件（每个源一个子目录）
│   └── {id}/
│       ├── analysis.md      ← 结构化分析报告
│       ├── lightcurve.png   ← 光变曲线图（u=蓝, g=绿, r=红）
│       └── cutout.png       ← SDSS 宿主星系 cutout（可选）
├── index.json           ← 所有源的标签索引
├── results/             ← 分类结果 JSON
└── templates/           ← Few-shot exemplar 配置文件
    ├── fewshot.json          ← 默认 exemplar（TDE+SN 各 1 个）
    ├── fewshot_text.json     ← text mode 3-shot（TDE+SN 各 3 个）
    └── fewshot_boundary.json ← 边界样本（冲突信号，锚定决策边界）
    summary/                  ← 汇总 JSON + 混淆矩阵/分布图 PNG
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

## 第三步：生成光变曲线图（多模态模式）

```bash
# 单个源
python plot.py /path/to/source_flux.npy --source-id WFST_J101658

# 批量处理
python plot.py --batch /home/.../Flux/TDE/ --max 50

# 为 index.json 中所有源生成
python plot.py --all
```

输出 `sources/{id}/lightcurve.png`：u 波段蓝色、g 波段绿色、r 波段红色，带误差棒和峰值标注。

> 多模态分类需要 PNG 文件。`classify.py --mode multimodal` 会自动读取，无 PNG 时降级为 text。

---

## Few-Shot Exemplar 管理

分类时使用的 few-shot 示例可通过 `templates/fewshot*.json` 精确控制，替代默认的随机采样。

### 三种预设

| 文件 | 用途 | TDE 示例 | SN 示例 |
|------|------|---------|---------|
| `fewshot.json` | **默认** multimodal (1-shot) | `wmx_TDE_2024lhc`<br>Δ=-8.7, 432pts, 强 TDE | `ZTF19aaapnxn`<br>Δ=+6.2, 444pts, 教科书 SN |
| `fewshot_text.json` | text mode (3-shot) | 3 个：lhc / pvu / uvz | 3 个：aaapnxn / aagrdcs / aajxwnz |
| `fewshot_boundary.json` | 边界样本 | `wmx_TDE_2022arb`<br>Δ=**0.0 Flat**（无颜色信号） | `ZTF19aagmsrr`<br>Δ=**-176 Red→Blue**（颜色像 TDE） |

### 原理

- **教科书 exemplar**（default/text）：信号全覆盖，教会模型"长什么样"
- **Boundary exemplar**：信号冲突——2022arb 的 TDE 没有颜色演化，aagmsrr 的 SN 却有极端 Red→Blue。迫使模型做**信号权重推理**而非简单模式匹配，锚定决策边界

### 加载逻辑

```
--exemplar-set boundary  →  templates/fewshot_boundary.json
--exemplar-set text      →  templates/fewshot_text.json
(无 flag)                →  templates/fewshot.json
(文件不存在/空)           →  回退随机采样（旧行为）
```

### 自定义

```bash
# 编辑 exemplar 列表（增删改 ID 即可，列表长度即 n_shot）
vim templates/fewshot.json

# 创建新 set
cp templates/fewshot.json templates/fewshot_my_custom.json
python classify.py WFST_J101658 --exemplar-set my_custom

# 回退随机采样
mv templates/fewshot.json templates/fewshot.json.bak
```

---

## 第四步：分类

```bash
# 分类单个源（默认：3-shot，no CoT，text 模式）
python classify.py WFST_J101658

# 多模态模式（需要 lightcurve.png）
python classify.py WFST_J101658 --mode multimodal --model qwen3.6-chat

# 分类所有 unknown 源
python classify.py --all-unlabeled

# 强制重新分类（覆盖已有结果）
python classify.py WFST_J101658 --force

# 调整 few-shot 数量
python classify.py WFST_J101658 --n-shot 2          # 2-shot
python classify.py WFST_J101658 --n-shot 0           # zero-shot

# 启用 Chain-of-Thought（逐步推理）
python classify.py WFST_J101658 --cot

# CoT + few-shot 组合
python classify.py WFST_J101658 --cot --n-shot 2

# 换模型
python classify.py WFST_J101658 --model qwen3.6-reasoner

# 切换 exemplar set
python classify.py WFST_J101658 --exemplar-set boundary    # 边界样本
python classify.py WFST_J101658 --exemplar-set text        # text 3-shot
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
      {"name": "Color evolution", "weight": 0.4, "direction": "SN"},
      {"name": "Rise time", "weight": 0.2, "direction": "SN"}
    ]
  },
  "quality": {
    "overall": "medium",
    "flags": ["Rise phase sparsely sampled"]
  },
  "cot": false,
  "cot_reasoning": ""
}
```

> CoT 模式下，`cot_reasoning` 字段会保存 LLM 的逐步推理原文（Step 1-4）。

---

## 评估准确率

```bash
# 默认：每类抽 30 条测试，3-shot
python eval.py

# 自定义参数
python eval.py --test-size 10 --n-shot 2 --classes TDE,SN

# 四种 Few-Shot × CoT 组合
python eval.py --n-shot 0                   --test-size 10 --classes TDE,SN   # 纯物理规则
python eval.py --n-shot 2                   --test-size 10 --classes TDE,SN   # 物理 + 范例
python eval.py --n-shot 0  --cot            --test-size 10 --classes TDE,SN   # 物理 + CoT
python eval.py --n-shot 2  --cot            --test-size 10 --classes TDE,SN   # 物理 + 范例 + CoT

# 详细输出（显示每条源的预测）
python eval.py --verbose
```

输出包括：混淆矩阵、每类 Precision/Recall/F1、错误案例、低置信度案例。

---

## 结果汇总

`summary.py` 直接读取 `results/*.json`，**不调 API**，零 token 成本。

```bash
# 完整汇总（已知准确率 + 未知分布）
python summary.py

# 只看未知源分布
python summary.py --unknown-only

# 只看已知源准确率
python summary.py --known-only

# 详细列出每条
python summary.py --verbose

# 只看低置信度结果
python summary.py --min-conf low
```

输出包含三部分：
- **Overview**：总数、类别分布、置信度分布
- **Known Sources**：混淆矩阵、Precision/Recall/F1、Unsure 率、错误案例
- **Unknown Sources**：TDE/SN/Unsure 分布（带柱状图）、置信度分层

加 `--plot` 自动生成两张图：
- `{name}.png` — Known 混淆矩阵热力图（蓝阶，学术白底）
- `{name}_unknown.png` — Unknown 分类分布柱状图（按置信度分层）
```bash
python summary.py --plot                # 出图
python summary.py --exemplar-set boundary --plot  # boundary set 的图
```

---

## 一键全流程

```bash
# 默认：multimodal，default exemplar set
bash run.sh

# 换 exemplar set
bash run.sh --set boundary
bash run.sh --set text --mode text

# 只汇总已有结果（不跑分类）
bash run.sh --summary-only

# 跳过出图
bash run.sh --skip-plot
```

等价于手动执行：
1. `python plot.py --all`
2. `python classify.py --all-unlabeled --mode multimodal --model qwen3.6-chat`
3. `python summary.py --plot`

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

每个源生成的分析报告包含 4 个部分：

| 章节 | 内容 |
|------|------|
| §1 Source Metadata | 基本信息（点数、波段、峰值等） |
| §2 Derived Features | 计算特征（形态、颜色演化、分阶段统计、数据质量） |

| §4 Raw Light Curve | 完整原始数据表 |
| §5 Classification Protocol | 给 LLM 的分类指令 |

> 注意：调用 API 时默认去掉 §3（原始数据表）以节省 token。System Prompt 以**颜色演化**和**上升形态**（凹形= TDE, 凸形= SN）为主要判据，上升时长和总跨度作为辅助参考。

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `sources/{id}/analysis.md` | 完整分析报告 |
| `sources/{id}/lightcurve.png` | 光变曲线图（多模态用） |
| `results/{id}.json` | 分类结果（含置信度和推理链） |
| `eval_report.json` | 评估报告（运行 `eval.py` 后生成） |
| `index.json` | 所有源的标签和元信息索引 |
| `templates/fewshot*.json` | Few-shot exemplar 配置文件 |
| `summary/*.json` | 汇总数据（Overview + Known + Unknown） |
| `summary/*.png` | 混淆矩阵热力图 + 未知源分布图 |
| `run.sh` | 一键全流程脚本 |

---

## 注意事项

1. **API 调用较慢**：USTC 代理每次约 25-60 秒，分类一条源约 1 分钟。
2. **不要用后台模式**：`classify.py` 必须前台运行（后台进程 SSL 连接有问题）。
3. **mock 源不参与 few-shot**：`synth_flux_*` 是合成数据，已从 `index.json` 移除。
4. **多模态模式**：需要 `sources/{id}/lightcurve.png`（用 `plot.py` 生成）。USTC 代理的 deepseek 模型不支持视觉，需指定 `--model qwen3.6-chat`。
5. **System Prompt 物理判据**：① 颜色演化 (g−r) → ② 上升形态 (凹/凸) → ③ 衰减形状 → ④ 数据质量。TDE 凹形上升（回落驱动），SN 凸形上升（激波冷却）。
6. **图层颜色**：u=蓝 ▲、g=绿 ●、r=红 ■。