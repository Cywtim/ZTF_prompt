┌─────────────────────────────────────────────────────────────┐
│                       ZTF_prompt 流水线                       │
└─────────────────────────────────────────────────────────────┘
  原始数据 (npy/csv)
  /data/TS/Flux/{TDE,SN,...}/
  │
  │ 
  ▼
┌────────────────┐
│    Mapfilter   │
└────────────────┘
  │
  │  python promt.py --batch <dir> --label TDE
  ▼
┌──────────────────────┐
│      promt.py        │
│                      │
│  npy/csv → MD 转换    │
│  - §1 元数据          │
│  - §2 导出特征        │
│  - §3 预测特征        │
│  - §4 原始数据(剥离)   │
│  - §5 分类协议        │
│  - §6 绘制光变曲线     │
│  - §5 下载阿拉丁图像   │
│                      │
│  → sources/<id>/     │
│     analysis.md      │
│  → index.json(标签)   │
└──────────────────────┘
  │
  │  python classify.py <source_id> --n-shot 2
  │  python eval.py --test-size 5 --n-shot 2 --classes TDE,SN
  ▼
┌──────────────────────┐      ┌─────────────────────┐
│   classify.py        │      │     config.py       │
│                      │      │                     │
│ 1. 读 index.json     │◄─────│ CLASSES = [TDE,SN]  │
│ 2. 随机抽 few-shot    │      │ API_KEY / MODEL     │
│ 3. _make_system_     │      └─────────────────────┘
│    prompt() ← 动态    │
│    生成物理规则        │
│ 4. 拼装 MD prompt     │
│ 5. 调 API (OpenAI)   │
│ 6. 解析 JSON 结果     │
│ 7. →results/<id>.json│
└──────────────────────┘
  │
  │  eval.py 包装层
  │  - 留出法：随机分 test/few-shot
  │  - 批量调 classify_one()
  │  - 输出: accuracy / F1 / confusion
  ▼
┌──────────────────────┐
│     输出 & 评估        │
│                      │
│ results/<id>.json    │  ← 单源结果
│ eval_report.json     │  ← 评估报告
│                      │
│ 指标:                 │
│  Accuracy/Precision  │
│  Recall / F1         │
│  Confusion Matrix    │
│  Token 消耗统计       │
└──────────────────────┘




Step 0: Shape Gate
  ├─ 平台期 > 50d       → SN（不管颜色）
  ├─ 多峰               → 不是单次 TDE
  ├─ 对称爆发           → 不是 TDE
  └─ 无结构             → Others
       ↓ 未触发
Step 1: 颜色 × WISE × 形状 矩阵
 光学颜色    W1-W2          形状        → 结果
──────────────────────────────────────────────────────
 TDE zone    < 0.5          concave     TDE (strong)
 TDE zone    < 0.5          convex      Unsure→TDE
 TDE zone    ≥ 0.8          any         AGN (strong)
 TDE zone    [0.5, 0.8)     concave     TDE (weak)
 TDE zone    [0.5, 0.8)     convex      SN (shape wins)
 TDE zone    N/A            concave     TDE (moderate)
 TDE zone    N/A            convex      Unsure→TDE
 SN          < 0.5          any         Unsure (IR-TDE vs opt-SN)
 SN          ≥ 0.8          any         AGN (weak)
 SN          [0.5,0.8)/N/A  convex      SN (strong)
 SN          [0.5,0.8)/N/A  concave     Unsure→SN
 None        ≥ 0.8          concave     AGN
 None        ≥ 0.8          convex      Others
 None        < 0.5          concave     TDE (weak)
 None        < 0.5          convex      SN (weak)
 None        [0.5,0.8)/N/A  concave     TDE? (very weak)
 None        [0.5,0.8)/N/A  convex      SN? (very weak)
 None        [0.5,0.8)/N/A  unclear     Unsure
 None        N/A            unclear     Unsure
Step 2: Tiebreakers (Gaia, 时标, host)