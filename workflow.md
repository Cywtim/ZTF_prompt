┌─────────────────────────────────────────────────────────────┐
│                       ZTF_prompt 流水线                       │
└─────────────────────────────────────────────────────────────┘
  原始数据 (npy/csv)
  /data/TS/Flux/{TDE,SN,...}/
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