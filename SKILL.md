---
name: llm-lightcurve-prompt
description: Use when classifying astronomical light curves via LLM prompt engineering (ZTF_prompt). Converts npy/csv to structured Markdown analysis reports, then calls LLM API for few-shot classification of TDE/SN/Others/AGN. Use for WFST or ZTF sources when the user says "prompt engineering", "LLM classify", or references the ZTF_prompt project.
version: 1.0.0
author: Fairy
license: MIT
metadata:
  hermes:
    tags: [astronomy, TDE, classification, prompt-engineering, LLM, ZTF, WFST]
    related_skills: [tde-research-wiki-query]
---

# LLM Light Curve Prompt Classification

## Overview

A prompt-engineering approach to astronomical transient classification. Instead of training GPT-2+LoRA models, this pipeline converts light curve data into structured Markdown analysis reports, then uses few-shot prompting to have an LLM classify transients as TDE/SN/Others/AGN.

## Project Location

```
/home/cyan/AppData/VScode/TDeck/ZTF_prompt/
```

Git repo: `https://github.com/Cywtim/ZTF_prompt.git`

## When to Use

- User wants to classify WFST/ZTF light curves via LLM (not trained models)
- User mentions "prompt engineering", "LLM classify", "ZTF_prompt"
- User wants explainable classification results with reasoning chains
- User has new data sources and wants quick classification without retraining

## Architecture

```
npy/csv data  →  promt.py  →  sources/{id}/analysis.md  →  classify.py  →  LLM API  →  results/{id}.json
                     ↓                                                         ↓
                index.json                                              (label, confidence, reasoning)
```

### Analysis.md structure (5 sections)

| Section | Content | Used by LLM? |
|---------|---------|:------------:|
| 1. Metadata | Source ID, MJD range, N pts, bands, peak | Yes |
| 2. Derived Features | Morphology, color evolution, per-phase summary, quality flags | Yes (primary) |
| 3. Predictive Features | TDE vs SN comparison table | Yes |
| 4. Raw Light Curve | Full data table (MJD, band, flux, err, phase, g-r) | No (stripped for API) |
| 5. Classification Protocol | LLM instructions, TDE/SN knowledge, output format | Yes (system prompt) |

### Weight control mechanism

The prompt balances derived features vs raw data via:
- **Position**: Features (Sections 2-3) first → LLM forms hypothesis from features
- **Confidence flags** (Section 2.4): HIGH/MEDIUM/LOW per feature → LLM naturally discounts noisy features
- **Explicit protocol** (Section 5): "Start from Derived Features, verify with Raw Data"
- **Raw data** (Section 4): Stripped for API calls, kept in local MD for human review

## Quick Reference Commands

```bash
cd /home/cyan/AppData/VScode/TDeck/ZTF_prompt

# Generate MD from data
python promt.py data/WFST_J101658.csv --label unknown
python promt.py --batch .../Flux/TDE/ --label TDE --max 50

# Classify
python classify.py WFST_J101658
python classify.py --all-unlabeled

# View results
python classify.py --results WFST_J101658

# Evaluate accuracy
python eval.py --test-size 30

# Manage labels
python promt.py --stats
python promt.py --relabel WFST_J101658 TDE
```

## Key Files

| File | Role |
|------|------|
| `config.py` | API config, paths, classification params |
| `promt.py` | npy/csv → analysis.md + index.json |
| `classify.py` | MD → few-shot prompt → LLM API → results JSON |
| `eval.py` | Hold-out evaluation with accuracy/F1/confusion matrix |
| `index.json` | Label index (not tracked in git) |
| `.env` | API key (not tracked in git) |
| `README.md` | Full tutorial |

## Configuration

`.env` file (not committed):
```
LLM_API_KEY=***
LLM_MODEL=deepseek-v4-pro
```

API: USTC proxy at `https://api.llm.ustc.edu.cn/v1`, OpenAI-compatible.

## Common Pitfalls

1. **Shared httpx client with connection pooling (NOT fresh per call)**: `_get_client()` creates a SINGLE global httpx Client with `keepalive_connections=5, max_connections=10`. Creating a fresh client per API call causes connection storms on proxy servers (USTC rate-limits new TCP+TLS handshakes). The shared client reuses keep-alive connections across calls. See `classify.py` for implementation.

2. **Retry backoff strategy**: 3 retries with exponential backoff: `5s → 10s → 20s` (was `1s → 2s → 4s`). Proxy servers need time to recover from congestion; short waits are useless. `call_api()` in `classify.py`.

3. **Inter-request delay in eval.py**: 1.5s `time.sleep` between successive eval sources. Without this, 10+ rapid API calls look like a DDoS and trigger proxy rate-limiting. `eval.py` line ~58.

4. **Synth mock sources**: `synth_flux_*` are synthetic TDE, should be excluded from few-shot pool. `index.json` already has them removed.

5. **Raw data too large**: Section 4 is auto-stripped for API calls (saves ~70% tokens). The API proxy drops connections for >20K char payloads.

6. **USTC proxy returns content in reasoning_content**: Both `qwen3.6-reasoner` and `deepseek-v4-pro` responses through the USTC proxy use `reasoning_content` field with `content=None`. `call_api()` handles this: `text = msg.content or getattr(msg, \"reasoning_content\", \"\") or \"\"`.

7. **Slow API**: ~25-60s per call. Plan for ~1 min per source classification. Eval at scale (40 sources) takes ~15-20 minutes with inter-request delays.

8. **Few-shot pool must exclude test source**: `sample_few_shot()` accepts `exclude` set. `eval.py` automatically excludes the entire test set from few-shot sampling.

9. **API key management**: Never commit `.env`. Use `.env.example` as template.

10. **Large sources are NOT a problem**: Sources with 500+ data points work fine because Section 4 (raw data) is auto-stripped for API calls. The few-shot prompt size is dominated by Sections 1-3 which are fixed-length regardless of data size.

11. **Eval accuracy baseline**: DeepSeek-v4-flash on binary TDE/SN classification achieves ~50% accuracy (10 sources, 2-shot). All errors are medium-confidence (score 0.60-0.70), suggesting a score<0.75 rejection threshold would improve precision at the cost of recall. Expect 0 connection errors after the shared-client fix.

## Results JSON Schema

```json
{
  "source_id": "...",
  "classification": {"label": "TDE", "confidence": "high", "score": 0.85},
  "reasoning": {
    "summary": "one-sentence verdict",
    "feature_based": "2-3 sentence feature analysis",
    "raw_audit": "1-2 sentence data quality check",
    "indicators": [
      {"name": "Color evolution", "value": "red-to-blue", "weight": 0.4, "direction": "TDE", "note": "..."}
    ]
  },
  "quality": {"overall": "high", "flags": [...]},
  "few_shot": [{"id": "...", "label": "TDE"}, ...],
  "tokens": {"prompt": 5000, "completion": 800, "total": 5800}
}
```

## Data Flow for New Sources

1. User provides npy/csv → `promt.py` generates `sources/{id}/analysis.md` and updates `index.json`
2. `classify.py` samples few-shot from labeled sources in `index.json`, builds prompt, calls API
3. Result saved to `results/{id}.json` with full reasoning chain

## Verification Checklist

- [ ] `.env` has valid API key
- [ ] `index.json` has labeled TDE and SN sources (at least 5 each)
- [ ] `classify.py` runs in foreground, not background
- [ ] Test with `python classify.py WFST_J101658` before batch runs
- [ ] Check results JSON has valid `classification.label` (not "?" or error)