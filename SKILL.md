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

A prompt-engineering approach to astronomical transient classification. Instead of training GPT-2+LoRA models, this pipeline converts light curve data into structured Markdown analysis reports, then uses few-shot prompting to have an LLM classify transients as TDE/SN. Two independent, composable features control the classification strategy:

| Feature | Flag | Effect |
|---------|------|--------|
| **Few-Shot** | `--n-shot N` | N labeled examples per class prepended to prompt (0 = zero-shot) |
| **CoT** | `--cot` | LLM must reason through physical discriminators step-by-step before outputting JSON |

All four combinations are valid and independently testable:

```
              n_shot=0              n_shot=2
          (zero-shot)            (few-shot)
         ┌──────────────┬──────────────────────┐
cot=False│ 纯物理规则     │ 物理规则 + 范例        │  ← original default
         ├──────────────┼──────────────────────┤
cot=True │ 物理规则 + CoT │ 物理规则 + 范例 + CoT  │  ← strongest
         └──────────────┴──────────────────────┘
```

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
                     ↓                                          ↓                        ↓
                index.json                           _make_system_prompt()      (label, confidence,
                                                    reads config.CLASSES →      reasoning, indicators)
                                                    injects physics rules
```

**System prompt is dynamically generated** by `_make_system_prompt()` which reads `config.CLASSES` to auto-sync class names, then injects physics rules (color evolution, timescale, decline shape, data quality, decision logic). Adding a class = editing `config.py` + adding rules in `_make_system_prompt()`.

### Analysis.md structure (4 sections)

| Section | Content | Used by LLM? |
|---------|---------|:------------:|
| 1. Metadata | Source ID, MJD range, N pts, bands, peak | Yes |
| 2. Derived Features | Morphology, color evolution, per-phase summary, quality flags | Yes (primary) |
| 3. Raw Light Curve | Full data table (MJD, band, flux, err, phase, g-r) | No (stripped for API) |
| 4. Classification Protocol | LLM instructions, TDE/SN knowledge, output format | Yes (system prompt) |

> **Section 3 (Predictive Features) was removed** (Jul 2026): its auto-generated hints were physically wrong and conflicted with the authoritative system prompt. The LLM now relies entirely on the physics rules in `_make_system_prompt()`. `read_md()` strips §3 by splitting on `"## Section 3:"` and keeping only §4 (now renumbered as §3).

### Weight control mechanism

The prompt balances derived features vs raw data via:
- **Position**: Features (Sections 2-3) first → LLM forms hypothesis from features
- **Confidence flags** (Section 2.4): HIGH/MEDIUM/LOW per feature → LLM naturally discounts noisy features
- **Explicit protocol** (Section 5): "Start from Derived Features, verify with Raw Data"
- **Raw data** (Section 4): Stripped for API calls, kept in local MD for human review

## Quick Reference Commands

```bash
cd /home/cyan/AppData/VScode/TDeck/ZTF_prompt

# Verify data completeness
python promt.py --stats                           # count registered sources
ls data/TS/Flux/TDE/*.npy | wc -l                 # count raw files

# Generate MD from data
python promt.py data/WFST_J101658.csv --label unknown
python promt.py --batch .../Flux/TDE/ --label TDE --max 50

# Generate light curve plot (required for --mode multimodal)
python plot.py data/WFST_J101658.csv
python plot.py --batch .../Flux/TDE/ --max 10
python plot.py --all                        # regenerate for all sources in index.json

# Classify — four modes
python classify.py WFST_J101658                              # default: 3-shot, no CoT
python classify.py WFST_J101658 --n-shot 0                   # zero-shot
python classify.py WFST_J101658 --cot                        # CoT reasoning
python classify.py WFST_J101658 --cot --n-shot 2             # CoT + few-shot (strongest)

# View results
python classify.py --results WFST_J101658

# Evaluate accuracy (MUST bypass local proxy)
NO_PROXY=api.llm.ustc.edu.cn python eval.py --n-shot 0                   --test-size 10 --classes TDE,SN
NO_PROXY=api.llm.ustc.edu.cn python eval.py --n-shot 2                   --test-size 10 --classes TDE,SN
NO_PROXY=api.llm.ustc.edu.cn python eval.py --n-shot 0  --cot            --test-size 10 --classes TDE,SN
NO_PROXY=api.llm.ustc.edu.cn python eval.py --n-shot 2  --cot            --test-size 10 --classes TDE,SN

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
| `plot.py` | npy/csv → sources/{id}/lightcurve.png (u=blue, g=green, r=red) |
| `cutout.py` | radec.txt → sources/{id}/cutout.png (SDSS/DSS survey cutout) |
| `extract_radec.py` | Multi-source coordinate extraction → sources/{id}/radec.txt |
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

## Unsure Category with Preference Direction

The pipeline supports an `Unsure` meta-class for cases where the LLM is genuinely torn between TDE and SN. It is distinct from forced classification — it's a deliberate "abstain with direction" vote.

### How it works

`config.CLASSES = ["TDE", "SN", "Unsure"]`. The `Unsure` class has **no ground-truth labels** in `index.json` — it only exists as a valid output for the LLM. It is excluded from few-shot sampling automatically (no labeled examples).

**Decision Logic Rule 5** (in system prompt):
```
5. WHEN TO USE 'Unsure': if 2+ physical indicators point in opposite
   directions (e.g. color=TDE but timescale=strong SN), classify as 'Unsure'.
   Set 'unsure_preference' to the class you slightly favor.
   Unsure is for genuinely torn cases — NOT a default when lazy.
```

**JSON output when Unsure:**
```json
{"classification": {"label": "Unsure", "confidence": "low", "score": 0.35,
                     "unsure_preference": "TDE"}, ...}
```

When `label` is TDE or SN, `unsure_preference` should be `null`.

**Eval handling:** Unsure predictions do NOT count toward accuracy (not a wrong answer, but not a hit either). They are reported separately:
```
Unsure rate: 3/20 (15%)
    1x true=TDE→TDE
    1x true=TDE→SN
    1x true=SN→SN
    Preference: SN: 2, TDE: 1
```

### Implementation notes

- `_make_system_prompt()` computes `real_str = "TDE|SN"` (filters out Unsure) for the `direction` field in indicators — indicators always point to physical classes, never to Unsure
- The `unsure_preference` field uses `real_str|null` in the JSON template
- `sample_few_shot()` skips classes with no labeled examples automatically
- In `eval.py`, the Unsure block is inserted before the error analysis section; `wrong` list excludes Unsure entries to avoid double-counting

## Common Pitfalls

### Connection Errors — Root Cause & Fix

**The real cause of `APIConnectionError` is the local HTTP proxy (`127.0.0.1:7890`) intercepting USTC API's SSL traffic.** USTC's HTTP/2 implementation has a bug (ALPN negotiation triggers `SSL: UNEXPECTED_EOF_WHILE_READING`). The proxy routes the request through itself, the TLS handshake breaks, and OpenAI's httpx client reports "Connection error."

**Fix:** Bypass the proxy for USTC API:
```bash
NO_PROXY=api.llm.ustc.edu.cn python eval.py --test-size 10 --n-shot 2 --classes TDE,SN
```

**What did NOT work and was rolled back:**
- Shared singleton `_get_client()` with custom `httpx.Client(http2=False)` → broke httpx compatibility (user's httpx too old for `http2` kwarg)
- Connection pooling / keep-alive → not the cause; proxy bypass was needed
- Extended retry backoff (5s→10s→20s) → useless when proxy blocks every connection
- **Current `classify.py` uses simple `_get_client()` that creates fresh OpenAI client each call** — this is fine when proxy is bypassed

**Layered diagnostic approach (`diag.py`):** When debugging connection issues, test each layer independently:
1. DNS resolution (`socket.getaddrinfo`)
2. TCP connectivity (`socket.create_connection`)
3. TLS handshake (`ssl.create_default_context`)
4. HTTP request (`httpx.get`, `urllib3`, `requests`)
5. OpenAI client call
This isolates the failing layer without guesswork.

### System Prompt — Physics Rules Are Essential

The original prompt was *"You are an astronomical transient classifier. Output ONLY JSON."* — zero physics knowledge, yielding ~40% accuracy.

The current prompt (v2, Jul 2026) injects physics rules as the system message. Key refinements from eval-driven iteration:

```
## Physical Discriminators (check in priority order)

### 1. Color Evolution (g − r) — STRONGEST signal
- delta_g-r > 10 uJy (strong red→blue evolution): STRONG TDE indicator
- delta_g-r < 5 uJy (flat/mild evolution): typical SN
- Color staying RED throughout: favors steady sources, flag as uncertain
- Caveat: some SN subtypes (IIb, IIn, SLSN-I) can also show red→blue
  color evolution, though less common. If color is the ONLY TDE signal
  and rise shape / duration point to SN, consider SN with medium confidence.

### 2. Rise Morphology — SHAPE matters more than duration

Concave rise (decelerating, d²F/dt² < 0): → STRONG TDE indicator
  Physical: fallback accretion — most bound debris returns first.

Convex rise (accelerating, d²F/dt² > 0): → favors SN
  Physical: shock cooling — optical flux accelerates as temperature drops.

Rise duration as secondary check:
  < 10 days  → favors SN (shock-cooling timescale)
  10–60 days → typical TDE (fallback timescale)
  > 60 days  → check shape

### 3. Decline Shape
- Steep power-law decline (>1 uJy/d sustained): favors TDE
- Plateau or very slow decline (<0.1 uJy/d): favors SN

### 4. Data Quality
- Total points < 10: inherently LOW confidence
- Single-band only: no color information

## Decision Logic (refined Jul 2026)
1. Check color evolution FIRST. If clear, follow it.
2. Check rise SHAPE: concave = TDE; convex = SN.
   Use rise duration + total span as secondary tiebreakers.
3. If conflicting signals: flag as medium confidence and explain why.
4. WHEN TO USE 'Unsure': if 2+ physical indicators point in opposite
   directions (e.g. color=TDE but rise shape=SN), classify as 'Unsure'.
   Set 'unsure_preference' to the class you slightly favor.
   Unsure is for genuinely torn cases — prefer committing if any
   clear signal exists.
```

**Adaptive Thresholds:** For bright sources (peak >500 μJy, e.g., WFST), the prompt auto-injects relative thresholds:
- Significant color evolution: delta_g-r > 5% of peak (rather than absolute 10 μJy)
- Significant decline: >0.5%/d of peak
- Flat color: delta_g-r < 2% of peak

This is done by `build_prompt()` checking `_peak_flux(target_id)` and prepending a flux-scale note.

**SN Subtype Caveat (Jul 2026):** Added after tracing 2 high-confidence SN→TDE misclassifications. Short-span SNe (IIb/IIn/SLSN-I) with strong color evolution are physically indistinguishable from TDEs using photometry alone. The caveat tells the LLM to flag such cases as medium-confidence SN rather than high-confidence TDE.

This is built by `_make_system_prompt()` in `classify.py`.

### Dynamic Class Names from config.py

`_make_system_prompt()` reads `config.CLASSES` to auto-generate class lists in the system prompt and JSON template. Adding a new class requires only:

1. `config.py`: `CLASSES = ["TDE", "SN", "AGN"]`
2. `classify.py` `_make_system_prompt()`: add physics rules for the new class

All class names in the prompt body and JSON schema auto-sync. No more grep-and-replace across the codebase.

### Accuracy Evolution (Jul 2026)

Refinement history for binary TDE/SN (20 sources, 1-shot multimodal unless noted):

| Iteration | Accuracy | Unsure | TDE Recall | SN Recall | Key Change |
|-----------|:--:|:--:|:--:|:--:|------|
| Text 3-shot baseline | 33% | 47% | 33% | 33% | Original prompt before refinements |
| Multimodal 1-shot | 60% | 30% | 70% | 50% | Added vision (lightcurve.png) |
| **Refined decision logic** | **75%** | **10%** | **90%** | **60%** | Color-first → shape → conflict medium → 2+ conflict Unsure; SN subtype caveat |

**Key insight:** The jump from 60%→75% came from restoring the clean decision tree (removing one-conflict=Unsure) and adding the SN subtype caveat. Remaining errors are short-span SNe (55-95d) that show TDE-like color evolution — a fundamental ambiguity in photometry-only classification.

### CoT (Chain-of-Thought) — HARMFUL for this task

**CoT + multimodal 1-shot produced 60% accuracy with 0% Unsure but 8 errors, including 3 high-confidence (0.90) SN→TDE misclassifications.** CoT reasoning amplifies the LLM's overconfidence — it builds a coherent narrative for a wrong answer instead of admitting uncertainty. **Non-CoT multimodal is the current best configuration.**

### Few-Shot Sampling — Strategic (not random)

**Current:** `sample_few_shot()` uses `random.sample(pool, n)` — pure random, no stratification. This produces wild variance: a 55d-span SN target may get a 2300d-span SN example.

**Problem:** SN span distribution is bimodal (short <200d and ultra-long >1000d). Random sampling often picks ultra-long SN examples, giving the LLM no reference for "what a short-span SN looks like" — leading to SN→TDE errors on short SNe.

**Planned fix:** Span-stratified sampling (§4 layers) or feature-vector similarity matching. Full analysis in `references/few-shot-strategic-sampling.md`.

### CoT + Few-Shot Architecture

The system prompt is dynamically generated by `_make_system_prompt(cot=False)`. When `cot=True`, it adds a "Reasoning Protocol" section instructing the LLM to reason through Step 1 (Color Evolution) → Step 2 (Rise Morphology: concave vs convex) → Step 3 (Decline Shape) → Step 4 (Synthesis) before outputting JSON.

**⚠️ CoT is NOT recommended.** See Accuracy Evolution above: CoT eliminates Unsure but creates high-confidence misclassifications. The LLM builds a coherent wrong narrative rather than admitting uncertainty. Current best config: multimodal 1-shot, non-CoT.

The prompt structure for `--cot --n-shot 2`:
```
[System] Physical Rules + CoT Reasoning Protocol (Step 1-4 template)
[User]
  ## Few-Shot Examples
  ### Example: TDE (ZTF_xxx)
  [analysis.md]
  ### Example: SN (ZTF_yyy)
  [analysis.md]

  ---

  ## Reasoning Task
  Reason through each physical discriminator step by step,
  then output the classification as JSON.

  ## Target: Classify This Source
  [target analysis.md]
```

CoT reasoning text is extracted from the raw response (text before the JSON) and saved as `cot_reasoning` in results JSON. `n_shot=0` goes to zero-shot without error (the `if not few_shot` guard was changed to `if n_shot > 0 and not few_shot`).

### Multimodal with Cutout Images

The multimodal pipeline supports multiple images per source: `lightcurve.png` (always first) and `cutout.png` (SDSS sky survey finder chart, downloaded from radec.txt coordinates). Images are loaded by explicit filename, checked in order, silently skipped if missing.

**Cutout pipeline (Jul 2026):**

```bash
# 1. Extract coordinates from ZTFSNIa JSON / WYB TXT / TNS CSV
python extract_radec.py          # → sources/{id}/radec.txt

# 2. Download SDSS cutout for all sources with valid coordinates
python cutout.py --all --survey SDSS --size 300

# 3. Enable in config (already done)
# config.py: MULTIMODAL_IMAGES = ["lightcurve.png", "cutout.png"]
```

**Coordinate sources (extract_radec.py):** See `references/coordinate-extraction.md` for detailed source-type-to-data-source mapping and TNS CSV parsing quirks.

| Source type | Data source | Path |
|-------------|------------|------|
| ZTF SN (ZTF19aa*) | ZTF alert JSON | `data/ZTFSNIa/{name}.json` |
| wmx_ TDE with IAU name | WYB forced photometry | `data/WYB_ZTF-basecorr/{IAU_name}.txt` |
| wmx_ TDE (recent) | TNS search CSV | `data/TNS_TDE_search_results.csv` |
| WFST J-coordinate | CSV filename parse | `data/*.csv` |

**Coverage (Jul 2026):** 165/302 sources have coordinates, 118 have SDSS cutouts (47 outside SDSS footprint, 137 missing coordinates). SDSS only covers Dec ≳ -10°; PanSTARRS/Legacy are blocked or unreachable from USTC network.

### Conditional System Prompt (cutout-aware)

`_make_system_prompt(cot=False, has_cutout=False)` dynamically includes/excludes the Host Galaxy section based on whether `cutout.png` exists for the target source. When `has_cutout=True`, §5 (Host Galaxy Context) and Decision Logic step 5 are added; when `has_cutout=False`, the prompt has no mention of cutout images — preventing model confusion when no image is provided.

`build_prompt()` checks `has_image(target_id, "cutout.png")` and passes the flag through. The CoT reasoning steps auto-renumber (Step 4 = Host when cutout present, Step 4 = Synthesis when absent).

**Visual design** (plot.py):
| Band | Color | Marker | Hex |
|:----:|:-----:|:------:|:---:|
| g | green | ● circle | `#2ecc71` |
| r | red | ■ square | `#e74c3c` |
| u | blue | ▲ triangle | `#3498db` |

Includes error bars, peak annotation with band label, and header with N pts + span. Output: 1480×729, 150 dpi, ~50 KB.
ls data/TS/Flux/TDE/*.npy | wc -l       # raw files
python promt.py --stats                   # registered entries
```

**Real example (Jul 2026):** Only 3 of 198 wmx_ TDE sources were in index.json. The user had been running eval on 55 TDE sources when 195 more were available — inflating class imbalance (55:48 → 250:50 after ingestion).

**Fix workflow:**
```bash
# 1. Ingest all missing sources
python promt.py --batch data/TS/Flux/TDE/ --label TDE

# 2. Verify counts match
python promt.py --stats

# 3. Generate plots for multimodal mode
python plot.py --all
```

**Resulting class imbalance:** After ingestion, TDE count can grow dramatically (~250 vs ~50 SN). This skews eval metrics — high TDE recall may reflect data bias rather than model quality. Consider stratified sampling for eval.

### Other Pitfalls

1. **Synth mock sources**: `synth_flux_*` are synthetic TDE, should be excluded from few-shot pool. `index.json` already has them removed.
2. **Raw data too large**: Section 3 (raw data) is auto-stripped for API calls (saves ~70% tokens). The API proxy drops connections for >20K char payloads.
3. **max_tokens**: Set to 3000 (raised from 2000 after truncated JSON responses). Classification JSON with reasoning indicators needs ~1000-1500 tokens; 2000 was insufficient for complex cases.
4. **USTC proxy returns content in reasoning_content**: Both `qwen3.6-reasoner` and `deepseek-v4-pro` responses through the USTC proxy use `reasoning_content` field with `content=None`. `call_api()` handles this: `text = msg.content or getattr(msg, "reasoning_content", "") or ""`.
5. **read_md() section renumbering**: After removing old §3 (Predictive Features), `read_md()` splits on `"## Section 3:"` and keeps the part after it starting at `"## Section 4:"` (now the new §3 — Raw Light Curve). The function handles both old 5-section and new 4-section structures.
6. **Slow API**: ~25-60s per call. Plan for ~1 min per source classification.
7. **Few-shot pool must exclude test source**: `sample_few_shot()` accepts `exclude` set. `eval.py` automatically excludes the entire test set from few-shot sampling.
8. **API key management**: Never commit `.env`. Use `.env.example` as template.
9. **Large sources are NOT a problem**: Sources with 500+ data points work fine because raw data is auto-stripped for API calls.
10. **Multimodal must include system prompt**: The original build_prompt() dropped the system prompt in multimodal mode. Fixed: multimodal now includes system message. Without physics rules, score dropped from 0.85 to 0.65 in testing (WFST_J101658).
11. **Correct rise physics**: TDE rise is concave (fast start, decelerating) — fallback accretion. SN rise is convex (slow start, accelerating in optical) — shock cooling. TDE rise 10-60 days; SN rise <10 days. The old Timescale section had these backwards (SN at 30-200 days).
12. **Anti-TDE ordering check**: Added to §0 of system prompt (2026-07-26). Light curves that DECLINE first then RISE later are inconsistent with TDE physics (TDE = single burst, must rise→peak→decline). Model should classify these as SN or Unsure.
13. **Multimodal images use explicit naming, not directory scanning**: `config.MULTIMODAL_IMAGES` defines which image files to include. Files not in the list are ignored. This prevents accidental inclusion of unrelated PNGs and keeps behavior predictable.
14. **TNS CSV has header-column mismatch**: `TNS_TDE_search_results.csv` has an unnamed extra column between `class` and `ra` (possibly sub-class or reps-count). `csv.DictReader` will misalign all columns after it. Parse by column index: col5=ra, col6=decl. The IAU name is `col1 + col2` (e.g., "TDE" + "2022czy" → "TDE2022czy").
15. **Multimodal without cutout → conditional prompt**: When `cutout.png` is absent, system prompt must NOT mention cutouts or host galaxies. Use `_make_system_prompt(has_cutout=False)` to auto-strip the Host Galaxy section. Otherwise the model looks for an image that isn't there and performance degrades (40% Unsure rate observed).
16. **Section 3 (Predictive Features) removed**: Its auto-generated hints were physically wrong (timescale mismatches) and conflicted with the system prompt. The LLM now relies entirely on physics rules in the system prompt, not on pre-computed hints.

## Results JSON Schema

```json
{
  "source_id": "...",
  "classification": {"label": "TDE", "confidence": "high", "score": 0.85},
  "reasoning": {
    "summary": "one-sentence verdict",
    "indicators": [
      {"name": "Color evolution", "value": "red-to-blue", "weight": 0.4, "direction": "TDE"}
    ]
  },
  "quality": {"flags": [...]},
  "few_shot": [{"id": "...", "label": "TDE"}, ...],
  "cot": false,
  "cot_reasoning": "",
  "tokens": {"prompt": 5000, "completion": 800, "total": 5800}
}
```

> When `--cot` is enabled, `cot_reasoning` contains the LLM's Step 1-4 reasoning text (everything before the JSON in the raw response). The `_extract_reasoning()` helper in `classify.py` strips the JSON to extract this.

## Data Flow for New Sources

1. User provides npy/csv → `promt.py` generates `sources/{id}/analysis.md` and updates `index.json`
2. `extract_radec.py` extracts coordinates from ZTFSNIa/WYB/TNS → `sources/{id}/radec.txt`
3. `plot.py` generates `sources/{id}/lightcurve.png`; `cutout.py --all` downloads SDSS cutouts → `sources/{id}/cutout.png`
4. `classify.py` samples few-shot from labeled sources, builds prompt (with conditional host galaxy section if cutout exists), calls API
5. Result saved to `results/{id}.json` with full reasoning chain

## Verification Checklist

- [ ] `.env` has valid API key
- [ ] `index.json` has labeled TDE and SN sources (at least 5 each)
- [ ] `sources/{id}/lightcurve.png` exists for multimodal targets
- [ ] `sources/{id}/radec.txt` exists and not "unknown" (run `extract_radec.py`)
- [ ] `sources/{id}/cutout.png` exists for SDSS footprint sources (run `cutout.py --all`)
- [ ] `classify.py` runs in foreground, not background
- [ ] Test with `python classify.py WFST_J101658` before batch runs
- [ ] Check results JSON has valid `classification.label` (not "?" or error)