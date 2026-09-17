# ZTF_prompt

An LLM-based astronomical light-curve classification tool. It converts `npy`/`csv` light-curve data into a structured Markdown analysis report, then calls a large model via few-shot prompting to classify the source.

---

## Directory Structure

```
ZTF_prompt/
├── .env                 ← API key configuration
├── config.py            ← global configuration
├── promt.py             ← data → MD analysis report
├── plot.py              ← data → light-curve PNG (for multimodal classification)
├── classify.py          ← MD → LLM → classification result
├── eval.py              ← evaluate accuracy (actively calls the API)
├── summary.py           ← aggregate existing results + produce plots (zero API cost)
├── run.sh               ← one-shot full workflow script
├── diag.py              ← network diagnostic tool
├── MAD.py               ← median absolute deviation (MAD) feature computation
├── enrich.py            ← enrich with host-galaxy Gaia/WISE info
├── enrich_analysis.py   ← enrich analysis.md (color ranges/AGN/baseline)
├── cutout.py            ← SDSS/DSS host-galaxy cutout download
├── download_cutouts.py  ← batch cutout download
├── extract_radec.py     ← extract ra/dec from metadata
├── eval_cutout.py       ← cutout quality assessment
├── promt_old.py         ← legacy version of promt.py (reference only, do not use)
│
├── prompts/             ← external System Prompt files
│   ├── system_v1.txt        ← Prompt v1
│   ├── system_v2.txt        ← Prompt v2
│   ├── system_v3.txt        ← Prompt v3 (current default)
│   └── cot.txt              ← Chain-of-Thought additional instructions
├── templates/           ← Few-shot exemplar configuration files
│   ├── fewshot.json          ← default exemplars (TDE+SN each ×1)
│   ├── fewshot_text.json     ← text mode 3-shot (TDE+SN each ×3)
│   └── fewshot_boundary.json ← boundary samples (conflicting signals, anchors decision boundary)
│
├── sources/             ← generated files (one subdirectory per source)
│   └── {id}/
│       ├── analysis.md      ← structured analysis report
│       ├── lightcurve.png   ← light-curve plot (u=blue, g=green, r=red)
│       └── cutout.png       ← SDSS host-galaxy cutout (optional)
├── index.json           ← label index of all sources
├── results/             ← classification result JSON
├── results_enriched/    ← enriched result JSON
├── data/                ← downloaded raw data (e.g. AGN flux)
├── WFSTtest/            ← WFST test data
└── summary/             ← summary JSON + confusion-matrix/distribution plot PNG
```

---

## Step 1: Configuration

Edit the `.env` file and fill in the API key:

```
LLM_API_KEY=***  LLM_MODEL=deepseek-v4-pro
```

- `LLM_API_KEY`: API key (required)
- `LLM_MODEL`: model name (default `deepseek-v4-pro`)

---

## Step 2: Generate labeled data (few-shot pool)

Classification needs sources with known labels as examples. First generate MD files for TDE and SN:

```bash
# TDE (real sources, no synth mocks)
python promt.py --batch /home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/TDE/ --label TDE

# SN
python promt.py --batch /home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/SN/ --label SN
```

> Note: `--batch` processes all `.npy` and `.csv` files in the directory. To process just a few, use single-file mode.

---

## Step 3: Generate light-curve plots (multimodal mode)

```bash
# Single source
python plot.py /path/to/source_flux.npy --source-id WFST_J101658

# Batch
python plot.py --batch /home/.../Flux/TDE/ --max 50

# Generate for all sources in index.json
python plot.py --all
```

Output is written to `sources/{id}/lightcurve.png`: u band in blue, g band in green, r band in red, with error bars and peak annotation.

> Multimodal classification needs the PNG file. `classify.py --mode multimodal` reads it automatically, and degrades to text mode if the PNG is missing.

---

## Few-Shot Exemplar Management

The few-shot examples used during classification can be precisely controlled via `templates/fewshot*.json`, replacing the default random sampling.

### Three presets

| File | Use | TDE examples | SN examples |
|------|-----|-------------|-------------|
| `fewshot.json` | **default** multimodal (1-shot) | `wmx_TDE_2024lhc`<br>Δ=-8.7, 432pts, strong TDE | `ZTF19aaapnxn`<br>Δ=+6.2, 444pts, textbook SN |
| `fewshot_text.json` | text mode (3-shot) | 3: lhc / pvu / uvz | 3: aaapnxn / aagrdcs / aajxwnz |
| `fewshot_boundary.json` | boundary samples | `wmx_TDE_2022arb`<br>Δ=**0.0 Flat** (no color signal) | `ZTF19aagmsrr`<br>Δ=**-176 Red→Blue** (color looks TDE) |

### Principle

- **Textbook exemplars** (default/text): full signal coverage, teaches the model "what it should look like".
- **Boundary exemplars**: conflicting signals — the 2022arb TDE has no color evolution, while the aagmsrr SN has extreme Red→Blue. This forces the model to do **signal-weight reasoning** rather than simple pattern matching, anchoring the decision boundary.

### Loading logic

```
--exemplar-set boundary  →  templates/fewshot_boundary.json
--exemplar-set text      →  templates/fewshot_text.json
(no flag)                →  templates/fewshot.json
(file missing/empty)     →  fallback to random sampling (legacy behavior)
```

### Customization

```bash
# Edit the exemplar list (add/remove/change IDs; list length = n_shot)
vim templates/fewshot.json

# Create a new set
cp templates/fewshot.json templates/fewshot_my_custom.json
python classify.py WFST_J101658 --exemplar-set my_custom

# Fall back to random sampling
mv templates/fewshot.json templates/fewshot.json.bak
```

---

## Step 4: Classification

```bash
# Classify a single source (default: 3-shot, no CoT, text mode)
python classify.py WFST_J101658

# Multimodal mode (needs lightcurve.png)
python classify.py WFST_J101658 --mode multimodal --model qwen3.6-chat

# Classify all unknown sources
python classify.py --all-unlabeled

# Force re-classification (overwrite existing result)
python classify.py WFST_J101658 --force

# Adjust few-shot count
python classify.py WFST_J101658 --n-shot 2          # 2-shot
python classify.py WFST_J101658 --n-shot 0           # zero-shot

# Enable Chain-of-Thought (step-by-step reasoning)
python classify.py WFST_J101658 --cot

# CoT + few-shot combination
python classify.py WFST_J101658 --cot --n-shot 2

# Switch model
python classify.py WFST_J101658 --model qwen3.6-reasoner

# Switch exemplar set
python classify.py WFST_J101658 --exemplar-set boundary    # boundary samples
python classify.py WFST_J101658 --exemplar-set text        # text 3-shot
```

---

## Step 5: View results

```bash
# View full result (classification + confidence + each decision indicator)
python classify.py --results WFST_J101658

# Read the JSON directly
cat results/WFST_J101658.json
```

Result JSON structure:

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

> In CoT mode, the `cot_reasoning` field stores the LLM's step-by-step reasoning text (Steps 1-4).

---

## Evaluating accuracy

```bash
# Default: 30 test samples per class, 3-shot
python eval.py

# Custom parameters
python eval.py --test-size 10 --n-shot 2 --classes TDE,SN

# Four Few-Shot × CoT combinations
python eval.py --n-shot 0                   --test-size 10 --classes TDE,SN   # physics rules only
python eval.py --n-shot 2                   --test-size 10 --classes TDE,SN   # physics + exemplars
python eval.py --n-shot 0  --cot            --test-size 10 --classes TDE,SN   # physics + CoT
python eval.py --n-shot 2  --cot            --test-size 10 --classes TDE,SN   # physics + exemplars + CoT

# Detailed output (shows per-source predictions)
python eval.py --verbose
```

Output includes: confusion matrix, per-class Precision/Recall/F1, misclassified cases, and low-confidence cases.

---

## Result summary

`summary.py` reads `results/*.json` directly and does **not** call the API — zero token cost.

```bash
# Full summary (known accuracy + unknown distribution)
python summary.py

# Unknown sources only
python summary.py --unknown-only

# Known sources accuracy only
python summary.py --known-only

# List every entry in detail
python summary.py --verbose

# Low-confidence results only
python summary.py --min-conf low
```

Output has three parts:
- **Overview**: total count, class distribution, confidence distribution
- **Known Sources**: confusion matrix, Precision/Recall/F1, Unsure rate, misclassified cases
- **Unknown Sources**: TDE/SN/Unsure distribution (with bar chart), confidence stratification

Add `--plot` to automatically generate two plots:
- `{name}.png` — Known-source confusion-matrix heatmap (blue scale, academic white background)
- `{name}_unknown.png` — Unknown classification distribution bar chart (stratified by confidence)

```bash
python summary.py --plot                # produce plots
python summary.py --exemplar-set boundary --plot  # boundary-set plots
```

---

## One-shot full workflow

```bash
# Default: multimodal, default exemplar set
bash run.sh

# Switch exemplar set
bash run.sh --set boundary
bash run.sh --set text --mode text

# Only summarize existing results (no classification)
bash run.sh --summary-only

# Skip plotting
bash run.sh --skip-plot
```

Equivalent to running manually:
1. `python plot.py --all`
2. `python classify.py --all-unlabeled --mode multimodal --model qwen3.6-chat`
3. `python summary.py --plot`

---

## Managing labels

```bash
# View statistics
python promt.py --stats

# List all sources of a class
python promt.py --list TDE

# Change a label
python promt.py --relabel WFST_J101658 TDE
```

---

## Complete workflow example

```bash
# 1. Configure
vim .env    # fill in API key

# 2. Generate few-shot pool (only needs to be done once)
python promt.py --batch .../Flux/TDE/ --label TDE
python promt.py --batch .../Flux/SN/  --label SN

# 3. Generate new data
python promt.py data/my_new_source.csv --label unknown

# 4. Classify
python classify.py --all-unlabeled

# 5. View
python classify.py --results my_new_source
```

---

## MD analysis report structure

Each source's analysis report has several sections:

| Section | Content |
|---------|---------|
| §1 Source Metadata | Basic info (point count, bands, peak, etc.) |
| §2 Derived Features | Computed features (morphology, color evolution, per-phase statistics, data quality) |
| §3 Raw Light Curve | Full raw data table |
| §4 Classification Protocol | Classification instructions for the LLM |

> Note: When calling the API, §3 (the raw data table) is dropped by default to save tokens. The System Prompt uses **color evolution** and **rise morphology** (concave = TDE, convex = SN) as the primary discriminators; rise duration and total span are secondary references.
> In Prompt v3, a new §0.5 Source Morphology / Stellar-Contamination Gate runs before §1: it uses the cutout to detect foreground/field stars and excludes them (classified as `Others` with `"star_like": true` in `reasoning`).

---

## Output files

| File | Content |
|------|---------|
| `sources/{id}/analysis.md` | Full analysis report |
| `sources/{id}/lightcurve.png` | Light-curve plot (multimodal use) |
| `results/{id}.json` | Classification result (confidence + reasoning chain) |
| `eval_report.json` | Evaluation report (generated after running `eval.py`) |
| `index.json` | Label and metadata index of all sources |
| `templates/fewshot*.json` | Few-shot exemplar configuration files |
| `summary/*.json` | Summary data (Overview + Known + Unknown) |
| `summary/*.png` | Confusion-matrix heatmap + unknown-source distribution plot |
| `run.sh` | One-shot full workflow script |

---

## Notes

1. **API calls are slow**: the USTC proxy takes roughly 25-60 s per call, so classifying one source takes about 1 minute.
2. **Do not run in background**: `classify.py` must run in the foreground (background-process SSL connections have issues).
3. **Mock sources are excluded from few-shot**: `synth_flux_*` are synthetic data and have been removed from `index.json`.
4. **Multimodal mode**: needs `sources/{id}/lightcurve.png` (generated with `plot.py`). The USTC proxy's deepseek model does not support vision, so specify `--model qwen3.6-chat`.
5. **System Prompt physical discriminators**: ① color evolution (g−r) → ② rise morphology (concave/convex) → ③ decline shape → ④ data quality. TDE has concave rise (fallback-driven), SN has convex rise (shock cooling).
6. **Plot colors**: u=blue ▲, g=green ●, r=red ■.
