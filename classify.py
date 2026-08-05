#!/usr/bin/env python3
"""
classify.py - Classify light curves using LLM via API

Usage:
  python classify.py WFST_J101658
  python classify.py --all-unlabeled
  python classify.py WFST_J101658 --n-shot 3 --mode text
"""

import sys, os, json, random, argparse, base64, re
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI

import config


def _get_client():
    """Create a fresh API client (avoids SSL session caching issues)."""
    return OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY, timeout=300)


# ═══════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════

def read_md(source_id, include_raw=False):
    """Read analysis.md. If include_raw=False, strip Section 4 (raw data) to save tokens."""
    path = config.SOURCES_DIR / source_id / "analysis.md"
    if not path.exists():
        return f"[analysis.md not found for {source_id}]"
    content = path.read_text()
    if not include_raw:
        # Keep Sections 1-2 + 4 (drop raw data Section 3)
        parts = content.split("## Section 3:")
        if len(parts) >= 2:
            # Remove Section 4, keep Section 5
            after_s4 = parts[1]
            sec5_start = after_s4.find("## Section 4:")
            if sec5_start >= 0:
                content = parts[0] + after_s4[sec5_start:]
            else:
                content = parts[0]
    return content


def read_image_b64(source_id, filename="lightcurve.png"):
    path = config.SOURCES_DIR / source_id / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def has_image(source_id, filename="lightcurve.png"):
    return (config.SOURCES_DIR / source_id / filename).exists()



def _peak_flux(source_id):
    """Extract peak flux (uJy) from analysis.md. Returns None if not found."""
    md_path = config.SOURCES_DIR / source_id / "analysis.md"
    if not md_path.exists():
        return None
    text = md_path.read_text()
    m = re.search(r"Peak flux \| ([\d.]+) uJy", text)
    return float(m.group(1)) if m else None

def _load_fewshot_json(exemplar_set=None):
    """Load curated few-shot exemplars from templates/fewshot{_set}.json.
    Returns dict {class_name: [source_id, ...]} or None if file not found.
    """
    if exemplar_set:
        path = config.TEMPLATES_DIR / f"fewshot_{exemplar_set}.json"
    else:
        path = config.TEMPLATES_DIR / "fewshot.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def sample_few_shot(idx, n_per_class=None, exclude=None, exemplar_set=None):
    """Sample few-shot examples from labeled sources, excluding certain IDs.

    Priority:
      1. templates/fewshot.json (or fewshot_{exemplar_set}.json) — curated, reproducible
      2. Fall back to random sampling from index.json
    """
    if exclude is None:
        exclude = set()

    # Try curated fewshot.json first
    fs = _load_fewshot_json(exemplar_set)
    if fs:
        selected = []
        for cls_name in config.CLASSES:
            for sid in fs.get(cls_name, []):
                if sid in exclude:
                    continue
                if sid not in idx:
                    print(f"  [warn] fewshot exemplar '{sid}' not in index.json, skipping")
                    continue
                selected.append((sid, cls_name))
        if selected:
            print(f"  [fewshot] using {len(selected)} curated exemplars from "
                  f"fewshot{'_' + exemplar_set if exemplar_set else ''}.json")
            return selected
        print(f"  [warn] fewshot{'_' + exemplar_set if exemplar_set else ''}.json "
              f"exists but no valid exemplars, falling back to random")

    # Fallback: random sampling
    if n_per_class is None:
        n_per_class = config.N_SHOT_TEXT
    selected = []
    for cls_name in config.CLASSES:
        pool = [sid for sid, info in idx.items()
                if info["label"] == cls_name and sid not in exclude]
        if not pool:
            continue
        n = min(n_per_class, len(pool))
        chosen = random.sample(pool, n)
        for sid in sorted(chosen):
            selected.append((sid, cls_name))
    return selected


def _make_system_prompt(cot=False, has_cutout=False):
    """Build system prompt dynamically from config.CLASSES.
    
    Args:
        cot: If True, include Chain-of-Thought reasoning instructions.
        has_cutout: If True, include host galaxy cutout guidance.
    """
    classes_str = "|".join(config.CLASSES)
    class_list = ", ".join(config.CLASSES)
    # For indicator direction: only real physical classes (not Unsure)
    real_classes = [c for c in config.CLASSES if c != "Unsure"]
    real_str = "|".join(real_classes)

    base = (
        f"You are an astronomical transient classifier specializing in TDE/SN light curves. "
        f"Classify each source as one of: {class_list}.\n"
        "\n"
        "Class definitions:\n"
        "- TDE: tidal disruption event — single burst, rise→peak→power-law decline.\n"
        "- SN: supernova — variable rise shapes, plateau or slow decline, Ni-56 tail.\n"
        "- AGN: active galactic nucleus — can mimic TDE light curve shape "
        "(rise→peak→decline), but distinguished by (a) strong IR excess from "
        "the dust torus (WISE W1−W2 typically > 0.8), AND/OR (b) significant "
        "pre-transient variability (f_pre_var > 0.15) indicating ongoing "
        "activity before the 'flare'. Always nuclear (gaia_sep ≈ 0). "
        "May show repeated flares or longer timescales than typical TDE.\n"
        "- Others: does NOT match TDE, SN, or AGN physical characteristics — e.g. CVs, "
        "variable stars, artifacts, or any transient with non-stellar "
        "light curve morphology. Use this when the source is clearly inconsistent "
        "with TDE, SN, and AGN physics. "
        "DISTINCTION from AGN: AGN typically has nuclear location (gaia_sep ≈ 0), "
        "may have WISE W1−W2 ≥ 0.8, and may show f_pre_var ≥ 0.15. "
        "Others should be used when the source FAILS these AGN checks OR has "
        "a light curve shape fundamentally non-AGN (strictly periodic, "
        "eclipsing, single spike with zero pre/post activity). "
        "When in doubt between AGN and Others, prefer AGN if W1−W2 ≥ 0.8 "
        "AND f_pre_var ≥ 0.15 are BOTH present.\n"
        "- Unsure: genuinely ambiguous between TDE/SN/AGN/Others.\n\n"
        "## Physical Discriminators (check in priority order)\n\n"
        "### 0. Light Curve Completeness — check FIRST\n"
        "Before applying any discriminator, assess what phases are covered:\n"
        "- CRITICAL ORDERING CHECK: A TDE MUST rise (brighten) BEFORE it declines.\n"
        "  If the light curve DECLINES first and then RISES later, this is NOT\n"
        "  a TDE — the temporal ordering is anti-TDE. Classify as **Others**.\n"
        "  Check Section 2.1 (Flux by Phase) or the light curve plot: if early-time\n"
        "  flux is HIGHER than mid/late-time flux (overall down→up shape), it is\n"
        "  inconsistent with TDE physics (TDE = single burst, rise→peak→decline).\n\n"
        "- Full coverage (rise + peak + decline): use ALL indicators below.\n"
        "- Peak + decline only (rise missing or too short, < 5 data points in rise):\n"
        "  SKIP §2 (Rise Morphology). Rely on §1 (Color Evolution in decline)\n"
        "  + §3 (Decline Shape) + §4 (Total Span). Use score ~0.1-0.2 lower than full-coverage cases to reflect reduced diagnostic power.\n"
        "- Decline only (no peak, no rise): few diagnostics available.\n"
        "  Rely primarily on color in the decline phase. Confidence should be\n"
        "  LOW — classify as Unsure unless signals are decisive.\n\n"
        "### 1. Color Evolution — FOCUS on DECLINE phase\n"
                "- analysis.md sections 2.2/2.3 provide THREE color pairs: **g−r**, **u−g**, **u−r** (mag).\n"
                "- Each has a mag column (PRIMARY) and uJy column (fallback). Use mag whenever available.\n"
                "- TDE color ranges (mag) — CHECK THIS FIRST:\n"
                "    g−r ∈ (−0.6, +0.1)     →  TDE color zone\n"
                "    u−g ∈ (−0.5, +0.4)     →  TDE color zone  (use when g−r unavailable)\n"
                "    u−r ∈ (−0.9, +0.2)     →  TDE color zone  (use when g−r unavailable)\n"
                "- RANGE RULE (primary): if the mag values stay WITHIN the TDE zone throughout\n"
                "  the light curve, this is a STRONG TDE indicator REGARDLESS of evolution\n"
                "  direction. Example: g−r=−0.44→−0.37 (still inside −0.6 to +0.1) = TDE-like.\n"
                "- EVOLUTION DIRECTION (secondary, only when delta > 0.15 mag):\n"
                "    Red→Blue (delta negative > 0.15): TDE indicator\n"
                "    Blue→Red (delta positive > 0.15) AND leaves TDE zone: SN indicator\n"
                "- Small delta (< 0.15 mag) WITHIN TDE zone: still TDE. Do NOT over-interpret.\n"
                "- Color outside TDE zone throughout: favors SN or Others.\n"
                "- Priority: g−r > u−g ≈ u−r. If g−r exists, use it first.\n"
                "- Rise-phase color is often noisy. Downgrade weight to ≤ 0.15 if only rise-phase\n"
                "  color is available.\n"
                "- UNCERTAINTY DISCOUNT: each color entry has an 'err' column (mag).\n"
                "  If the median |err| across all phases > 0.15 mag → color measurement\n"
                "  is UNRELIABLE → downgrade optical color weight by 0.10.\n"
                "  Large errors mean the true color could be outside the TDE zone even\n"
                "  if the central value is inside — be more conservative.\n"
                "- Caveat: some SN subtypes (IIb, IIn, SLSN-I) can show TDE-like color.\n\n"
        "### 2. Rise Morphology — SHAPE matters more than duration\n"
        "\n"
        "Concave rise (decelerating, d²F/dt² < 0):\n"
        "  → STRONG TDE indicator\n"
        "  Physical: fallback accretion — most bound debris returns first,\n"
        "  giving rapid initial flux increase that decelerates toward peak.\n"
        "  Look for: steepest slope in early rise, flattening near peak.\n"
        "  Hint: in per-phase Phase column (Section 2.3),\n"
        "  negative early → approaching zero = concave (TDE-like).\n"
        "\n"
        "Convex rise (accelerating, d²F/dt² > 0):\n"
        "  → favors SN\n"
        "  Physical: shock cooling — post-breakout envelope is UV-hot,\n"
        "  optical flux is initially dim and accelerates as temperature drops\n"
        "  into the optical band.\n"
        "  Look for: slope increasing through the rise, steepest just before peak.\n"
        "\n"
        "Rise duration as secondary check:\n"
        "  < 10 days  → favors SN (shock-cooling timescale)\n"
        "  10–60 days → typical TDE (fallback timescale)\n"
        "  > 60 days  → check shape: concave still possible TDE; convex = unusual\n"
        "\n"
        "Total span:\n"
        "  < 200 days  → favors TDE (t^(-5/3) decay ~months)\n"
        "  > 1000 days → favors SN (Ni decay tail can last years)\n"
        "\n"
        "### 3. Decline Shape\n"
        "- Steep power-law decline (>1 uJy/d sustained): favors TDE\n"
        "- Plateau or very slow decline (<0.1 uJy/d): favors SN\n\n"
        "### 4. Data Quality\n"
        "- Total points < 10: sparse data — signals inherently weaker. Use low score (0.3-0.5) rather than defaulting to Unsure\n"
        "- Total points 10–20: moderate data quality — score typically 0.5-0.7\n"
        "- Rise phase < 5 points: rise shape is UNRESOLVED — skip §2, do not guess\n"
        "- Sparse sampling (mean cadence > 5 days): color evolution may be\n"
        "  undersampled, downgrade weight of color indicator by 0.1\n"
        "- Single-band only: no color information, be cautious\n"
        "- Isolated outlier points far from the main trend: IGNORE them. Extreme\n"
        "  outliers (>3σ from local trend, or single points jumping far above/below\n"
        "  the bulk) are likely artifacts (bad subtraction, cosmic rays, etc.).\n"
        "  Do NOT let a single outlier drive rise/decline shape judgments — assess\n"
        "  the overall trend, not individual extreme points.\n"
        "- If quality is poor and signals are ambiguous, reflect this in the score (typically 0.3-0.5)\n\n" +
        (
        "### 5. Host Galaxy Context (from cutout image)\n"
        "The SDSS cutout shows the host galaxy. Use as a SECONDARY check:\n"
        "- Red, smooth, elliptical host → favors TDE (TDEs prefer quiescent galaxies)\n"
        "- Blue, spiral, or irregular host → weakly favors SN (star-forming hosts)\n"
        "- Source centered on galaxy nucleus → favors TDE (nuclear events)\n"
        "- Source clearly in disk/arm → favors SN (traces star formation)\n"
        "- No obvious host (faint/blended): this indicator is UNINFORMATIVE, skip it\n"
        "IMPORTANT: host morphology is a WEAK prior (weight ≤ 0.2). NEVER override\n"
        "strong color/rise signals based on host alone.\n\n"
        if has_cutout else ""
        ) +
        "### 6. Gaia Separation (host proximity)\n"
        "- Check Section 1 for 'Gaia Sep' field. If absent, skip this indicator.\n"
        "- gaia_sep < 2 arcsec → source at/near galaxy nucleus → nuclear event\n"
        "  → favors TDE or AGN over SN (TDE and AGN are both nuclear).\n"
        "  Use WISE W1−W2 (below) to break the TDE/AGN degeneracy.\n"
        "- gaia_sep ≥ 2 arcsec → source in galaxy disk → favors SN.\n"
        "- This is a WEAK prior (weight ≤ 0.15). Do NOT override strong light curve\n"
        "  signals based on gaia_sep alone.\n"
        "- Note: some TDE candidates have been found off-nucleus (recoiling SMBH,\n"
        "  merger remnant). Treat large gaia_sep as a soft counter-indicator, not an\n"
        "  absolute disqualifier.\n\n"
        "### 7. WISE IR Color (dust/AGN discriminator) — KEY TDE/AGN DEGENERACY BREAKER\n"
        "- Check Section 1 for 'WISE W1/W2/W1-W2' field. If absent, skip this indicator.\n"
        "- W1−W2 < 0.5 → IR-bluish, low dust → STRONG TDE indicator (weight ≤ 0.25).\n"
        "  A TDE-like light curve + W1−W2 < 0.5 makes TDE classification robust.\n"
        "- 0.5 ≤ W1−W2 < 0.8 → ambiguous IR color → weak TDE indicator (weight ≤ 0.15).\n"
        "  Cannot exclude mild AGN activity; downgrade TDE confidence accordingly.\n"
        "- W1−W2 ≥ 0.8 → IR-red, dust torus → STRONG AGN indicator (weight ≤ 0.25).\n"
        "  CRITICAL: if light curve shape is TDE-like but W1−W2 ≥ 0.8, classify as\n"
        "  AGN, NOT TDE. The IR excess (hot dust torus) distinguishes AGN from the\n"
        "  relatively dust-free environment of a TDE.\n"
        "- WISE W1/W2 magnitudes as secondary check: very bright W1 (< 11 mag) may\n"
        "  indicate a luminous AGN host, but do NOT override the W1−W2 color.\n"
        "- Sum of color (§1) + WISE (§7) weights ≤ 0.5 combined.\n\n"
        "### 8. Pre-Transient Variability (f_pre_var) — AGN DISCRIMINATOR\n"
        "- Check analysis.md Section 2 for 'Pre-Transient RMS' or 'f_pre_var' field.\n"
        "  If absent, skip this indicator.\n"
        "- f_pre_var = intrinsic host variability amplitude / peak flux of the transient.\n"
        "  It measures whether the source was ALREADY varying before the 'flare'.\n"
        "- f_pre_var < 0.10: LOW pre-transient variability.\n"
        "  → Host was QUIET before the transient → favors TDE or SN (real outburst).\n"
        "  → Strong ANTI-AGN signal: a genuine TDE/SN on a quiet host.\n"
        "- 0.10 ≤ f_pre_var < 0.15: MODERATE. Some pre-flare activity.\n"
        "  → MILD AGN WARNING. Do not disqualify TDE, but note the ambiguity.\n"
        "- f_pre_var ≥ 0.15: HIGH pre-transient variability.\n"
        "  → Host was ACTIVE before the transient → AGN WARNING (weight ≤ 0.15).\n"
        "  The 'flare' may be AGN variability rather than a TDE.\n"
        "  If combined with W1−W2 < 0.5 → contradictory signals → downgrade to Unsure.\n"
        "  If combined with W1−W2 ≥ 0.8 → converging AGN evidence → classify AGN.\n"
        "- CRITICAL: f_pre_var should be used WITH WISE, not instead of it.\n"
        "  f_pre_var high + W1−W2 low → genuine tension (host active but IR-cold).\n"
        "  f_pre_var high + W1−W2 high → converging AGN evidence.\n"
        "- Weight ≤ 0.10 in isolation. Use as a cross-check, not a primary driver.\n\n"

                "## Decision Logic\n"
        "### Step 0: Shape Gate — rule out non-TDE shapes FIRST\n"
        "These are STRUCTURAL exclusion criteria. If ANY trigger, classify directly\n"
        "as the specified class — do NOT proceed to the Joint Matrix (Step 1).\n"
        "The shape gate overrides ALL color/WISE/variability signals.\n\n"
        "- PLATEAU (SN IIP type): the main decline phase ITSELF is flat/slow (>50 d)\n"
        "  with flux staying near peak level, then a sharp drop \u2192 **SN** (IIP-like).\n"
        "  DISTINGUISH from TDE late-time plateau: if the light curve first DECLINES\n"
        "  significantly (to < 50% of peak), THEN flattens \u2192 this is a TDE LATE\n"
        "  PLATEAU, NOT an SN IIP plateau. Do NOT apply this gate in that case.\n"
        "  The TDE late plateau is handled as a feature in the Joint Matrix (\u00a78).\n"
        "- DECLINE-THEN-RISE (anti-TDE): overall light curve goes DOWN first,\n"
        "  then UP later \u2192 **Others**. TDE physics requires rise\u2192peak\u2192decline;\n"
        "  the reverse (decline\u2192minimum\u2192rise) is structurally incompatible.\n"
        "  Check: early-time flux HIGHER than mid-time flux, late-time flux\n"
        "  also HIGHER than mid-time (V-shaped envelope, trough in middle third).\n"
        "- MULTI-PEAK: two or more distinct flux peaks (separated by > 20 d,\n"
        "  each with prominence > 15% of total range) \u2192 **Others**.\n"
        "  A single TDE cannot produce multiple distinct outbursts.\n"
        "  Do NOT route through WISE/AGN: multi-peak = Others regardless of IR.\n"
        "- SYMMETRIC: rise and decline have similar timescales (ratio 0.5\u20132.0)\n"
        "  \u2192 **Others** (TDE rise << decline is a defining characteristic).\n"
        "- LOW-AMPLITUDE: flux variation < 10% of median over full span\n"
        "  \u2192 **Others**. No clear transient event — likely variable star,\n"
        "  low-state AGN, or artifact. Check: std(flux)/median(flux) < 0.1.\n"
        "- CHAOTIC/OSCILLATORY: rapid up-down reversals (> 40% of consecutive\n"
        "  flux differences reverse direction), no coherent rise\u2192peak\u2192decline\n"
        "  envelope \u2192 **Others**. Likely CV, eclipsing binary, or artifact.\n"
        "- NO CLEAR STRUCTURE: wavy/irregular, no discernible rise\u2192peak\u2192decline\n"
        "  \u2192 classify as Others.\n"
        "If NONE of these exclusion criteria trigger, proceed to Step 1.\n"
        "If a gate routes to 'Others', perform a QUICK AGN CHECK:\n"
        "  If W1\u2212W2 \u2265 0.8 AND f_pre_var \u2265 0.15 \u2192 reclassify as **AGN** instead.\n"
        "  (The structural weirdness + strong IR + activity = likely AGN, not artifact).\n"
        "  Otherwise \u2192 keep as Others.\n\n"
        "### Step 1: Color \u00d7 WISE \u00d7 Shape \u00d7 f_pre_var Joint Matrix\n"
        "Apply rows TOP-TO-BOTTOM (first match wins).\n"
        "f_pre_var: LOW = < 0.10, HIGH = \u2265 0.15, N/A = unavailable.\n"
        "If f_pre_var is between 0.10\u20130.15 \u2192 treat as N/A (too ambiguous to use).\n\n"
        "| Optical Color | W1-W2    | Shape   | f_pre_var | \u2192 Result |\n"
        "|:---|:---|:---|:---|:---|\n"
        "| TDE zone | < 0.5   | concave | LOW or N/A | **TDE** (strong) |\n"
        "| TDE zone | < 0.5   | concave | HIGH       | **Unsure\u2192TDE** (AGN warning) |\n"
        "| TDE zone | < 0.5   | convex  | LOW or N/A | **Unsure\u2192TDE** |\n"
        "| TDE zone | < 0.5   | convex  | HIGH       | **Unsure** (color+TDE vs var+AGN) |\n"
        "| TDE zone | \u2265 0.8   | any     | any        | **AGN** (strong) |\n"
        "| TDE zone | [0.5,0.8) | concave | LOW     | **TDE** (weak, IR ambiguous) |\n"
        "| TDE zone | [0.5,0.8) | concave | HIGH    | **Unsure\u2192AGN** (IR+f_pre_var both warn) |\n"
        "| TDE zone | [0.5,0.8) | convex  | any     | **SN** (shape wins) |\n"
        "| TDE zone | N/A      | concave | LOW or N/A | **TDE** (moderate) |\n"
        "| TDE zone | N/A      | concave | HIGH       | **Unsure\u2192TDE** (AGN warning, no IR) |\n"
        "| TDE zone | N/A      | convex  | any        | **Unsure\u2192TDE** |\n"
        "| SN    | < 0.5      | any     | LOW        | **TDE** (weak, IR+quiet host override optical) |\n"
        "| SN    | < 0.5      | any     | HIGH or N/A| **Unsure** (IR-TDE vs opt-SN) |\n"
        "| SN    | \u2265 0.8      | any     | any        | **AGN** (weak) |\n"
        "| SN    | [0.5,0.8) or N/A | convex  | any   | **SN** (strong) |\n"
        "| SN    | [0.5,0.8) or N/A | concave | any   | **Unsure\u2192SN** |\n"
        "| None  | \u2265 0.8      | concave | any        | **AGN** |\n"
        "| None  | \u2265 0.8      | convex  | HIGH       | **AGN** (weak) |\n"
        "| None  | \u2265 0.8      | convex  | LOW or N/A | **Others** |\n"
        "| None  | < 0.5      | concave | LOW or N/A | **TDE** (weak) |\n"
        "| None  | < 0.5      | concave | HIGH       | **Unsure** (TDE shape but AGN-like var) |\n"
        "| None  | < 0.5      | convex  | any        | **SN** (weak) |\n"
        "| None  | [0.5,0.8) or N/A | concave | any   | **TDE?** (very weak) |\n"
        "| None  | [0.5,0.8) or N/A | convex  | any   | **SN?** (very weak) |\n"
        "| None  | [0.5,0.8) or N/A | unclear | any   | **Unsure** |\n"
        "| None  | N/A        | unclear | any        | **Unsure** |\n\n"
        "Definitions:\n"
        "- Optical Color (from Section 2.2):\n"
        "  TDE = g\u2212r \u2208 (\u22120.6, +0.1) OR u\u2212g \u2208 (\u22120.5, +0.4)\n"
        "    OR u\u2212r \u2208 (\u22120.9, +0.2). Values stay WITHIN zone throughout.\n"
        "  SN = consistently OUTSIDE TDE zone, OR Blue\u2192Red (\u0394>+0.15)\n"
        "    that LEAVES the TDE zone.\n"
        "  None = no band pair available, or color straddles TDE/SN boundary.\n"
        "- W1-W2 (from Section 1): <0.5 = TDE, [0.5,0.8) = ambiguous, \u22650.8 = AGN.\n"
        "  N/A = WISE data unavailable.\n"
        "- Shape: concave = TDE, convex = SN, unclear = cannot determine.\n"
        "- f_pre_var (from Section 2): LOW = < 0.10, HIGH = \u2265 0.15.\n"
        "  0.10\u20130.15 \u2192 treat as N/A (too ambiguous).\n"
        "  N/A = pre-transient data unavailable.\n"
        "- Top-to-bottom: first matching row applies.\n"
        "- Weight caps: optical color + WISE combined \u2264 0.5. Shape \u2264 0.5.\n"
        "  f_pre_var \u2264 0.10 (ancillary, not part of the 0.5 cap).\n"
        "  When color + shape + WISE + f_pre_var all agree \u2192 score \u2265 0.90.\n"
        "  When color vs shape conflict \u2192 score \u2264 0.55 (uncertain).\n\n"
        "### Step 2: Tiebreakers\n"
        "- Gaia sep (small \u2192 TDE/AGN; large \u2192 SN) \u2014 weight \u2264 0.15\n"
        "- Rise duration: < 10 d \u2192 SN, 10\u201360 d \u2192 TDE, > 60 d \u2192 check shape\n"
        "- Total span: < 200 d \u2192 TDE, 200\u20131000 d \u2192 ambiguous, > 1000 d \u2192 SN\n"
        "  (unless a clear late-time plateau is present \u2014 in that case, long span\n"
        "  is expected for TDEs and this tiebreaker is UNINFORMATIVE).\n"
        "- \u0394m_g (Magnitude Excess): transient peak minus ZTF reference magnitude\n"
        "  in g-band (from analysis.md Section 2.2).\n"
        "  \u0394m_g > 1.5: transient much brighter than host \u2192 real outburst\n"
        "    \u2192 favors TDE/SN over AGN flickering (weight \u2264 +0.10 toward TDE).\n"
        "  \u0394m_g < 0.5: transient barely brighter than host \u2192 AGN-like\n"
        "    \u2192 weak AGN indicator (weight \u2264 \u22120.10 toward TDE).\n"
        "  N/A: skip this tiebreaker.\n"
        "- f_plat_var (Plateau Variability): intrinsic variability during the\n"
        "  plateau phase, normalized by peak flux (from analysis.md Section 2.3).\n"
        "  f_plat_var < 0.05: plateau is STABLE \u2192 favors TDE\n"
        "    (TDE late-time emission should be smooth).\n"
        "  f_plat_var \u2265 0.10: plateau is ACTIVE \u2192 AGN WARNING\n"
        "    (ongoing stochastic variability, not TDE afterglow).\n"
        "  Weight \u2264 0.05. N/A if no plateau detected \u2192 skip.\n"
        "- When 2+ indicators conflict \u2192 classify as Unsure with preference.\n" +
        (
        "- Host galaxy (cutout): WEAK tiebreaker (weight \u2264 0.2).\n"
        "  Never override the color+shape matrix based on host alone.\n"
        if has_cutout else ""
        )
    )

    if cot:
            cot_section = (
                "\n## Reasoning Protocol (Chain-of-Thought)\n"
                "Before outputting the JSON, reason through each step in plain text.\n"
                "Label each step clearly with 'Step 0:', 'Step 1:', etc.\n\n"
                "Step 0 — Completeness: assess what phases are covered "
                "(full / peak+decline / decline only). "
                "If rise is missing, note that shape cannot be assessed.\n"
                "Step 1 — Shape Gate: check ALL exclusion criteria. "
                "PLATEAU (>50 d flat) \u2192 SN. "
                "DECLINE-THEN-RISE (down\u2192up, V-shape) \u2192 Others. "
                "MULTI-PEAK (\u22652 distinct peaks) \u2192 Others. "
                "SYMMETRIC (rise\u2248decline) \u2192 Others. "
                "LOW-AMPLITUDE (std/median < 0.1) \u2192 Others. "
                "CHAOTIC (>40% direction flips) \u2192 Others. "
                "NO STRUCTURE \u2192 Others. "
                "If any trigger: classify directly, set score \u2265 0.85, "
                "do NOT proceed to Steps 2\u20136.\n"
                "If gate routes to Others: check W1\u2212W2 \u2265 0.8 AND f_pre_var \u2265 0.15 "
                "\u2192 reclassify as AGN.\n"
                "Step 2 — Optical Color: extract g−r/u−g/u−r from Section 2.2. "
                "Are values in TDE zone (g−r ∈ −0.6 to +0.1) throughout? "
                "→ TDE optical color. Outside or Blue→Red leaving zone? → SN optical color. "
                "Check color err: if median |err| > 0.15 mag → note 'color uncertain'.\n"
                "Step 3 — WISE IR Color: check Section 1 for W1−W2. "
                "< 0.5 → TDE; [0.5,0.8) → ambiguous; ≥ 0.8 → AGN. "
                "If unavailable, note N/A.\n"
                "Step 3b — Pre-Transient Variability: check Section 2 for f_pre_var. "
                "< 0.10 → LOW (quiet host, anti-AGN); ≥ 0.15 → HIGH (AGN warning). "
                "If 0.10–0.15 or unavailable → skip.\n"
                "Step 4 — Joint Matrix: apply Color × WISE × Shape × f_pre_var matrix "
                "from Decision Logic Step 1. Output the matched classification.\n"
                "Step 5 — Tiebreakers: Gaia sep, rise duration, total span, "
                "\u0394m_g (magnitude excess > 1.5 \u2192 burst; < 0.5 \u2192 AGN-like), "
                "f_plat_var (plateau stability), "
                "host cutout (if available). Confirm or adjust classification. "
                "If matrix result conflicts with tiebreakers → Unsure.\n" +
                (
                "Step 6 — Host Galaxy: describe the host from cutout. "
                "Red/smooth/elliptical → TDE; blue/spiral → SN. "
                "No obvious host → skip.\n"
                if has_cutout else ""
                ) +
                "Step " + ("7" if has_cutout else "6") + " — Synthesis: state final classification "
                "from the matrix, adjusted by tiebreakers. "
                "If Step 1 Shape Gate triggered, explain why. "
                "Output JSON on next line.\n\n"
                "## Response Format\n"
            )
    else:
        cot_section = (
            "\n## Response Format\n"
            "Output ONLY a JSON object (no markdown, no thinking process):\n"
        )

    json_template = (
            '## Score Calibration\n'
            'Score represents classification certainty, NOT a probability:\n'
            '- 0.85–1.00: Decisive. Multiple strong indicators converge on same class.\n'
            '- 0.70–0.84: Strong. Primary indicators agree, minor counter-signals exist.\n'
            '- 0.55–0.69: Moderate. Signals favor this class but one meaningful counter-indicator.\n'
            '- 0.40–0.54: Weak/Mixed. Signals ambiguous or sparse; leaning this direction.\n'
            '- 0.25–0.39: Very weak. Only one indicator; essentially Unsure with a hint.\n'
            '- 0.00–0.24: Pure Unsure. No clear signal; quality too poor to judge.\n'
            'DO NOT default to 0.40 for Unsure — use the full range based on signal strength.\n\n'
            'UNCERTAINTY DISCOUNTS (apply BEFORE final score):\n'
            '- Color measurement unreliable (median |err| > 0.15 mag): \u22120.05\n'
            '- Rise unresolved (< 5 points): skip rise indicators entirely\n'
            '- Sparse data (< 10 total points): cap score at 0.55\n'
            '- f_pre_var HIGH + W1\u2212W2 LOW (contradiction): downgrade by \u22120.10\n\n'
            f'{{"classification":{{"label":"{classes_str}","confidence":"high|medium|low",'
            f'"score":0.0-1.0,"unsure_preference":"{real_str}|null"}},'
            f'"reasoning":{{"primary_signal":"...","indicators":[{{'
            f'"name":"...","value":"...","weight":0.0-1.0,"direction":"{real_str}"}}]}},'
            + (
            f'"host_galaxy":{{"morphology":"elliptical|spiral|irregular|unclear",'
            f'"color":"red|blue|unclear","position":"nucleus|disk|unclear",'
            f'"favors":"TDE|SN|inconclusive","weight":0.0-0.2}},'
            if has_cutout else
            f'"host_galaxy":{{"available":false,"note":"no cutout image"}},'
            ) +
            '"quality":{"flags":[]}}'
        )

    return base + cot_section + json_template


def build_prompt(target_id, few_shot, mode="text", cot=False):
    """Build the API prompt. Returns messages list.
    
    Args:
        target_id: source ID to classify.
        few_shot: list of (source_id, label) tuples (empty → zero-shot).
        mode: "text" or "multimodal".
        cot: if True, include Chain-of-Thought reasoning instructions.
    """
    target_has_cutout = has_image(target_id, "cutout.png")
    system_text = _make_system_prompt(cot=cot, has_cutout=target_has_cutout)

    user_content = []

    # Few-shot examples (skip if zero-shot)
    if few_shot:
        user_content.append({"type": "text", "text":
                             "## Few-Shot Examples\n"
                             "Study these labeled examples using the physical rules above.\n"})

        for fs_id, fs_label in few_shot:
            user_content.append({"type": "text", "text": f"### Example: {fs_label} ({fs_id})\n"})

            if mode == "multimodal":
                for img_name in config.MULTIMODAL_IMAGES:
                    if has_image(fs_id, img_name):
                        b64 = read_image_b64(fs_id, img_name)
                        if b64:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            })

            md_text = read_md(fs_id)
            user_content.append({"type": "text", "text": md_text + "\n"})

    # Target source
    target_header = "\n---\n\n"
    
    # Flux-scale note: color should use mag (scale-independent). Only decline rate
    # thresholds need scale adjustment since they're still in uJy/d.
    peak = _peak_flux(target_id)
    if peak and (peak > 500 or peak < 100):
        adj_decline = max(round(peak * 0.005), 1)
        adj_slow = max(round(peak * 0.0005), 0.1)
        target_header += (
            f"**Scale Note:** this source has peak={peak:.0f} uJy. "
            f"Color thresholds are in MAG (scale-independent, see system prompt). "
            f"For decline rate: steep > {adj_decline} uJy/d, slow < {adj_slow} uJy/d.\n\n"
        )


    if cot:
        target_header += (
            "## Reasoning Task\n"
            "Reason through each physical discriminator step by step "
            "(Step 1: Shape Gate → Step 2: Optical Color → "
            "Step 3: WISE → Step 4: Joint Matrix → Step 5: Tiebreakers),\n"
            "then output the classification as JSON.\n\n"
        )

    target_header += "## Target: Classify This Source\n\n"
    user_content.append({"type": "text", "text": target_header})

    if mode == "multimodal":
        for img_name in config.MULTIMODAL_IMAGES:
            if has_image(target_id, img_name):
                b64 = read_image_b64(target_id, img_name)
                if b64:
                    label = img_name.replace(".png", "").replace("_", " ")
                    user_content.append({"type": "text", "text": f"[Image: {label}]\n"})
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })

    target_md = read_md(target_id)
    user_content.append({"type": "text", "text": target_md})

    if mode == "text":
        # Flatten to single string for text-only API
        flat = "\n".join(
            item["text"] for item in user_content if item["type"] == "text"
        )
        return [{"role": "system", "content": system_text},
                {"role": "user", "content": flat}]

    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user_content}]


# ═══════════════════════════════════════════════════
# API call
# ═══════════════════════════════════════════════════

MAX_RETRIES = 3

def call_api(messages, model=None):
    """Call the LLM API and return response text."""
    if model is None:
        model = config.MODEL
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                temperature=config.TEMPERATURE,
                messages=messages,
                max_tokens=3000,
            )
            msg = response.choices[0].message
            # Handle USTC API proxy: content may be None, fall back to reasoning_content
            text = msg.content or getattr(msg, "reasoning_content", None) or ""
            return text, response.usage
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e}")
                import time
                time.sleep(2 ** attempt)
    raise last_error


# ═══════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════

def parse_response(raw_text):
    """Extract JSON from LLM response. Handles markdown code blocks."""
    text = raw_text.strip()
    # Try to extract from ```json ... ``` block
    if "```json" in text:
        start = text.index("```json") + 7
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            text = text[start:].strip()  # truncated block, try as-is
    elif "```" in text:
        start = text.index("```") + 3
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            text = text[start:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find first { and last }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    return {"error": "json_parse_failed", "raw": raw_text}


def _extract_reasoning(raw_text, parsed):
    """Extract CoT reasoning text by removing the JSON part from raw response."""
    if not raw_text:
        return ""
    # Try to find where the JSON starts and take text before it
    json_str = json.dumps(parsed, ensure_ascii=False)
    # Match by first 60 chars of JSON (robust against whitespace variations)
    json_head = json_str[:min(60, len(json_str))]
    idx = raw_text.find(json_head)
    if idx >= 0:
        return raw_text[:idx].strip()
    # Fallback: everything before the first {
    for i, ch in enumerate(raw_text):
        if ch == '{':
            return raw_text[:i].strip()
    return raw_text.strip()


# ═══════════════════════════════════════════════════
# Result saving
# ═══════════════════════════════════════════════════

def save_result(source_id, parsed, raw_response, usage, mode, model, few_shot, cot=False):
    result = {
        "source_id": source_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "tokens": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
        "classification": parsed.get("classification", {}),
        "reasoning": parsed.get("reasoning", {}),
        "quality": parsed.get("quality", {}),
        "few_shot": [{"id": fs[0], "label": fs[1]} for fs in few_shot],
        "cot": cot,
        "cot_reasoning": _extract_reasoning(raw_response, parsed),
        "_raw_response": raw_response,
    }
    out_path = config.RESULTS_DIR / f"{source_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ═══════════════════════════════════════════════════
# Classify one source
# ═══════════════════════════════════════════════════

def classify_one(source_id, mode="text", n_shot=None, model=None, force=False, cot=False,
                exemplar_set=None):
    """Classify a single source.
    
    Args:
        cot: if True, enable Chain-of-Thought reasoning.
        exemplar_set: name of curated exemplar set (e.g. "boundary", "textbook").
                      Loads templates/fewshot_{exemplar_set}.json.
    """
    # Check if already done
    result_path = config.RESULTS_DIR / f"{source_id}.json"
    if result_path.exists() and not force:
        print(f"  [skip] {source_id} -- result already exists (use --force to redo)")
        return None

    # Check source exists
    md_path = config.SOURCES_DIR / source_id / "analysis.md"
    if not md_path.exists():
        print(f"  [error] {source_id} -- analysis.md not found. Run promt.py first.")
        return None

    # Load index
    idx = promt_load_index()

    # Sample few-shot (exclude the target itself)
    if n_shot is None:
        n_shot = config.N_SHOT_MULTIMODAL if mode == "multimodal" else config.N_SHOT_TEXT

    few_shot = sample_few_shot(idx, n_per_class=n_shot, exclude={source_id},
                               exemplar_set=exemplar_set)
    if n_shot > 0 and not few_shot:
        print(f"  [error] {source_id} -- no labeled examples available in index.json")
        return None

    # Build prompt
    mode_str = f"{mode}, {len(few_shot)}-shot"
    if cot:
        mode_str += ", CoT"
    print(f"  {source_id}: building prompt ({mode_str})")
    messages = build_prompt(source_id, few_shot, mode, cot=cot)

    # Call API
    raw_text, usage = call_api(messages, model=model)

    # Parse
    parsed = parse_response(raw_text)

    # Save
    result = save_result(source_id, parsed, raw_text, usage, mode,
                         model or config.MODEL, few_shot, cot=cot)

    # Print summary
    cls_info = result.get("classification", {})
    label = cls_info.get("label", "?")
    conf = cls_info.get("confidence", "?")
    score = cls_info.get("score", 0)
    print(f"  [done] {source_id} -> {label} ({conf}, score={score:.2f}) "
          f"[{result['tokens']['total']} tokens]")

    return result


def classify_all_unlabeled(mode="text", n_shot=None, model=None, force=False, cot=False,
                          exemplar_set=None):
    """Classify all sources labeled 'unknown' in index.json."""
    idx = promt_load_index()
    unlabeled = [sid for sid, info in idx.items() if info["label"] == "unknown"]
    if not unlabeled:
        print("No unlabeled sources found in index.json")
        return
    print(f"Classifying {len(unlabeled)} unlabeled sources...")
    done = 0
    for sid in sorted(unlabeled):
        result = classify_one(sid, mode=mode, n_shot=n_shot, model=model, force=force, cot=cot,
                              exemplar_set=exemplar_set)
        if result:
            done += 1
    print(f"\nDone: {done}/{len(unlabeled)} classified")


def show_result(source_id):
    path = config.RESULTS_DIR / f"{source_id}.json"
    if not path.exists():
        print(f"No result found for {source_id}")
        return
    r = json.loads(path.read_text())
    c = r.get("classification", {})
    q = r.get("quality", {})
    reasoning = r.get("reasoning", {})
    print(f"\n{'='*60}")
    print(f"  {source_id}")
    print(f"{'='*60}")
    print(f"  Classification: {c.get('label', '?')}  ({c.get('confidence', '?')}, score={c.get('score', 0):.2f})")
    print(f"  Model: {r.get('model', '?')}  |  Mode: {r.get('mode', '?')}  |  {r['tokens']['total']} tokens")
    if reasoning.get("summary"):
        print(f"\n  Summary: {reasoning['summary']}")
    if reasoning.get("indicators"):
        print(f"\n  Key Indicators:")
        for ind in reasoning["indicators"]:
            direction_mark = "->TDE" if ind.get("direction") == "TDE" else ("->SN" if ind.get("direction") == "SN" else "")
            print(f"    [{ind.get('weight', 0):.2f}] {ind.get('name', '?')}: {ind.get('value', '?')} {direction_mark}")
    if q.get("flags"):
        print(f"\n  Quality Flags:")
        for f in q["flags"]:
            if isinstance(f, dict):
                print(f"    [{f.get('severity', '?')}] {f.get('flag', '?')}: {f.get('detail', '?')}")
            else:
                print(f"    - {f}")
    print(f"{'='*60}\n")


def promt_load_index():
    if config.INDEX_FILE.exists():
        return json.loads(config.INDEX_FILE.read_text())
    return {}


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="classify.py - LLM-based light curve classification")
    parser.add_argument("source_id", nargs="?", help="source ID to classify or show results")
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text")
    parser.add_argument("--n-shot", type=int, help="few-shot examples per class")
    parser.add_argument("--model", help="model override")
    parser.add_argument("--all-unlabeled", action="store_true", help="classify all unlabeled")
    parser.add_argument("--results", action="store_true", help="show saved results")
    parser.add_argument("--force", action="store_true", help="reclassify even if result exists")
    parser.add_argument("--cot", action="store_true", help="enable Chain-of-Thought reasoning")
    parser.add_argument("--exemplar-set", help="curated exemplar set name (loads templates/fewshot_NAME.json)")
    args = parser.parse_args()

    if args.all_unlabeled:
        classify_all_unlabeled(mode=args.mode, n_shot=args.n_shot, model=args.model,
                               force=args.force, cot=args.cot,
                               exemplar_set=args.exemplar_set)
    elif args.results and args.source_id:
        show_result(args.source_id)
    elif args.source_id:
        classify_one(args.source_id, mode=args.mode, n_shot=args.n_shot, model=args.model,
                     force=args.force, cot=args.cot,
                     exemplar_set=args.exemplar_set)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()