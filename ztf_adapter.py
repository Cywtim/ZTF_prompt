#!/usr/bin/env python3
"""
ztf_adapter.py v3 — ZTF Lasair npy → sources/{name}_PRF/

Generates analysis.md compatible with classify.py (matching promt.py structure):
  Section 1: Source Metadata (with rise/decline)
  Section 2: Derived Features
    2.1 Global Morphology
    2.2 Color Evolution (g-r only, ZTF has no u band)
    2.3 Per-Phase Summary
    2.4 Data Quality Flags
    2.5 Baseline Detection
  Section 3: Raw Light Curve (with g-r color columns)
  Section 4: Classification Protocol

Usage:
  python ztf_adapter.py ZTF21aaaokyp --ra 176.65 --dec 30.09
  python ztf_adapter.py --batch tde_list.txt
"""
import sys, os, argparse, urllib.request
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Config ---
PROJECT_ROOT = Path(__file__).parent
SOURCES_DIR = PROJECT_ROOT / "sources"
ZTF_DATA_DIR = Path(os.environ.get(
    "ZTF_DATA_DIR",
    str(Path.home() / "AppData" / "VScode" / "TDeck" / "ZTF_TDE" / "data"),
))
ZTF_FLUX_TDE = ZTF_DATA_DIR / "TS" / "Flux" / "TDE"

def get_flux_dir(category="TDE"):
    """Get flux data directory for a given category."""
    cat_map = {"TDE": "TDE", "SN": "SN", "AGN": "AGN", "Others": "Others", "unlabeled": "unlabeled"}
    return ZTF_DATA_DIR / "TS" / "Flux" / cat_map.get(category, category)

BAND_STR = {1: "g", 2: "r"}
BAND_COLORS = {"g": "#2ca02c", "r": "#d62728"}

# Baseline detection params
BASELINE_FRAC = 0.10
BASELINE_ABS_FLOOR = 30.0
BASELINE_MIN_CONSEC = 3
BASELINE_MIN_SPAN = 5.0
BASELINE_TAIL_FRAC = 0.20
BASELINE_TAIL_MED_THRESH = 0.15

# Burst trimming
TRIM_MIN_PTS = 30
TRIM_POST_PEAK_FRAC = 0.15  # flux falls below this fraction → trim tail


def load_ztf_npy(ztf_name, category="TDE"):
    flux_dir = get_flux_dir(category)
    path = flux_dir / f"{ztf_name}_difference_photometry_flux.npy"
    if not path.exists():
        raise FileNotFoundError(f"ZTF data not found: {path}")
    arr = np.load(path)
    return arr[arr[:, 0].argsort()]


# ══════════════════════════════════════════
# Color computation (adapted from promt.py)
# ══════════════════════════════════════════

def _interp_flux(target_mjd, ref_mjd, ref_flux):
    """Linearly interpolate ref flux to target_mjd."""
    n_ref = len(ref_mjd)
    if n_ref == 0:
        return None
    if n_ref == 1:
        return float(ref_flux[0])
    # Sort ref by MJD
    order = np.argsort(ref_mjd)
    rmjd = ref_mjd[order]
    rflux = ref_flux[order]
    # Find bracketing indices
    idx = np.searchsorted(rmjd, target_mjd)
    if idx == 0:
        return float(rflux[0])
    if idx >= n_ref:
        return float(rflux[-1])
    # Linear interpolation
    frac = (target_mjd - rmjd[idx - 1]) / (rmjd[idx] - rmjd[idx - 1])
    return float(rflux[idx - 1] + frac * (rflux[idx] - rflux[idx - 1]))


def compute_color_pairs(arr):
    """Compute g-r colors at matching MJDs.

    Returns arrays aligned with arr: color_gr_mag, color_gr_flux (or NaN if no pair).
    """
    mjd, band, flux = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
    g_mask = band == 1
    r_mask = band == 2

    g_mjd = mjd[g_mask]
    g_flux = flux[g_mask]
    r_mjd = mjd[r_mask]
    r_flux = flux[r_mask]

    n = len(arr)
    color_gr_mag = np.full(n, np.nan)
    color_gr_flux = np.full(n, np.nan)

    for i in range(n):
        if band[i] == 1:  # g-band point → interpolate r
            r_at_mjd = _interp_flux(mjd[i], r_mjd, r_flux)
            if r_at_mjd is not None and r_at_mjd > 0 and flux[i] > 0:
                color_gr_mag[i] = -2.5 * np.log10(flux[i] / r_at_mjd)
                color_gr_flux[i] = flux[i] - r_at_mjd
        elif band[i] == 2:  # r-band point → interpolate g
            g_at_mjd = _interp_flux(mjd[i], g_mjd, g_flux)
            if g_at_mjd is not None and g_at_mjd > 0 and flux[i] > 0:
                color_gr_mag[i] = -2.5 * np.log10(g_at_mjd / flux[i])
                color_gr_flux[i] = g_at_mjd - flux[i]

    return color_gr_mag, color_gr_flux


# ══════════════════════════════════════════
# Baseline detection
# ══════════════════════════════════════════

def _is_quiet(flux_abs, threshold):
    return flux_abs < threshold


def _find_quiet_blocks(indices, quiet_flags):
    blocks = []
    current = []
    for i, q in zip(indices, quiet_flags):
        if q:
            current.append(i)
        else:
            if len(current) >= BASELINE_MIN_CONSEC:
                blocks.append(np.array(current))
            current = []
    if len(current) >= BASELINE_MIN_CONSEC:
        blocks.append(np.array(current))
    return blocks


def detect_baseline(arr):
    mjd, band, flux = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
    peak_idx = np.argmax(flux)
    peak_mjd, peak_flux = mjd[peak_idx], flux[peak_idx]

    threshold = max(BASELINE_FRAC * peak_flux, BASELINE_ABS_FLOOR)
    quiet = _is_quiet(np.abs(flux), threshold)

    result = {
        "pre_detected": False, "pre_mjd_start": None, "pre_mjd_end": None,
        "pre_n_pts": 0, "pre_mean_flux": 0,
        "post_detected": False, "post_mjd_start": None, "post_mjd_end": None,
        "post_n_pts": 0, "post_mean_flux": 0,
        "warnings": [], "peak_mjd": peak_mjd, "peak_flux": peak_flux,
        "threshold": threshold,
    }

    pre_mask = mjd < peak_mjd
    post_mask = mjd > peak_mjd

    # Pre-peak
    if pre_mask.sum() >= BASELINE_MIN_CONSEC:
        pre_indices = np.where(pre_mask)[0]
        blocks = _find_quiet_blocks(pre_indices, quiet[pre_mask])
        if blocks:
            last_block = blocks[-1]
            block_mjd = mjd[last_block]
            if len(last_block) >= BASELINE_MIN_CONSEC and (block_mjd.max() - block_mjd.min()) >= BASELINE_MIN_SPAN:
                result["pre_detected"] = True
                result["pre_mjd_start"] = block_mjd.min()
                result["pre_mjd_end"] = block_mjd.max()
                result["pre_n_pts"] = len(last_block)
                result["pre_mean_flux"] = float(np.mean(flux[last_block]))
            elif len(last_block) >= BASELINE_MIN_CONSEC:
                result["warnings"].append(
                    f"Pre-peak quiet block ({len(last_block)} pts, "
                    f"{block_mjd.max()-block_mjd.min():.0f}d) too short to confirm baseline"
                )
    else:
        result["warnings"].append("Insufficient pre-peak data")

    # Post-peak
    if post_mask.sum() >= BASELINE_MIN_CONSEC:
        post_indices = np.where(post_mask)[0]
        blocks = _find_quiet_blocks(post_indices, quiet[post_mask])
        if blocks:
            last_block = blocks[-1]
            block_mjd = mjd[last_block]
            if len(last_block) >= BASELINE_MIN_CONSEC and (block_mjd.max() - block_mjd.min()) >= BASELINE_MIN_SPAN:
                result["post_detected"] = True
                result["post_mjd_start"] = block_mjd.min()
                result["post_mjd_end"] = block_mjd.max()
                result["post_n_pts"] = len(last_block)
                result["post_mean_flux"] = float(np.mean(flux[last_block]))
        if not result["post_detected"]:
            post_mjd_vals = mjd[post_mask]
            post_flux_vals = np.abs(flux[post_mask])
            tail_n = max(BASELINE_MIN_CONSEC, int(len(post_flux_vals) * BASELINE_TAIL_FRAC))
            if tail_n > 0 and len(post_flux_vals) >= 10:
                tail_flux = post_flux_vals[-tail_n:]
                tail_median = np.median(tail_flux)
                if tail_median < BASELINE_TAIL_MED_THRESH * peak_flux:
                    result["post_detected"] = True
                    tail_mjd = post_mjd_vals[-tail_n:]
                    result["post_mjd_start"] = tail_mjd.min()
                    result["post_mjd_end"] = tail_mjd.max()
                    result["post_n_pts"] = tail_n
                    result["post_mean_flux"] = float(tail_median)
    else:
        result["warnings"].append("Insufficient post-peak data")

    # Pre-peak activity check
    if result["pre_detected"] and pre_mask.sum() > result["pre_n_pts"]:
        pre_flux_outside = flux[pre_mask][~quiet[pre_mask]]
        if len(pre_flux_outside) > 0 and np.max(pre_flux_outside) > 0.3 * peak_flux:
            result["warnings"].append(
                f"Pre-peak activity detected: max outside baseline = "
                f"{np.max(pre_flux_outside):.0f} μJy ({np.max(pre_flux_outside)/peak_flux*100:.0f}% of peak)"
            )

    return result


# ══════════════════════════════════════════
# Burst trimming (Lasair diff photometry artifact mitigation)
# ══════════════════════════════════════════

def trim_to_burst(arr, baseline):
    """Trim to burst period to remove Lasair diff photometry artifacts.

    Rule: if pre-peak baseline is NOT detected, the early data may show a false
    "decline" caused by the reference image being taken mid-transient.
    Trim everything before the pre-peak flux minimum to isolate the real rise.

    Also trims post-peak tail where flux falls below TRIM_POST_PEAK_FRAC of peak.

    Returns (trimmed_arr, trim_info).
    If trimmed result has < TRIM_MIN_PTS points, returns original + warning.
    """
    mjd, band, flux, fluxerr = arr[:, 0], arr[:, 1].astype(int), arr[:, 2], arr[:, 3]
    peak_idx = int(np.argmax(flux))
    peak_flux = flux[peak_idx]

    trim_info = {
        "applied": False,
        "cut_pre": 0,
        "cut_post": 0,
        "n_original": len(arr),
        "n_kept": len(arr),
        "reason": "",
    }

    # --- Determine cuts on ORIGINAL indices ---
    cut_pre = 0
    if not baseline["pre_detected"]:
        pre_fluxes = flux[:peak_idx + 1]
        if len(pre_fluxes) >= 3:
            pre_min_idx = int(np.argmin(pre_fluxes))
            cut_pre = pre_min_idx
            trim_info["reason"] = (
                "Pre-peak baseline not detected. "
                "Trimmed {n} early points (Lasair reference likely mid-transient)."
            ).format(n=cut_pre)

    cut_post = 0
    # Only trim post-peak tail if baseline detection confirmed a quiet zone
    if baseline["post_detected"] and baseline["post_mjd_start"] is not None:
        # Cut from baseline start onward
        post_start_mjd = baseline["post_mjd_start"]
        for i in range(peak_idx + 1, len(mjd)):
            if mjd[i] >= post_start_mjd:
                cut_post = len(arr) - i
                break

    # --- Apply cuts on original array ---
    if cut_pre > 0 or cut_post > 0:
        keep_start = cut_pre
        keep_end = len(arr) - cut_post
        trimmed = arr[keep_start:keep_end]

        if len(trimmed) >= TRIM_MIN_PTS:
            trim_info["applied"] = True
            trim_info["cut_pre"] = cut_pre
            trim_info["cut_post"] = cut_post
            trim_info["n_kept"] = len(trimmed)
            return trimmed, trim_info
        else:
            trim_info["reason"] += (
                f" TRIM ABORTED: would leave {len(trimmed)} pts (<{TRIM_MIN_PTS}). "
                f"Data kept as-is with artifact warning added."
            )
            return arr, trim_info

    return arr, trim_info


# ══════════════════════════════════════════
# Helpers (adapted from promt.py)
# ══════════════════════════════════════════

def _mag_or_dash(val):
    if np.isnan(val):
        return "--"
    return f"{val:+.3f}"


def _clevel(n):
    if n >= 15:
        return "HIGH"
    elif n >= 5:
        return "MEDIUM"
    return "LOW"


def _ctrend(values):
    """Classify color trend: Red, Blue, or Flat."""
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return "Flat"
    # Simple: mean of first half vs mean of second half
    mid = len(valid) // 2
    first = np.mean(valid[:mid]) if mid > 0 else valid[0]
    last = np.mean(valid[mid:]) if mid < len(valid) else valid[-1]
    diff = last - first
    if diff > 0.05:
        return "Red"
    elif diff < -0.05:
        return "Blue"
    return "Flat"


def _phase_trend(phase_vals):
    """Classify phase trend."""
    if len(phase_vals) < 2:
        return "flat"
    diff = phase_vals[-1] - phase_vals[0]
    if diff > 0.1:
        return "rise"
    elif diff < -0.1:
        return "fall"
    return "flat"


# ══════════════════════════════════════════
# Analysis generation
# ══════════════════════════════════════════

def compute_features(arr):
    """Compute all features for analysis.md."""
    mjd, band, flux, fluxerr = arr[:, 0], arr[:, 1].astype(int), arr[:, 2], arr[:, 3]
    n = len(arr)
    span = mjd.max() - mjd.min()
    t0 = mjd.min()
    day = mjd - t0

    gn = int((band == 1).sum())
    rn = int((band == 2).sum())

    pi = int(np.argmax(flux))
    pk_day = day[pi]
    pk_flux = flux[pi]
    pk_band = BAND_STR[int(band[pi])]
    pk_mjd = mjd[pi]

    # Rise time (days from first point to peak)
    rise_t = pk_day

    # Decline rate (linear fit to post-peak)
    post = day[day >= pk_day]
    post_flux_vals = flux[day >= pk_day]
    if len(post) >= 3:
        drate = np.polyfit(post, post_flux_vals, 1)[0]
    else:
        drate = 0.0

    # Rise rate
    rise_rate = (pk_flux - flux[0]) / max(rise_t, 1)

    # Compute colors
    color_gr_mag, color_gr_flux = compute_color_pairs(arr)
    gr_valid = (~np.isnan(color_gr_mag)).sum()

    # g-r at peak
    gr_peak = color_gr_mag[pi] if not np.isnan(color_gr_mag[pi]) else None

    # Cadence
    mjd_sorted = np.sort(np.unique(mjd))
    cadence = np.mean(np.diff(mjd_sorted)) if len(mjd_sorted) > 1 else 0

    # Pre-peak days
    pre_mask = mjd < pk_mjd
    pre_days = pk_mjd - mjd.min() if pre_mask.sum() > 0 else 0

    # Post-peak points
    post_n = int((day >= pk_day).sum())

    return {
        "n": n, "span": span, "t0": t0, "day": day,
        "gn": gn, "rn": rn,
        "pk_day": pk_day, "pk_flux": pk_flux, "pk_band": pk_band, "pk_mjd": pk_mjd,
        "rise_t": rise_t, "drate": drate, "rise_rate": rise_rate,
        "color_gr_mag": color_gr_mag, "color_gr_flux": color_gr_flux,
        "gr_valid": gr_valid, "gr_peak": gr_peak,
        "cadence": cadence, "pre_days": pre_days,
        "post_n": post_n, "mjd": mjd, "pi": pi, "flux_min": flux.min(),
        "flux_max": flux.max(), "fluxerr": fluxerr,
    }


def generate_analysis_md(arr, f, baseline, ztf_name, source_dir, trim_info=None, label="TDE"):
    """Generate analysis.md compatible with classify.py (promt.py structure)."""
    if trim_info is None:
        trim_info = {"applied": False, "cut_pre": 0, "cut_post": 0,
                     "n_original": len(arr), "n_kept": len(arr), "reason": ""}
    n, span, t0, day = f["n"], f["span"], f["t0"], f["day"]
    bl = baseline

    L = []

    # ============ Section 1: Metadata ============
    L.append(f"# {ztf_name}_PRF — ZTF Light Curve Analysis")
    L.append("")
    L.append("## Section 1: Source Metadata")
    L.append("")
    L.append("| Property | Value |")
    L.append("|----------|-------|")
    L.append(f"| Source ID | {ztf_name}_PRF |")
    L.append(f"| Survey | ZTF |")
    L.append(f"| Bands | g, r |")
    L.append(f"| Label | {label} |")
    L.append(f"| MJD range | {t0:.1f} to {t0+span:.1f} (span = {span:.0f} d) |")
    L.append(f"| Total points | {n} (g: {f['gn']}, r: {f['rn']}) |")
    L.append(f"| Peak flux | {f['pk_flux']:.1f} μJy ({f['pk_band']}-band, day {f['pk_day']:.1f}) |")
    L.append(f"| Rise time | {f['rise_t']:.0f} d |")
    L.append(f"| Decline rate | {f['drate']:+.3f} μJy/d |")
    L.append(f"| Cadence | ~{f['cadence']:.1f} d mean |")
    L.append(f"| g-r at peak | {f['gr_peak']:.3f} mag |" if f['gr_peak'] is not None else "| g-r at peak | N/A |")
    L.append("")

    # ============ Section 2: Derived Features ============
    L.append("## Section 2: Derived Features")
    L.append("")

    # --- 2.1 Global Morphology ---
    L.append("### 2.1 Global Morphology")
    L.append("")
    L.append("| Indicator | Value | Hint |")
    L.append("|-----------|-------|------|")
    rise_hint = "fast" if f["rise_t"] < 30 else ("moderate" if f["rise_t"] < 60 else "slow")
    L.append(f"| Rise rate | +{f['rise_rate']:.2f} μJy/d | {rise_hint} |")
    dec_hint = "power-law" if abs(f["drate"]) < 0.5 else "steep"
    L.append(f"| Decline | {f['drate']:+.3f} μJy/d | {dec_hint} |")
    L.append("")

    # --- 2.2 Color Evolution ---
    L.append("### 2.2 Color Evolution (mag)")
    L.append("")
    L.append("> TDE g−r zone: ∈ (−0.6, +0.1) mag. Red→Blue (Δ<−0.15) = TDE-like.")
    L.append("> Blue→Red (Δ>+0.15) = SN-like. Flat = unclear.")
    L.append("")

    # g-r subtable
    gr_mag = f["color_gr_mag"]
    gr_flux = f["color_gr_flux"]
    ec = gr_mag[day <= span * 0.3]
    mc = gr_mag[(day > span * 0.3) & (day < span * 0.7)]
    lc = gr_mag[day >= span * 0.7]
    ec_valid = ec[~np.isnan(ec)]
    mc_valid = mc[~np.isnan(mc)]
    lc_valid = lc[~np.isnan(lc)]

    if len(ec_valid) + len(mc_valid) + len(lc_valid) > 0:
        L.append("#### g − r")
        L.append("")
        L.append("| Phase | N pairs | g-r (mag) | sigma | g-r (μJy) | Trend |")
        L.append("|-------|:-------:|:---------:|:-----:|:---------:|:-----:|")

        for clabel, c_arr, fc_arr in [
            ("Early (0-30 pct)", ec, gr_flux[day <= span * 0.3]),
            ("Mid (30-70 pct)", mc, gr_flux[(day > span * 0.3) & (day < span * 0.7)]),
            ("Late (70-100 pct)", lc, gr_flux[day >= span * 0.7]),
        ]:
            valid = c_arr[~np.isnan(c_arr)]
            if len(valid) == 0:
                L.append(f"| {clabel} | 0 | -- | -- | -- | -- |")
            else:
                f_valid = fc_arr[~np.isnan(c_arr)]
                L.append(
                    f"| {clabel} | {len(valid)} | {np.nanmean(c_arr):+.3f} | "
                    f"{np.nanstd(c_arr):.3f} | {np.mean(f_valid):+.1f} | {_ctrend(c_arr)} |"
                )

        if len(ec_valid) and len(lc_valid):
            dc = np.nanmean(lc_valid) - np.nanmean(ec_valid)
            if abs(dc) < 0.05:
                evo = "Flat (no clear evolution)"
            elif np.nanmean(ec_valid) > np.nanmean(lc_valid):
                evo = "Red to Blue (TDE-like)"
            else:
                evo = "Blue to Red (SN-like)"
            L.append(
                f"| **Evolution** | -- | delta = {dc:+.3f} mag | -- | -- | **{evo}** |"
            )
        L.append("")
    else:
        L.append("*No g-r color data available.*")
        L.append("")

    # --- 2.3 Per-Phase Summary ---
    L.append("### 2.3 Per-Phase Summary")
    L.append("")
    L.append("| Cutoff | N pts | Bands (g/r) | Phase | g-r (mag) | Trend |")
    L.append("|--------|:-----:|:-----------:|:-----:|:---------:|:-----:|")

    for pct in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
        mask = day <= span * pct
        n_mask = int(mask.sum())
        if n_mask < 2:
            continue
        b = arr[mask, 1].astype(int)
        bg = int((b == 1).sum())
        br = int((b == 2).sum())

        # Phase
        phase_vals = f["day"][mask] / max(f["day"][mask].max(), 1)

        # g-r at latest points
        tn = min(5, n_mask)
        c_gr = gr_mag[mask]
        cm_gr = np.nanmean(c_gr[-tn:]) if tn > 0 else np.nan
        cs_gr = np.nanstd(c_gr[-tn:]) if tn > 1 else 0

        L.append(
            f"| {int(pct*100)}% | {n_mask} | {bg}/{br} | "
            f"{_phase_trend(f['day'][mask]):5s} | "
            f"{_mag_or_dash(cm_gr)} +/- {cs_gr:.3f} | {_ctrend(c_gr)} |"
        )
    L.append("")

    # --- 2.4 Data Quality Flags ---
    L.append("### 2.4 Data Quality Flags")
    L.append("")
    L.append("| Feature | Confidence | Reason |")
    L.append("|---------|:----------:|--------|")
    early_n = int((day <= span * 0.1).sum()) if span > 0 else 0
    L.append(f"| Rise phase | {_clevel(early_n)} | {early_n} pts in 10% window |")
    gr_ec = gr_mag[day <= span * 0.3]
    gr_lc = gr_mag[day >= span * 0.7]
    L.append(f"| Color coverage | {_clevel(max((~np.isnan(gr_ec)).sum(), (~np.isnan(gr_lc)).sum()))} | {f['gr_valid']} g-r pairs |")
    L.append(f"| Decline | {_clevel(f['post_n'])} | {f['post_n']} post-peak pts |")
    L.append(f"| Pre-peak data | {_clevel(int(day[day < f['pk_day']].sum()) if f['pk_day'] > 0 else 0)} | {f['pre_days']:.0f} d pre-peak |")
    L.append("")

    # --- 2.5 Baseline Detection ---
    L.append("### 2.5 Baseline Detection")
    L.append("")
    L.append(f"> Threshold: |flux| < {bl['threshold']:.1f} μJy = quiet.")
    L.append("")
    L.append("| Zone | Detected | MJD Range | N pts | Mean Flux |")
    L.append("|------|:--------:|-----------|:-----:|:---------:|")
    for zone, detected, start, end, n_pts, mean_f in [
        ("Pre-peak", bl["pre_detected"], bl["pre_mjd_start"], bl["pre_mjd_end"],
         bl["pre_n_pts"], bl["pre_mean_flux"]),
        ("Post-peak", bl["post_detected"], bl["post_mjd_start"], bl["post_mjd_end"],
         bl["post_n_pts"], bl["post_mean_flux"]),
    ]:
        if detected:
            L.append(f"| {zone} | ✅ YES | {start:.1f} – {end:.1f} | {n_pts} | {mean_f:.0f} μJy |")
        else:
            L.append(f"| {zone} | ❌ NO | — | — | — |")
    L.append("")
    if bl["warnings"]:
        L.append("**⚠️ Warnings:**")
        for w in bl["warnings"]:
            L.append(f"- {w}")
        L.append("")
    L.append("**Instruction:** Baseline zones marked YES represent quiescent periods — IGNORE for transient shape analysis.")
    L.append("")

    # --- 2.5b Burst Trimming ---
    if trim_info["applied"]:
        L.append("### 2.6 Burst Trimming (ZTF Diff Photometry Artifact Mitigation)")
        L.append("")
        L.append(f"> ⚠️ **Trimmed {trim_info['cut_pre']} pre-peak + {trim_info['cut_post']} post-peak points.**")
        L.append(f"> {trim_info['reason']}")
        L.append(f"> Original: {trim_info['n_original']} pts → Kept: {trim_info['n_kept']} pts")
        L.append("")
        L.append("**Why:** ZTF Lasair difference photometry subtracts a reference image. If the")
        L.append("reference was taken during the transient's rise, early data shows a spurious")
        L.append("\"decline\" that is NOT part of the real light curve. Section 3 below contains")
        L.append("ONLY the trimmed burst period — analyze THIS, not the original artifact region.")
        L.append("")
    elif trim_info.get("reason") and "ABORTED" in trim_info.get("reason", ""):
        L.append("### 2.6 Burst Trimming — ABORTED")
        L.append("")
        L.append(f"> ⚠️ {trim_info['reason']}")
        L.append("")
        L.append("**⚠️ Diff Photometry Warning:** Pre-peak baseline not detected. The early data")
        L.append("may show a spurious \"decline\" caused by the Lasair reference image being taken")
        L.append("mid-transient. DO NOT interpret early flux decline as part of the light curve shape.")
        L.append("Judge the transient by its rise from the flux minimum onward.")
        L.append("")

    # ============ Section 3: Raw Light Curve ============
    L.append("## Section 3: Raw Light Curve")
    L.append("")
    L.append(f"> Flux in μJy. Day = MJD - {t0:.1f}. Band: g=1, r=2.")
    L.append("")
    L.append("| Num | Day | B | Flux | Err | Phase | g-r (mag) | g-r (μJy) |")
    L.append("|-----|-----|---|:----:|:---:|:-----:|:---------:|:---------:|")

    for i in range(n):
        mjd_val, band_int, flux_val, fluxerr_val = arr[i, 0], int(arr[i, 1]), arr[i, 2], arr[i, 3]
        band_str = BAND_STR.get(band_int, "?")
        phase = -1 if mjd_val < f["pk_mjd"] else +1
        ph_str = f"{day[i]/max(span,1):+.3f}" if span > 0 else "+0.000"
        gr_m = _mag_or_dash(gr_mag[i])
        gr_f = f"{gr_flux[i]:+.1f}" if not np.isnan(gr_flux[i]) else "--"
        L.append(
            f"| {i+1} | {day[i]:.1f} | {band_str} | {flux_val:.2f} | {fluxerr_val:.2f} | "
            f"{ph_str} | {gr_m} | {gr_f} |"
        )
    L.append("")

    # ============ Section 4: Classification Protocol ============
    L.append("## Section 4: Classification Protocol")
    L.append("")
    L.append("### System Instruction")
    L.append("Classify this transient light curve as **TDE / SN / AGN / Others**.")
    L.append("")
    L.append("### ZTF-Specific Knowledge")
    L.append("- **TDE:** g−r ∈ (−0.6, +0.1), red→blue evolution, fast rise + power-law decay, late plateau")
    L.append("- **SN:** diverse colors, Ni-56 decay tail, SLSNe show long plateaus")
    L.append("- **AGN:** stochastic variability, pre-transient activity, red WISE colors")
    L.append("- **Key discriminators:** color evolution direction, plateau presence, decline rate, baseline")
    L.append("")

    L.append("### Output Format")
    L.append("Return ONLY valid JSON:")
    L.append("```")
    L.append('{"classification":{"label":"TDE","confidence":"medium","score":0.6},')
    L.append('"reasoning":{"summary":"...","feature_based":"...","raw_audit":"...",')
    L.append('"indicators":[{"name":"...","value":"...","weight":0.3,"direction":"TDE","note":"..."}]},')
    L.append('"quality":{"overall":"medium","flags":[]}}')
    L.append("```")

    md_text = "\n".join(L) + "\n"
    md_path = source_dir / "analysis.md"
    md_path.write_text(md_text)
    return md_path


# ══════════════════════════════════════════
# Plotting, cutout, main
# ══════════════════════════════════════════

def generate_lightcurve_plot(arr, baseline, ztf_name, source_dir, trim_info=None):
    mjd, band, flux = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    mjd0 = mjd.min()
    day = mjd - mjd0

    if baseline["pre_detected"]:
        ax.axvspan(baseline["pre_mjd_start"] - mjd0, baseline["pre_mjd_end"] - mjd0,
                   alpha=0.12, color="gray")
    if baseline["post_detected"]:
        ax.axvspan(baseline["post_mjd_start"] - mjd0, baseline["post_mjd_end"] - mjd0,
                   alpha=0.12, color="gray")

    # Mark trimmed region if applicable
    if trim_info and trim_info.get("applied"):
        ax.text(0.02, 0.95, f"Trimmed: {trim_info['cut_pre']}+{trim_info['cut_post']} pts removed",
                transform=ax.transAxes, fontsize=9, color="red",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="red"))

    for b_int, label, color in [(1, "ZTF-g", BAND_COLORS["g"]), (2, "ZTF-r", BAND_COLORS["r"])]:
        mask = band == b_int
        if mask.sum() > 0:
            ax.errorbar(day[mask], flux[mask], yerr=arr[mask, 3],
                       fmt="o", ms=4, capsize=2, color=color, label=label,
                       alpha=0.8, markeredgewidth=0.5, markeredgecolor="white")

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"Days since MJD {mjd0:.0f}", fontsize=11)
    ax.set_ylabel("Flux (μJy)", fontsize=11)
    ax.set_title(f"{ztf_name} — ZTF Light Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plot_path = source_dir / "lightcurve.png"
    fig.savefig(plot_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    return plot_path


def download_sdss_cutout(ra, dec, source_dir):
    url = (f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
           f"?ra={ra}&dec={dec}&scale=0.4&width=300&height=300")
    cutout_path = source_dir / "cutout.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                cutout_path.write_bytes(resp.read())
                return cutout_path
    except Exception as e:
        print(f"  [WARN] Cutout failed: {e}")
    return None


def process_one(ztf_name, ra=None, dec=None, category="TDE"):
    source_dir = SOURCES_DIR / f"{ztf_name}_PRF"
    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {ztf_name} (cat={category}) ===")

    arr_raw = load_ztf_npy(ztf_name, category=category)
    print(f"  Data: {arr_raw.shape[0]} pts, g={sum(arr_raw[:,1]==1)}, r={sum(arr_raw[:,1]==2)}")

    bl = detect_baseline(arr_raw)
    pre_s = "YES" if bl["pre_detected"] else "NO"
    post_s = "YES" if bl["post_detected"] else "NO"
    print(f"  Baseline: pre={pre_s} post={post_s}")

    # Trim to burst period
    arr_trim, trim_info = trim_to_burst(arr_raw, bl)
    if trim_info.get("applied"):
        print(f"  Trim: cut_pre={trim_info['cut_pre']} cut_post={trim_info['cut_post']} "
              f"→ {trim_info['n_kept']}/{trim_info['n_original']} pts kept")
    elif trim_info.get("reason"):
        print(f"  Trim: ABORTED — {trim_info['reason'][:100]}")

    f = compute_features(arr_trim)

    print(f"  rise={f['rise_t']:.0f}d  decline={f['drate']:+.2f} uJy/d  "
          f"g-r pairs={f['gr_valid']}")

    md_path = generate_analysis_md(arr_trim, f, bl, ztf_name, source_dir, trim_info, label=category)
    print(f"  ✓ analysis.md ({md_path.stat().st_size} B)")

    lc_path = generate_lightcurve_plot(arr_trim, bl, ztf_name, source_dir, trim_info)
    print(f"  ✓ lightcurve.png ({lc_path.stat().st_size} B)")

    if ra and dec:
        cutout_path = download_sdss_cutout(ra, dec, source_dir)
        if cutout_path:
            print(f"  ✓ cutout.png ({cutout_path.stat().st_size} B)")
        else:
            print(f"  ⊘ cutout failed")
    else:
        print(f"  ⊘ no coords")

    return source_dir


def main():
    parser = argparse.ArgumentParser(description="ztf_adapter.py v3 — ZTF → PRF (promt.py compatible)")
    parser.add_argument("ztf_name", nargs="?")
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--batch")
    parser.add_argument("--category", default="TDE", choices=["TDE", "SN", "AGN", "Others", "unlabeled"])
    args = parser.parse_args()
    if args.ztf_name:
        process_one(args.ztf_name, ra=args.ra, dec=args.dec, category=args.category)
    elif args.batch:
        with open(args.batch) as f:
            for name in [l.strip() for l in f if l.strip() and not l.startswith("#")]:
                process_one(name, category=args.category)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()