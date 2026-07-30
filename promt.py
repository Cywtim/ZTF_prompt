#!/usr/bin/env python3
"""
promt.py - Convert light curve data (npy/csv) to analysis.md

Usage:
  python promt.py data/WFST_J101658.csv --label unknown
  python promt.py --batch data/TS/Flux/TDE/ --label TDE
  python promt.py --stats
  python promt.py --list TDE
  python promt.py --relabel WFST_J101658 TDE
"""

import sys, os, csv, json, glob, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

import config
from MAD import mad_clip

BAND_STR = {1: "g", 2: "r", 3: "u"}
BAND_INT = {"WFST-g": 1, "WFST-r": 2, "WFST-u": 3}


def load_npy(path):
    arr = np.load(path)
    return arr[arr[:, 0].argsort()]


def load_csv(path):
    raw = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            band = row.get("band", "").strip()
            if band not in BAND_INT:
                continue
            raw[band].append((float(row["MJD"]), float(row["flux"]), float(row["fluxerr"])))
    data = {}
    for band, pts in raw.items():
        seen = set()
        unique = []
        for mjd, flux, fluxerr in sorted(pts):
            k = (round(mjd, 5), band)
            if k in seen:
                continue
            seen.add(k)
            unique.append((mjd, flux, fluxerr))
        data[band] = unique
    return data


def build_array(data):
    rows = []
    for band_str, pts in data.items():
        bnum = BAND_INT[band_str]
        for mjd, flux, fluxerr in pts:
            rows.append([mjd, bnum, flux, fluxerr])
    arr = np.array(rows, dtype=np.float32)
    return arr[arr[:, 0].argsort()]


def auto_convert_units(arr):
    med = np.median(np.abs(arr[:, 2]))
    if med < 1.0:
        arr[:, 2] *= 1000.0
        arr[:, 3] *= 1000.0
    return arr


def _compute_pair_color(n, band, flux, mjd, idx_a, idx_b, b_a, b_b):
    """Compute mag-based and flux-based color for band pair (b_a - b_b).
    Returns (color_mag, color_flux) arrays of length n.
    """
    color_mag = np.full(n, np.nan)
    color_flux = np.zeros(n)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return color_mag, color_flux
    for i in range(n):
        if band[i] == b_a:
            j = idx_b[np.argmin(np.abs(mjd[idx_b] - mjd[i]))]
            color_flux[i] = flux[i] - flux[j]
            if flux[i] > 0 and flux[j] > 0:
                color_mag[i] = -2.5 * np.log10(flux[i] / flux[j])
        elif band[i] == b_b:
            j = idx_a[np.argmin(np.abs(mjd[idx_a] - mjd[i]))]
            color_flux[i] = flux[j] - flux[i]
            if flux[j] > 0 and flux[i] > 0:
                color_mag[i] = -2.5 * np.log10(flux[j] / flux[i])
    return color_mag, color_flux


def compute_features(arr):
    arr = arr.copy()
    t0 = arr[:, 0].min()
    arr[:, 0] -= t0
    mjd, band, flux, fluxerr = arr.T
    n = len(arr)
    slope = np.zeros(n)
    for b in np.unique(band):
        idx = np.where(band == b)[0]
        if len(idx) < 2:
            continue
        sm, sf = mjd[idx], flux[idx]
        o = np.argsort(sm)
        sm, sf = sm[o], sf[o]
        dt = np.maximum(np.diff(sm), 1e-3)
        ss = np.diff(sf) / dt
        ss = np.concatenate([[ss[0]], ss])
        slope[idx[o]] = ss
    smean = np.abs(slope).mean() + 1e-6
    phase = np.tanh(-slope / smean)

    gi = np.where(band == 1)[0]
    ri = np.where(band == 2)[0]
    ui = np.where(band == 3)[0]

    # g-r
    gr_mag, gr_flux = _compute_pair_color(n, band, flux, mjd, gi, ri, 1, 2)
    # u-g
    ug_mag, ug_flux = _compute_pair_color(n, band, flux, mjd, ui, gi, 3, 1)
    # u-r
    ur_mag, ur_flux = _compute_pair_color(n, band, flux, mjd, ui, ri, 3, 2)

    return arr, {
        "phase": phase,
        "color_gr_mag": gr_mag,
        "color_gr_flux": gr_flux,
        "color_ug_mag": ug_mag,
        "color_ug_flux": ug_flux,
        "color_ur_mag": ur_mag,
        "color_ur_flux": ur_flux,
        "t0": t0,
    }


def phase_trend(p):
    m = np.mean(p)
    if m < -0.2:
        return "rise"
    elif m > 0.2:
        return "fall"
    return "plat"


def ctrend(c):
    if len(c) == 0:
        return "---"
    return "Red" if np.nanmean(c) > 0 else "Blue"


def clevel(n):
    if n >= 8:
        return "HIGH"
    elif n >= 4:
        return "MEDIUM"
    return "LOW"


def _mag_or_dash(val):
    """Format mag value or '--' if NaN."""
    if np.isnan(val):
        return "--"
    return f"{val:+.3f}"


def _color_subtable(label, band_pair, color_mag, color_flux, arr, span):
    """Generate a color sub-table for one band pair (e.g. g-r, u-g, u-r).
    Returns list of markdown lines, or empty list if no valid color data.
    """
    color_arr = color_mag
    ec = color_arr[arr[:, 0] <= span * 0.3]
    mc = color_arr[(arr[:, 0] > span * 0.3) & (arr[:, 0] < span * 0.7)]
    lc = color_arr[arr[:, 0] >= span * 0.7]
    ec_valid = ec[~np.isnan(ec)]
    mc_valid = mc[~np.isnan(mc)]
    lc_valid = lc[~np.isnan(lc)]

    if len(ec_valid) + len(mc_valid) + len(lc_valid) == 0:
        return []

    lines = []
    lines.append(f"#### {label}\n")
    lines.append(f"| Phase | N pairs | {band_pair} (mag) | sigma | {band_pair} (uJy) | Trend |")
    lines.append("|-------|:-------:|:---------:|:-----:|:---------:|:-----:|")

    for clabel, c_arr in [("Early (0-30 pct)", ec), ("Mid (30-70 pct)", mc), ("Late (70-100 pct)", lc)]:
        valid = c_arr[~np.isnan(c_arr)]
        if len(valid) == 0:
            lines.append(f"| {clabel} | 0 | -- | -- | -- | -- |")
        else:
            if clabel.startswith("Early"):
                farr = color_flux[arr[:, 0] <= span * 0.3]
            elif clabel.startswith("Mid"):
                farr = color_flux[(arr[:, 0] > span * 0.3) & (arr[:, 0] < span * 0.7)]
            else:
                farr = color_flux[arr[:, 0] >= span * 0.7]
            lines.append(
                f"| {clabel} | {len(valid)} | {np.nanmean(c_arr):+.3f} | "
                f"{np.nanstd(c_arr):.3f} | {np.mean(farr):+.1f} | {ctrend(valid)} |"
            )

    if len(ec_valid) and len(lc_valid):
        dc = np.nanmean(lc) - np.nanmean(ec)
        if abs(dc) < 0.05:
            evo = "Flat (no clear evolution)"
        elif np.nanmean(ec) > np.nanmean(lc):
            evo = "Red to Blue (TDE-like)"
        else:
            evo = "Blue to Red (SN-like)"
        lines.append(
            f"| **Evolution** | -- | delta = {dc:+.3f} mag | -- | -- | **{evo}** |"
        )
    lines.append("")
    return lines


def generate_md(source_id, arr, f, label="unknown"):
    n = len(arr)
    span = arr[:, 0].max()
    t0 = f["t0"]
    gn = int((arr[:, 1] == 1).sum())
    rn = int((arr[:, 1] == 2).sum())
    un = int((arr[:, 1] == 3).sum())
    pi = int(np.argmax(arr[:, 2]))
    pk_day = arr[pi, 0]
    pk_flux = arr[pi, 2]
    pk_band = BAND_STR[int(arr[pi, 1])]
    rise_t = pk_day
    post = arr[arr[:, 0] >= pk_day]
    drate = (post[-1, 2] - post[0, 2]) / (post[-1, 0] - post[0, 0] + 1e-6) if len(post) >= 2 else 0.0
    rise_rate = (arr[pi, 2] - arr[0, 2]) / max(rise_t, 1)

    # Count valid color pairs for each band combination
    gr_valid = (~np.isnan(f["color_gr_mag"])).sum()
    ug_valid = (~np.isnan(f["color_ug_mag"])).sum()
    ur_valid = (~np.isnan(f["color_ur_mag"])).sum()

    L = []
    L.append(f"# {source_id} -- Light Curve Analysis\n")
    L.append("## Section 1: Source Metadata\n")
    L.append("| Property | Value |")
    L.append("|----------|-------|")
    L.append(f"| Source ID | {source_id} |")
    L.append(f"| Label | {label} |")
    L.append(f"| MJD range | {t0:.1f} to {t0 + span:.1f} (span = {span:.0f} d) |")
    L.append(f"| Total points | {n} (g: {gn}, r: {rn}, u: {un}) |")
    L.append(f"| Peak flux | {pk_flux:.1f} uJy ({pk_band}-band, day {pk_day:.1f}) |")
    L.append(f"| Rise time | {rise_t:.0f} d |")
    L.append(f"| Decline rate | {drate:+.3f} uJy/d |")
    L.append(f"| Cadence | ~{span / max(n, 1):.1f} d mean |")
    L.append("")

    L.append("## Section 2: Derived Features\n")
    L.append("### 2.1 Global Morphology\n")
    L.append("| Indicator | Value | Hint |")
    L.append("|-----------|-------|------|")
    rise_hint = "fast" if rise_t < 30 else ("moderate" if rise_t < 60 else "slow")
    L.append(f"| Rise rate | +{rise_rate:.2f} uJy/d | {rise_hint} |")
    dec_hint = "power-law" if drate > -0.15 else "steep"
    L.append(f"| Decline | {drate:+.3f} uJy/d | {dec_hint} |")
    L.append("")

    L.append("### 2.2 Color Evolution (mag)\n")
    L.append("> TDE color zones: g−r ∈ (−0.6, +0.1), u−g ∈ (−0.5, +0.4), u−r ∈ (−0.9, +0.2)\n")
    L.append("> Range-first: if values stay within TDE zone → TDE-like regardless of delta direction.\n")
    L.append("> Evolution (secondary): Red→Blue (Δ<−0.15) = TDE; Blue→Red (Δ>+0.15) outside zone = SN.\n")
    L.append("")

    # g-r subtable
    gr_lines = _color_subtable("g − r", "g-r", f["color_gr_mag"], f["color_gr_flux"], arr, span)
    if gr_lines:
        L.extend(gr_lines)

    # u-g subtable
    ug_lines = _color_subtable("u − g", "u-g", f["color_ug_mag"], f["color_ug_flux"], arr, span)
    if ug_lines:
        L.extend(ug_lines)

    # u-r subtable
    ur_lines = _color_subtable("u − r", "u-r", f["color_ur_mag"], f["color_ur_flux"], arr, span)
    if ur_lines:
        L.extend(ur_lines)

    if not (gr_lines or ug_lines or ur_lines):
        L.append("*No color data available (single-band source).*\n")

    L.append("### 2.3 Per-Phase Summary\n")
    # Build header dynamically based on available colors
    color_cols = "g-r (mag)"
    if ug_valid > 0:
        color_cols += " | u-g (mag)"
    if ur_valid > 0:
        color_cols += " | u-r (mag)"
    L.append(f"| Cutoff | N pts | Bands (g/r/u) | Phase | {color_cols} | Trend |")
    L.append(f"|--------|:-----:|:-------------:|:-----:|{':---------:|' * max(1, (ug_valid > 0) + (ur_valid > 0) + 1)}:-----:|")
    for pct in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
        mask = arr[:, 0] <= span * pct
        n_mask = int(mask.sum())
        if n_mask < 2:
            continue
        p = f["phase"][mask]
        b = arr[mask, 1]
        tn = min(5, n_mask)
        pm = np.mean(p[-tn:])
        bg = int((b == 1).sum())
        br = int((b == 2).sum())
        bu = int((b == 3).sum())

        # g-r
        c_gr = f["color_gr_mag"][mask]
        cm_gr = np.nanmean(c_gr[-tn:]) if tn > 0 else np.nan
        cs_gr = np.nanstd(c_gr[-tn:]) if tn > 1 else 0
        gr_str = f"{_mag_or_dash(cm_gr)} +/- {cs_gr:.3f}"

        row = f"| {int(pct * 100)}% | {n_mask} | {bg}/{br}/{bu} | {pm:+.3f} | {gr_str}"

        if ug_valid > 0:
            c_ug = f["color_ug_mag"][mask]
            cm_ug = np.nanmean(c_ug[-tn:]) if tn > 0 else np.nan
            cs_ug = np.nanstd(c_ug[-tn:]) if tn > 1 else 0
            row += f" | {_mag_or_dash(cm_ug)} +/- {cs_ug:.3f}"

        if ur_valid > 0:
            c_ur = f["color_ur_mag"][mask]
            cm_ur = np.nanmean(c_ur[-tn:]) if tn > 0 else np.nan
            cs_ur = np.nanstd(c_ur[-tn:]) if tn > 1 else 0
            row += f" | {_mag_or_dash(cm_ur)} +/- {cs_ur:.3f}"

        row += f" | {phase_trend(p[-tn:])} |"
        L.append(row)
    L.append("")

    L.append("### 2.4 Data Quality Flags\n")
    L.append("| Feature | Confidence | Reason |")
    L.append("|---------|:----------:|--------|")
    early_n = int((arr[:, 0] <= span * 0.1).sum()) if span > 0 else 0
    L.append(f"| Rise phase | {clevel(early_n)} | {early_n} pts in 10% window |")
    gr_ec = f["color_gr_mag"][arr[:, 0] <= span * 0.3]
    gr_ec_n = (~np.isnan(gr_ec)).sum()
    gr_lc = f["color_gr_mag"][arr[:, 0] >= span * 0.7]
    gr_lc_n = (~np.isnan(gr_lc)).sum()
    color_detail = f"{gr_ec_n} g-r pairs" if gr_valid > 0 else ""
    if ug_valid and ur_valid:
        color_detail += f", {ug_valid} u-g, {ur_valid} u-r"
    elif ur_valid:
        color_detail += f", {ur_valid} u-r"
    L.append(f"| Color coverage | {clevel(max(gr_ec_n, gr_lc_n))} | {color_detail} |")
    L.append(f"| Decline | {clevel(len(post))} | {len(post)} post-peak pts |")
    L.append("")


    L.append("## Section 3: Raw Light Curve\n")
    # Build header dynamically
    raw_header = "| Num | Day | B | Flux | Err | Phase | g-r (mag) | g-r (uJy)"
    if ug_valid > 0:
        raw_header += " | u-g (mag) | u-g (uJy)"
    if ur_valid > 0:
        raw_header += " | u-r (mag) | u-r (uJy)"
    raw_header += " |"
    L.append(f"> Flux in uJy. Day = MJD - {t0:.1f}. Phase: -1=rising, +1=falling.\n")
    L.append(raw_header)
    sep = "|-----|-----|---|:----:|:---:|:-----:|:---------:|:---------:"
    if ug_valid > 0:
        sep += "|:---------:|:---------:"
    if ur_valid > 0:
        sep += "|:---------:|:---------:"
    sep += "|"
    L.append(sep)
    for i in range(n):
        bs = BAND_STR[int(arr[i, 1])]
        row = (f"| {i + 1} | {arr[i, 0]:.1f} | {bs} | {arr[i, 2]:.1f} | "
               f"{arr[i, 3]:.2f} | {f['phase'][i]:+.3f} | "
               f"{_mag_or_dash(f['color_gr_mag'][i])} | {f['color_gr_flux'][i]:+.1f}")
        if ug_valid > 0:
            row += f" | {_mag_or_dash(f['color_ug_mag'][i])} | {f['color_ug_flux'][i]:+.1f}"
        if ur_valid > 0:
            row += f" | {_mag_or_dash(f['color_ur_mag'][i])} | {f['color_ur_flux'][i]:+.1f}"
        row += " |"
        L.append(row)
    L.append("")

    L.append("## Section 4: Classification Protocol\n")
    L.append("### System Instruction")
    L.append("Classify this transient light curve as **TDE / SN / AGN / Others**.\n")
    L.append("### Knowledge Base")
    L.append("- **TDE:** fast rise (t^-5/3), red-to-blue color reversal, power-law decay, no late plateau")
    L.append("- **SN:** diverse rise times, Ni-56 decay tail. SLSNe: long plateaus. SN IIn: fast rise.")
    L.append("- **Key:** color reversal direction; plateau presence; decline power-law slope\n")
    L.append("### Evidence Weighting")
    L.append("1. Form hypothesis from Derived Features (Sections 2-3)")
    L.append("2. Verify against Raw Data (Section 4)")
    L.append("3. Agreement = HIGH confidence; conflict = explain and downgrade\n")
    L.append("### Output Format")
    L.append("Return ONLY valid JSON:")
    L.append("```")
    L.append('{"classification":{"label":"TDE","confidence":"medium","score":0.6},')
    L.append('"reasoning":{"summary":"...","feature_based":"...","raw_audit":"...",')
    L.append('"indicators":[{"name":"...","value":"...","weight":0.3,"direction":"TDE","note":"..."}]},')
    L.append('"quality":{"overall":"medium","flags":[]}}')
    L.append("```")

    return "\n".join(L)


def load_index():
    if config.INDEX_FILE.exists():
        with open(config.INDEX_FILE) as f:
            return json.load(f)
    return {}


def save_index(idx):
    with open(config.INDEX_FILE, "w") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)


def process_one(path, label="unknown", force=False, source_id=None):
    path = Path(path)
    if source_id is None:
        source_id = path.stem
        for sfx in ["_flux_uJy", "_flux", "_lc", "_difference_photometry_flux"]:
            if source_id.endswith(sfx):
                source_id = source_id[:-len(sfx)]
                break
    src_dir = config.SOURCES_DIR / source_id
    if (src_dir / "analysis.md").exists() and not force:
        print(f"  [skip] {source_id} - already exists")
        return source_id
    if path.suffix == ".npy":
        arr = load_npy(str(path))
    elif path.suffix == ".csv":
        data = load_csv(str(path))
        arr = build_array(data)
    else:
        print(f"  [error] {source_id} - unsupported format: {path.suffix}")
        return None
    arr = auto_convert_units(arr)
    arr = arr[np.isin(arr[:, 1], [1, 2, 3])]
    arr, _mask, _mad_stats = mad_clip(arr)
    if len(arr) < config.MIN_PTS:
        print(f"  [skip] {source_id} - only {len(arr)} pts after filtering")
        return None
    arr_rel, f = compute_features(arr)
    md = generate_md(source_id, arr_rel, f, label)
    src_dir.mkdir(parents=True, exist_ok=True)
    with open(src_dir / "analysis.md", "w") as fh:
        fh.write(md)
    idx = load_index()
    idx[source_id] = {
        "label": label,
        "n_points": len(arr),
        "bands": {"g": int((arr[:, 1] == 1).sum()), "r": int((arr[:, 1] == 2).sum()), "u": int((arr[:, 1] == 3).sum())},
        "span_days": round(float(arr_rel[:, 0].max()), 1),
    }
    save_index(idx)
    n_pts = len(arr)
    print(f"  [done] {source_id} -> sources/{source_id}/analysis.md ({n_pts} pts)")
    return source_id


def process_batch(dir_path, label, max_files=None):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"Error: directory not found: {dir_path}")
        return
    files = sorted(dir_path.glob("*.npy")) + sorted(dir_path.glob("*.csv"))
    files = [f for f in files if not f.stem.startswith("synth_flux_")]
    if not files:
        print(f"No npy/csv files found in {dir_path}")
        return
    if max_files:
        files = files[:max_files]
    print(f"Processing {len(files)} files from {dir_path} (label={label})")
    done = 0
    for f in files:
        sid = process_one(f, label=label)
        if sid:
            done += 1
    print(f"Done: {done}/{len(files)} generated")


def cmd_stats():
    idx = load_index()
    if not idx:
        print("index.json is empty.")
        return
    counts = defaultdict(int)
    for info in idx.values():
        counts[info["label"]] += 1
    print(f"index.json: {len(idx)} sources")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


def cmd_list(label_filter=None):
    idx = load_index()
    for sid, info in sorted(idx.items()):
        if label_filter and info["label"] != label_filter:
            continue
        print(f"  {sid}  [{info['label']}]  {info.get('n_points', '?')} pts")


def cmd_relabel(source_id, new_label):
    idx = load_index()
    if source_id not in idx:
        print(f"Error: {source_id} not in index.json")
        return
    old = idx[source_id]["label"]
    idx[source_id]["label"] = new_label
    save_index(idx)
    print(f"  {source_id}: {old} -> {new_label}")
    md_path = config.SOURCES_DIR / source_id / "analysis.md"
    if md_path.exists():
        content = md_path.read_text()
        content = content.replace(f"| Label | {old} |", f"| Label | {new_label} |")
        md_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="promt.py - convert light curves to analysis.md")
    parser.add_argument("path", nargs="?", help="npy/csv file path")
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--source-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch")
    parser.add_argument("--max", type=int, help="max files in batch mode")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--list")
    parser.add_argument("--relabel", nargs=2, metavar=("ID", "LABEL"))
    args = parser.parse_args()
    if args.stats:
        cmd_stats()
    elif args.list:
        cmd_list(args.list if args.list != "all" else None)
    elif args.relabel:
        cmd_relabel(args.relabel[0], args.relabel[1])
    elif args.batch:
        process_batch(args.batch, args.label, max_files=args.max)
    elif args.path:
        process_one(args.path, label=args.label, force=args.force, source_id=args.source_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()