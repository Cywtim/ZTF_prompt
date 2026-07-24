#!/usr/bin/env python3
"""
plot.py - Draw light curve PNG for multimodal classification

Usage:
  python plot.py data/WFST_J101658.csv
  python plot.py --batch data/TS/Flux/TDE/
  python plot.py data/xxx.npy --source-id WFST_J101658 --force

Output: sources/{source_id}/lightcurve.png
  u-band → blue,  g-band → green,  r-band → red
"""

import sys, os, argparse
from pathlib import Path
import numpy as np

# Reuse data loading from promt.py
from promt import (
    load_npy, load_csv, build_array, auto_convert_units, BAND_STR, BAND_INT,
)

import config

# ── Band → color ──────────────────────────────────────────
# BAND_STR = {1: "g", 2: "r", 3: "u"}
BAND_COLOR = {
    1: "#2ecc71",   # g → green
    2: "#e74c3c",   # r → red
    3: "#3498db",   # u → blue
}
BAND_MARKER = {1: "o", 2: "s", 3: "^"}
BAND_ZH = {1: "g 波段", 2: "r 波段", 3: "u 波段"}

# ── Plotting (lazy import to avoid heavy startup) ─────────

def draw_lightcurve(source_id, arr, out_path):
    """Draw light curve: u=blue, g=green, r=red.  Saves to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    t0 = arr[:, 0].min()
    day = arr[:, 0] - t0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#fafafa")

    for bnum in [1, 2, 3]:         # g, r, u
        mask = arr[:, 1] == bnum
        if not mask.any():
            continue
        d = day[mask]
        fl = arr[mask, 2]
        fe = arr[mask, 3]
        color = BAND_COLOR[bnum]
        label = f"{BAND_STR[bnum]} (n={mask.sum()})"

        ax.errorbar(d, fl, yerr=fe,
                    fmt=BAND_MARKER[bnum],
                    color=color, mec="white", mew=0.4,
                    ecolor=color, elinewidth=0.6, capsize=2, capthick=0.6,
                    markersize=5, alpha=0.85, label=label)

    # ── Annotations ──
    pk_idx = np.argmax(arr[:, 2])
    pk_day = day[pk_idx]
    pk_flux = arr[pk_idx, 2]
    pk_band_str = BAND_STR[int(arr[pk_idx, 1])]
    pk_color = BAND_COLOR[int(arr[pk_idx, 1])]

    ax.axvline(pk_day, color=pk_color, ls="--", lw=0.8, alpha=0.5)
    ax.annotate(
        f"peak: {pk_flux:.0f} μJy ({pk_band_str})",
        xy=(pk_day, pk_flux),
        xytext=(pk_day + 3, pk_flux * 1.05),
        fontsize=8, color=pk_color,
        arrowprops=dict(arrowstyle="->", color=pk_color, lw=0.8),
    )

    # ── Labels ──
    n_total = len(arr)
    span = day.max()
    ax.set_title(f"{source_id}  (N={n_total}, span={span:.0f} d)", fontsize=11, weight="bold")
    ax.set_xlabel(f"Day (MJD − {t0:.1f})", fontsize=10)
    ax.set_ylabel("Flux (μJy)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8,
              ncol=3, columnspacing=0.8).set_zorder(10)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.tick_params(labelsize=8)

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path

# ── Source ID extraction (same logic as promt.py) ──────────

def _strip_suffix(stem):
    for sfx in ["_flux_uJy", "_flux", "_lc", "_difference_photometry_flux"]:
        if stem.endswith(sfx):
            return stem[:-len(sfx)]
    return stem

# ── Single-file processing ─────────────────────────────────

def plot_one(path, force=False, source_id=None):
    path = Path(path)
    if source_id is None:
        source_id = _strip_suffix(path.stem)

    src_dir = config.SOURCES_DIR / source_id
    out_path = src_dir / "lightcurve.png"

    src_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  [skip] {source_id} - lightcurve.png exists (use --force to overwrite)")
        return source_id

    # Load
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
    if len(arr) < config.MIN_PTS:
        print(f"  [skip] {source_id} - only {len(arr)} pts after filtering")
        return None

    draw_lightcurve(source_id, arr, out_path)
    print(f"  [done] {source_id} -> sources/{source_id}/lightcurve.png ({len(arr)} pts)")
    return source_id

# ── Batch ──────────────────────────────────────────────────

def plot_batch(dir_path, max_files=None, force=False):
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

    print(f"Plotting {len(files)} files from {dir_path}")
    done = 0
    for f in files:
        sid = plot_one(f, force=force)
        if sid:
            done += 1
    print(f"Done: {done}/{len(files)} plotted")

# ── All existing sources ──────────────────────────────────

def plot_all(force=False):
    """Regenerate lightcurve.png for all sources in index.json."""
    from promt import load_index
    idx = load_index()
    if not idx:
        print("index.json is empty.")
        return

    # Try to find original data file for each source
    # Heuristic: look in ZTF Flux dirs + WFST dirs
    search_dirs = [
        config.ZTF_FLUX_DIR / "TDE",
        config.ZTF_FLUX_DIR / "SN",
        config.ZTF_DATA_DIR / "TS" / "Flux" / "TDE",
        config.ZTF_DATA_DIR / "TS" / "Flux" / "SN",
    ]

    # Build a lookup: source_id → file path
    file_map = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npy")) + sorted(d.glob("*.csv")):
            sid = _strip_suffix(f.stem)
            if sid not in file_map:  # first match wins
                file_map[sid] = f

    done = 0
    for sid in sorted(idx.keys()):
        if sid in file_map:
            result = plot_one(file_map[sid], force=force, source_id=sid)
            if result:
                done += 1
        else:
            print(f"  [warn] {sid} - no source data file found")

    print(f"\nDone: {done}/{len(idx)} plotted")

# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="plot.py - draw light curve PNG for multimodal classification"
    )
    parser.add_argument("path", nargs="?", help="npy/csv file path")
    parser.add_argument("--source-id", help="override source ID")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing lightcurve.png")
    parser.add_argument("--batch", help="directory of npy/csv files")
    parser.add_argument("--max", type=int, help="max files in batch")
    parser.add_argument("--all", action="store_true",
                        help="regenerate for all sources in index.json")
    args = parser.parse_args()

    if args.all:
        plot_all(force=args.force)
    elif args.batch:
        plot_batch(args.batch, max_files=args.max, force=args.force)
    elif args.path:
        plot_one(args.path, force=args.force, source_id=args.source_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()