#!/usr/bin/env python3
"""Dry-run: 单个源 min-gap cadence 截断 → 生成 _PRF_cadence{x} 分析产物。
只写新目录，不碰原始 npy / sources/*_PRF/ / results/*_PRF.json。
不跑 LLM 分类（这一步先验证截断+analysis.md 生成）。
"""
import sys
sys.path.insert(0, "/home/cyan/AppData/VScode/TDeck/ZTF_prompt")

import numpy as np
from pathlib import Path
import ztf_adapter as Z


def min_gap_resample(mjd, gap_days):
    """保留首个点，之后只保留与前一个保留点间隔 >= gap_days 的点。"""
    keep = [0]
    last = mjd[0]
    for i in range(1, len(mjd)):
        if mjd[i] - last >= gap_days:
            keep.append(i)
            last = mjd[i]
    return keep


def run_one(ztf_name, category, cadence_days):
    suffix = f"_PRF_cadence{cadence_days}"
    source_dir = Z.SOURCES_DIR / f"{ztf_name}{suffix}"

    # 读原始 npy
    arr_raw = Z.load_ztf_npy(ztf_name, category=category)
    n_raw = arr_raw.shape[0]

    # 内存中 min-gap 截断
    mjd = arr_raw[:, 0]
    keep = min_gap_resample(mjd, cadence_days)
    arr_sub = arr_raw[keep]
    n_sub = arr_sub.shape[0]

    print(f"=== {ztf_name} (cat={category}) cadence={cadence_days}d ===")
    print(f"  原始 {n_raw} pts -> 截断 {n_sub} pts "
          f"(g={int((arr_sub[:,1]==1).sum())}, r={int((arr_sub[:,1]==2).sum())})")

    if n_sub < 5:
        print(f"  [SKIP] 截断后点数过少 ({n_sub})，跳过")
        return None

    # 复用 ztf_adapter 的完整下游：baseline -> trim -> features -> md -> plot
    bl = Z.detect_baseline(arr_sub)
    arr_trim, trim_info = Z.trim_to_burst(arr_sub, bl)
    f = Z.compute_features(arr_trim)

    print(f"  trim: applied={trim_info.get('applied')} "
          f"n_kept={trim_info.get('n_kept')}/{trim_info.get('n_original')}")
    print(f"  rise={f['rise_t']:.0f}d decline={f['drate']:+.2f} g-r pairs={f['gr_valid']}")

    # 写入新目录（suffix 带 cadence，不会覆盖原始 _PRF）
    source_dir.mkdir(parents=True, exist_ok=True)
    md_path = Z.generate_analysis_md(arr_trim, f, bl, ztf_name, source_dir,
                                     trim_info, label=category, suffix=suffix)
    lc_path = Z.generate_lightcurve_plot(arr_trim, bl, ztf_name, source_dir, trim_info)
    print(f"  已生成: {md_path}")
    print(f"  已生成: {lc_path}")
    return source_dir


if __name__ == "__main__":
    # 用 TDE 类点数最多的源做 dry-run
    run_one("ZTF21aanxhjv", category="TDE", cadence_days=5)
