#!/usr/bin/env python3
"""批量 cadence 截断 + LLM 分类（多进程版）。

每个源在独立子进程里跑完整流程，避免 matplotlib 线程冲突。
并行度 = --workers（默认 6，对应 3 个 API key 的并发上限）。

用法:
  python scripts_cadence_batch.py --cadence 5 [--limit N] [--workers 6]
"""
import sys, csv, time, argparse, os
import multiprocessing as mp

sys.path.insert(0, "/home/cyan/AppData/VScode/TDeck/ZTF_prompt")

CSV_PATH = "/home/cyan/AppData/VScode/TDeck/ZTF_prompt/ztf_quality_selection.csv"


def min_gap_resample(mjd, gap_days):
    keep = [0]
    last = mjd[0]
    for i in range(1, len(mjd)):
        if mjd[i] - last >= gap_days:
            keep.append(i)
            last = mjd[i]
    return keep


def _worker(args):
    """子进程入口：处理单个源。返回 dict。"""
    row, cadence = args
    ztf_name = row["source_id"]
    label = row["label"]
    category = {"TDE": "TDE", "AGN": "AGN", "SN": "SN"}[label]
    suffix = f"_PRF_cadence{cadence}"
    out_id = f"{ztf_name}{suffix}"

    import numpy as np
    from pathlib import Path
    import ztf_adapter as Z

    result_path = Z.SOURCES_DIR.parent / "results" / f"{out_id}.json"
    if result_path.exists():
        return {"id": out_id, "status": "skip", "reason": "result exists"}

    try:
        arr_raw = Z.load_ztf_npy(ztf_name, category=category)
    except FileNotFoundError as e:
        return {"id": out_id, "status": "skip", "reason": f"no npy"}

    n_raw = arr_raw.shape[0]
    keep = min_gap_resample(arr_raw[:, 0], cadence)
    arr_sub = arr_raw[keep]
    n_sub = arr_sub.shape[0]

    if n_sub < 2:
        return {"id": out_id, "status": "skip", "reason": f"<2 pts ({n_sub})"}

    source_dir = Z.SOURCES_DIR / out_id
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        bl = Z.detect_baseline(arr_sub)
        arr_trim, trim_info = Z.trim_to_burst(arr_sub, bl)
        f = Z.compute_features(arr_trim)
        Z.generate_analysis_md(arr_trim, f, bl, ztf_name, source_dir,
                               trim_info, label=category, suffix=suffix)
        Z.generate_lightcurve_plot(arr_trim, bl, ztf_name, source_dir, trim_info)
    except Exception as e:
        return {"id": out_id, "status": "fail", "reason": f"md/plot: {e}"}

    try:
        from classify import classify_pipeline
        import config
        r = classify_pipeline(out_id, model=config.MODEL, cot=False)
        return {"id": out_id, "status": "ok", "true": label,
                "pred": r.get("label", "?"), "score": r.get("score"),
                "conf": r.get("confidence"), "n_raw": n_raw,
                "n_sub": n_sub, "n_trim": arr_trim.shape[0]}
    except Exception as e:
        return {"id": out_id, "status": "fail", "reason": f"classify: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", type=int, required=True, choices=[5, 10])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import csv as _csv
    rows = [r for r in _csv.DictReader(open(CSV_PATH))]
    if args.limit:
        rows = rows[:args.limit]

    print(f"cadence={args.cadence}d, 源数={len(rows)}, workers={args.workers}", flush=True)
    t0 = time.time()

    tasks = [(r, args.cadence) for r in rows]
    ok = skip = fail = 0
    results = []

    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.append(res)
            st = res.get("status")
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail += 1
            el = time.time() - t0
            # 完成一个就刷一条
            line = f"[{i}/{len(rows)}] {st} {res.get('id','')}"
            if "pred" in res:
                line += f" {res['true']}->{res['pred']} score={res['score']}"
            if "reason" in res:
                line += f" ({res['reason'][:60]})"
            line += f" | ok={ok} skip={skip} fail={fail} {el/60:.1f}min"
            print(line, flush=True)

    out_csv = f"/tmp/cadence{args.cadence}_summary.csv"
    import json as _json
    fields = ["id", "status", "true", "pred", "score", "conf",
              "n_raw", "n_sub", "n_trim", "reason"]
    with open(out_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for info in results:
            w.writerow({k: info.get(k, "") for k in fields})

    el = (time.time() - t0) / 60
    print(f"\nDONE: ok={ok} skip={skip} fail={fail} in {el:.1f}min", flush=True)
    print(f"汇总: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
