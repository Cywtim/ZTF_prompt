#!/usr/bin/env python3
"""
download_sn.py — 从 Lasair 批量下载 ZTF SN 光变 + cutout + analysis.md（SN_PRF 后缀）

数据源：tns_search_SN_n.csv（由 fetch_sn_lasair.py 生成，含 Disc. Internal Name/RA/DEC/Redshift/Class）

流程：
1. 光变 → ZTF_TDE/data/TS/Flux/SN/{ZTF}_difference_photometry_flux.npy
2. cutout → sources/{ZTF}_SN_PRF/cutout.png
3. analysis.md + lightcurve.png → sources/{ZTF}_SN_PRF/

后缀：source 目录用 `_SN_PRF`（区别于 TDE 的 `_PRF`），category=SN。

用法:
  python download_sn.py --test 3
  python download_sn.py --max 1000
  python download_sn.py --batch          # batch 文件里的 ZTF 名清单
"""
import sys, os, csv, subprocess, time, argparse, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "tns_search_SN_n.csv"

# flux 目录（对齐 ztf_adapter.load_ztf_npy(category="SN")）
ZTF_DATA_DIR = Path(os.environ.get(
    "ZTF_DATA_DIR",
    str(Path.home() / "AppData" / "VScode" / "TDeck" / "ZTF_TDE" / "data"),
))
FLUX_SN_DIR = ZTF_DATA_DIR / "TS" / "Flux" / "SN"
SUFFIX = "_SN_PRF"

from download_agn import fetch_lasair_lightcurve, hms_to_deg, download_cutout


def download_one(ztf_name, ra_deg, dec_deg, z, cls, label="SN"):
    """单个对象：光变 + cutout + analysis.md。"""
    FLUX_SN_DIR.mkdir(parents=True, exist_ok=True)

    status = {"flux": False, "cutout": False, "analysis": False}

    # 1. 光变
    npy_path = FLUX_SN_DIR / f"{ztf_name}_difference_photometry_flux.npy"
    if npy_path.exists():
        arr = np.load(npy_path)
        status["flux"] = True
    else:
        time.sleep(1.0)
        arr = fetch_lasair_lightcurve(ztf_name)
        if arr is not None and len(arr) >= 5:
            np.save(npy_path, arr)
            status["flux"] = True
        else:
            return status, 0

    npts = len(arr)

    # 2 + 3. 用 ztf_adapter.process_one 生成 analysis.md + lightcurve.png + cutout
    if ra_deg and dec_deg:
        import ztf_adapter
        try:
            ztf_adapter.process_one(
                ztf_name, ra=ra_deg, dec=dec_deg,
                category="SN", suffix=SUFFIX,
            )
            status["analysis"] = True
            src_dir = ztf_adapter.SOURCES_DIR / f"{ztf_name}{SUFFIX}"
            if (src_dir / "cutout.png").exists():
                status["cutout"] = True
        except Exception as e:
            print(f"  [WARN] analysis 失败: {e}")

    return status, npts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", default="")     # 一行一个 ZTF 用户名的 batch
    args = ap.parse_args()

    targets = []
    if args.batch:
        with open(args.batch) as f:
            for name in [l.strip() for l in f if l.strip()]:
                targets.append({"ztf": name, "name": "", "ra": None, "dec": None,
                                "z": "", "cls": "SN"})
    else:
        if not CSV_PATH.exists():
            print(f"错误: CSV 不存在 {CSV_PATH}")
            sys.exit(1)
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                ztf = row.get("Disc. Internal Name", "").strip()
                ra = row.get("RA", "").strip()
                dec = row.get("DEC", "").strip()
                z = row.get("Redshift", "").strip()
                cls = row.get("Class", "").strip()
                if not ztf.startswith("ZTF"):
                    continue
                try:
                    ra_f, dec_f = float(ra), float(dec)
                except ValueError:
                    ra_f = dec_f = None
                targets.append({"ztf": ztf, "name": row.get("Name", ""),
                                "ra": ra_f, "dec": dec_f, "z": z, "cls": cls})

    if args.start:
        targets = targets[args.start:]
    if args.test:
        targets = targets[: args.test]
    elif args.max:
        targets = targets[: args.max]

    total = len(targets)
    print(f"待处理 {total} 个 SN，suffix={SUFFIX}，flux→{FLUX_SN_DIR}")

    ok_flux = ok_analysis = ok_cutout = fail = 0
    t0 = time.time()
    for i, t in enumerate(targets):
        ztf = t["ztf"]
        label = "SN"  # category 统一 SN
        st, npts = download_one(ztf, t["ra"], t["dec"], t["z"], t["cls"])
        if not st["flux"]:
            fail += 1
            print(f"[{i+1}/{total}] {ztf} ({t['cls']}): 无光变数据")
            continue
        ok_flux += 1
        if st["analysis"]:
            ok_analysis += 1
        if st["cutout"]:
            ok_cutout += 1
        print(f"[{i+1}/{total}] {ztf} ({t['cls']}, z={t['z']}): "
              f"flux={npts}pts analysis={'Y' if st['analysis'] else 'N'} cutout={'Y' if st['cutout'] else 'N'}")
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  --- 进度 {i+1}/{total} ({el:.0f}s, {(i+1)/el:.2f} obj/s) "
                  f"flux={ok_flux} analysis={ok_analysis} cutout={ok_cutout} fail={fail} ---")

    el = time.time() - t0
    print(f"\n完成: flux={ok_flux} analysis={ok_analysis} cutout={ok_cutout} fail={fail} "
          f"耗时 {el:.0f}s")


if __name__ == "__main__":
    main()
