#!/usr/bin/env python3
"""
download_agn.py — 从 Lasair 批量下载 TNS AGN 光变数据 + SDSS cutout

1. 读取 tns_search_AGN_n.csv 中的 ZTF AGN
2. 从 Lasair 抓取 forced photometry → data/downloaded_AGN/flux/*.npy
3. 下载 SDSS cutout → data/downloaded_AGN/sources/{name}_PRF/cutout.png

用法:
  python download_agn.py              # 下载全部 64 个
  python download_agn.py --test 1     # 先测试 1 个
  python download_agn.py --max 10     # 限 10 个
"""

import sys, os, csv, subprocess, time, argparse, re, math
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "tns_search_AGN_n.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "downloaded_AGN"
FLUX_DIR = OUTPUT_DIR / "flux"
SOURCES_DIR = OUTPUT_DIR / "sources"

BAND_MAP = {"g": "1", "r": "2"}


def fetch_lasair_lightcurve(ztf_name):
    """从 Lasair 抓取 forced photometry，返回 (mjd, band_int, flux_μJy, flux_err_μJy) 数组或 None.
    
    解析新版 Lasair 页面中的 exportLightcurve 表格。
    列: MJD, UTC, Filter, unforced mag, status, forced flux (μJy), images, alert packet
    只保留 +ve 且有 forced flux 值的行。
    """
    url = f"https://lasair-ztf.lsst.ac.uk/objects/{ztf_name}/"
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "25", url],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    html = result.stdout

    # 找 exportLightcurve table
    idx = html.find("exportLightcurve")
    if idx < 0:
        return None

    # 找包含它的 <tbody>
    tbody_end = html.find("</tbody>", idx)
    tbody_start = html.rfind("<tbody>", 0, tbody_end)
    if tbody_start < 0 or tbody_end <= tbody_start:
        return None

    tbody = html[tbody_start:tbody_end + 8]

    # 逐行解析
    rows_html = re.findall(r"<tr>\s*(.*?)\s*</tr>", tbody, re.DOTALL)

    data = []
    for row_html in rows_html:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", t).strip() for t in tds]

        if len(clean) < 6:
            continue
        mjd_str = clean[0].strip()
        if not re.match(r"^\d{5,}", mjd_str):
            continue

        band_str = clean[2].strip()
        if band_str not in ("g", "r"):
            continue

        status = clean[4].strip()
        flux_str = clean[5].strip() if len(clean) > 5 else ""

        # 只保留 +ve 且有 forced flux
        if status != "+ve":
            continue
        if not flux_str:
            continue

        # 解析 "36.1 ± 3.6" 格式
        flux_parts = flux_str.replace("&plusmn;", "±").split("±")
        try:
            flux_ujy = float(flux_parts[0].strip())
            flux_err = float(flux_parts[1].strip()) if len(flux_parts) > 1 else 0.0
        except ValueError:
            continue

        if flux_ujy == 0:
            continue

        mjd = float(mjd_str)
        band_int = 1 if band_str == "g" else 2
        data.append((mjd, band_int, flux_ujy, flux_err))

    if not data:
        return None

    arr = np.array(data, dtype=np.float64)
    return arr


def hms_to_deg(ra_str, dec_str):
    """HH:MM:SS.S +DD:MM:SS.S → (ra_deg, dec_deg)"""
    try:
        h, m, s = ra_str.split(":")
        ra_deg = (float(h) + float(m) / 60 + float(s) / 3600) * 15
        sign = -1 if dec_str.startswith("-") else 1
        d_parts = dec_str.lstrip("+-").split(":")
        dec_deg = sign * (float(d_parts[0]) + float(d_parts[1]) / 60 + float(d_parts[2]) / 3600)
        return ra_deg, dec_deg
    except (ValueError, IndexError):
        return None, None


def download_cutout(ra_deg, dec_deg, out_path):
    """下载 SDSS cutout，返回 True/False"""
    url = (
        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra_deg}&dec={dec_deg}&scale=0.4&width=300&height=300"
    )
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", str(out_path), "--max-time", "30", url],
                timeout=35,
            )
            if result.returncode == 0 and out_path.stat().st_size > 500:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description="下载 TNS AGN 光变 + cutout")
    parser.add_argument("--test", type=int, default=0, help="只测试 N 个")
    parser.add_argument("--max", type=int, default=0, help="最多下载 N 个")
    parser.add_argument("--start", type=int, default=0, help="从第 N 个开始")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"错误: CSV 不存在 {CSV_PATH}")
        sys.exit(1)

    targets = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            ztf_name = row.get("Disc. Internal Name", "").strip()
            z = row.get("Redshift", "").strip()
            if not ztf_name.startswith("ZTF") or not z:
                continue
            ra, dec = row.get("RA", "").strip(), row.get("DEC", "").strip()
            targets.append({
                "ztf_name": ztf_name,
                "at_name": row["Name"],
                "ra": ra,
                "dec": dec,
                "redshift": z,
            })

    if args.start:
        targets = targets[args.start:]
    if args.test:
        targets = targets[: args.test]
    elif args.max:
        targets = targets[: args.max]

    total = len(targets)
    print(f"待处理: {total} 个 ZTF AGN")
    print(f"Flux 输出: {FLUX_DIR}")
    print(f"Cutout 输出: {SOURCES_DIR}\n")

    FLUX_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    ok_flux = no_data = ok_cutout = errors = 0

    for i, t in enumerate(targets):
        ztf_name = t["ztf_name"]
        at_name = t["at_name"]
        print(f"[{i + 1}/{total}] {ztf_name}  ({at_name}, z={t['redshift']})")

        # Step 1: 光变数据
        npy_path = FLUX_DIR / f"{ztf_name}_difference_photometry_flux.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            n_g = int(sum(arr[:, 1] == 1))
            n_r = int(sum(arr[:, 1] == 2))
            print(f"  [skip] flux — 已存在 ({len(arr)} pts, g={n_g}, r={n_r})")
            ok_flux += 1
        else:
            time.sleep(1.5)
            arr = fetch_lasair_lightcurve(ztf_name)
            if arr is None or len(arr) < 5:
                print(f"  [fail] flux — 无足够数据")
                no_data += 1
                continue
            np.save(npy_path, arr)
            n_g = int(sum(arr[:, 1] == 1))
            n_r = int(sum(arr[:, 1] == 2))
            ok_flux += 1
            print(f"  [done] flux — {len(arr)} pts (g={n_g}, r={n_r})")

        # Step 2: cutout
        source_id = f"{ztf_name}_PRF"
        src_dir = SOURCES_DIR / source_id
        src_dir.mkdir(parents=True, exist_ok=True)
        cutout_path = src_dir / "cutout.png"

        if cutout_path.exists():
            print(f"  [skip] cutout — 已存在")
            ok_cutout += 1
        elif not t["ra"] or not t["dec"]:
            print(f"  [skip] cutout — 无坐标")
        else:
            ra_deg, dec_deg = hms_to_deg(t["ra"], t["dec"])
            if ra_deg is None:
                print(f"  [fail] cutout — 坐标解析失败")
            elif download_cutout(ra_deg, dec_deg, cutout_path):
                ok_cutout += 1
                print(f"  [done] cutout — {cutout_path.stat().st_size} B")
            else:
                print(f"  [fail] cutout — SDSS 不可达或不在 footprint")

        if (i + 1) % 5 == 0:
            print(f"  --- 进度: flux={ok_flux} nodata={no_data} cutout={ok_cutout} err={errors} ---\n")

    print(f"\n{'=' * 50}")
    print(f"完成: flux_ok={ok_flux}  no_data={no_data}  cutout={ok_cutout}  errors={errors}")
    print(f"数据位置: {OUTPUT_DIR}")
    print(f"\n下一步 — 生成分析:")
    print(f"  1. 复制 npy 到 ZTF Flux 目录:")
    print(f"     cp {FLUX_DIR}/*.npy /home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/AGN/")
    print(f"  2. 运行 ztf_adapter.py 生成 PRF 分析:")
    print(f"     for f in {FLUX_DIR}/*.npy; do")
    print(f"       name=$(basename $f _difference_photometry_flux.npy)")
    print(f"       python ztf_adapter.py $name --category AGN")
    print(f"     done")
    print(f"  3. 或者让我直接帮你跑一条测试")


if __name__ == "__main__":
    main()