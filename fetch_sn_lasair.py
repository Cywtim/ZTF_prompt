#!/usr/bin/env python3
"""
fetch_sn_lasair.py — 从 Lasair filter 页批量生成 ZTF SN CSV（含坐标+红移）

数据源：Lasair /filters/1004/run/ (SN Ia) + /filters/1006/run/ (SN II)
每个 filter 页含 ~998 个 ZTF 对象（objectId / tns_name / TNS_confirmed_classes）。
坐标(ra_deg,dec_deg)与红移 z 需从单个对象页 /objects/{ZTF}/ 的静态 HTML 提取。

输出 CSV 列对齐 download_agn.py：
  Disc. Internal Name, Name, RA(deg), DEC(deg), Redshift, Class

用法:
  python fetch_sn_lasair.py --per-filter 500   # 每个 filter 取 500（默认）
  python fetch_sn_lasair.py --per-filter 3     # 小测试
"""
import sys, re, csv, json, time, subprocess, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FILTERS = {1004: "SN Ia", 1006: "SN II"}

HEADERS = ["Disc. Internal Name", "Name", "RA", "DEC", "Redshift", "Class"]


def fetch(url):
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "25", url],
                           capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def parse_filter_objects(html):
    """filter 页表格 → [(ztf, tns, class), ...] 去重保序"""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    out, seen = [], set()
    for r in rows:
        tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.DOTALL)
        clean = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip() for t in tds]
        clean = [c for c in clean if c]
        if len(clean) >= 8 and clean[1].startswith("ZTF"):
            ztf, tns, cls = clean[1], clean[6], clean[7]
            if ztf not in seen:
                seen.add(ztf)
                out.append((ztf, tns, cls))
    return out


def parse_object_page(html):
    """对象页 → (ra_deg, dec_deg, redshift) 或 None"""
    coord = re.search(r"(\d{2,3}\.\d+)\s*,\s*([+-]?\d{1,2}\.\d+)", html)
    z = re.search(r"z\s*=\s*(\d+\.\d+)", html)
    if not coord:
        return None
    ra, dec = float(coord.group(1)), float(coord.group(2))
    redshift = z.group(1) if z else ""
    return ra, dec, redshift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-filter", type=int, default=500)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tns_search_SN_n.csv"))
    args = ap.parse_args()

    rows = []
    for fid, cls in FILTERS.items():
        print(f"[filter {fid}] 抓取对象清单 ({cls}) ...")
        html = fetch(f"https://lasair-ztf.lsst.ac.uk/filters/{fid}/run/")
        objs = parse_filter_objects(html)
        objs = objs[: args.per_filter]
        print(f"  → 取 {len(objs)} 个")
        rows += [(fid, cls, o) for o in objs]

    # 去重（跨 filter）
    seen, uniq = set(), []
    for fid, cls, o in rows:
        if o[0] not in seen:
            seen.add(o[0])
            uniq.append((fid, cls, o))
    print(f"\n去重后共 {len(uniq)} 个对象，开始抓取对象页元数据 ...")

    out_rows, ok, fail = [], 0, 0
    t0 = time.time()
    for i, (fid, cls, (ztf, tns, _cls)) in enumerate(uniq):
        html = fetch(f"https://lasair-ztf.lsst.ac.uk/objects/{ztf}/")
        meta = parse_object_page(html)
        if meta is None:
            fail += 1
            continue
        ra, dec, z = meta
        out_rows.append([ztf, tns, f"{ra:.6f}", f"{dec:.6f}", z, cls])
        ok += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  [{i+1}/{len(uniq)}] ok={ok} fail={fail} ({rate:.1f} obj/s, ETA {(len(uniq)-i-1)/rate:.0f}s)")
        time.sleep(0.15)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(out_rows)

    print(f"\n完成: ok={ok} fail={fail} → {args.out}")
    print(f"列: {HEADERS}")


if __name__ == "__main__":
    main()
