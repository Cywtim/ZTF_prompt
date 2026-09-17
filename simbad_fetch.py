#!/usr/bin/env python3
"""simbad_fetch.py — 从 SIMBAD 批量解析 SN 的坐标+红移（备选数据源，无防爬限制）"""
import re, csv, time, subprocess, argparse, concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def simbad_ascii(ident):
    url = f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={ident}&output.format=ASCII"
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "20", url],
                           capture_output=True, text=True, timeout=25)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""

def parse_simbad(html):
    """SIMBAD ASCII 输出 → (ra_deg, dec_deg, redshift, otype) 或 None"""
    if "Identifier not found" in html or not html.strip():
        return None
    # ICRS 坐标: "Coordinates(ICRS,ep=J2000,eq=2000): 10 13 48.230  +18 07 38.35"
    m = re.search(r"Coordinates\(ICRS[^)]*\):\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]\d+)\s+(\d+)\s+([\d.]+)", html)
    if not m:
        return None
    h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
    d_sign = -1 if m.group(4).startswith("-") else 1
    d, dm, ds = float(m.group(4).lstrip("+-")), float(m.group(5)), float(m.group(6))
    ra_deg = (h + mi/60 + s/3600) * 15
    dec_deg = d_sign * (d + dm/60 + ds/3600)
    # Redshift
    z = re.search(r"^Redshift:\s*([\d.]+)", html, re.MULTILINE)
    redshift = z.group(1) if z else ""
    # Object type
    t = re.search(r"Object\s+\S+\s+---\s+(\S+)", html)
    otype = t.group(1) if t else ""
    return ra_deg, dec_deg, redshift, otype

def fetch_one(ident):
    html = simbad_ascii(ident)
    return ident, parse_simbad(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/tmp/lasair_sn_objects.json")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tns_search_SN_n.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    import json
    objs = json.load(open(args.input))
    # 只处理有 tns 名的
    idents = [(o["ztf"], f"SN{o['tns']}" if o["tns"] else o["ztf"]) for o in objs]
    if args.max:
        idents = idents[: args.max]

    # 去重 ident
    seen, uniq = set(), []
    for ztf, ident in idents:
        if ident not in seen:
            seen.add(ident)
            uniq.append((ztf, ident))
    print(f"共 {len(uniq)} 个 ident，{args.workers} 并发查 SIMBAD ...")

    rows, ok, fail = [], 0, 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, ident): ztf for ztf, ident in uniq}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            ztf = futs[fut]
            ident, meta = fut.result()
            if meta is None:
                fail += 1
            else:
                ra, dec, z, otype = meta
                rows.append([ztf, ident.replace("SN", "", 1) if ident.startswith("SN") else ident,
                             f"{ra:.6f}", f"{dec:.6f}", z, otype])
                ok += 1
            if (i+1) % 100 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(uniq)}] ok={ok} fail={fail} ({(i+1)/el:.1f}/s)")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Disc. Internal Name", "Name", "RA", "DEC", "Redshift", "Class"])
        w.writerows(rows)

    print(f"\n完成: ok={ok} fail={fail} → {args.out}")

if __name__ == "__main__":
    main()
