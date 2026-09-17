#!/usr/bin/env python3
"""
fetch_sn_lasair_parallel.py — 并行抓取 Lasair filter 页 SN 的坐标+红移 → CSV

与 fetch_sn_lasair.py 相同逻辑，但对象页抓取用 ThreadPoolExecutor 并行（默认 16 并发）。
7.7s/对象 × 1000 对象，16 并发 ≈ 8 分钟。

用法:
  python fetch_sn_lasair_parallel.py --per-filter 500 --workers 16
"""
import re, csv, time, subprocess, argparse, concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FILTERS = {1004: "SN Ia", 1006: "SN II"}
HEADERS = ["Disc. Internal Name", "Name", "RA", "DEC", "Redshift", "Class"]


def fetch(url, retries=3, max_time=30):
    for attempt in range(retries):
        try:
            r = subprocess.run(["curl", "-s", "--max-time", str(max_time), url],
                               capture_output=True, text=True, timeout=max_time + 5)
            if r.returncode == 0 and r.stdout and len(r.stdout) > 10000:
                return r.stdout
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return ""


def parse_filter_objects(html):
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    out, seen = [], set()
    for r in rows:
        tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.DOTALL)
        clean = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip() for t in tds]
        clean = [c for c in clean if c]
        if len(clean) >= 8 and clean[1].startswith("ZTF"):
            if clean[1] not in seen:
                seen.add(clean[1])
                out.append((clean[1], clean[6], clean[7]))
    return out


def parse_object_page(html):
    """对象页 → (ra_deg, dec_deg, redshift, tns_name) 或 None"""
    coord = re.search(r"(\d{2,3}\.\d+)\s*,\s*([+-]?\d{1,2}\.\d+)", html)
    z = re.search(r"z\s*=\s*(\d+\.\d+)", html)
    if not coord:
        return None
    # TNS 名：从 "Transient Name Server" 区块后找第一个 SN20xx（对象自己的 TNS 名）
    tns = ""
    tns_pos = html.find("Transient Name Server")
    if tns_pos >= 0:
        m = re.search(r"SN(20\d\d[a-z]+)", html[tns_pos:tns_pos + 5000])
        if m:
            tns = m.group(1)
    return float(coord.group(1)), float(coord.group(2)), (z.group(1) if z else ""), tns


def fetch_one(ztf):
    html = fetch(f"https://lasair-ztf.lsst.ac.uk/objects/{ztf}/")
    meta = parse_object_page(html)
    return ztf, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-filter", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tns_search_SN_n.csv"))
    ap.add_argument("--batch-limit", type=int, default=0,
                    help="本次最多抓取多少个对象（0=不限），配合 --resume 小步慢跑")
    ap.add_argument("--resume", action="store_true",
                    help="续跑：读取已有 CSV，跳过已抓对象，数据累加")
    args = ap.parse_args()

    existing_rows, done = [], set()
    if args.resume and Path(args.out).exists():
        with open(args.out, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0] == "Disc. Internal Name":
                    continue
                existing_rows.append(row)
                done.add(row[0])
        print(f"续跑：已有 {len(existing_rows)} 行数据，跳过 {len(done)} 个已抓对象")

    targets = []   # (cls, ztf, tns)
    # 本地清单回退：filter 页被封时用之前保存的 1996 个对象清单
    local_list = Path("/tmp/lasair_sn_objects.json")
    for fid, cls in FILTERS.items():
        objs = []
        for attempt in range(5):
            print(f"[filter {fid}] 抓取 {cls} 清单 (尝试 {attempt+1}) ...")
            html = fetch(f"https://lasair-ztf.lsst.ac.uk/filters/{fid}/run/", max_time=30)
            objs = parse_filter_objects(html)
            if len(objs) > 0:
                break
            print(f"  ⚠ 解析 0 个，可能是限流，退避 {attempt+1}s 重试 ...")
            time.sleep(3 * (attempt + 1))
        if not objs and local_list.exists():
            print(f"  ⚠ filter 页仍不可用，回退本地清单 (filter {fid})")
            import json as _json
            objs = [(o["ztf"], o["tns"], o["class"])
                    for o in _json.load(open(local_list)) if o["filter"] == fid]
        objs = objs[: args.per_filter]
        print(f"  → {len(objs)} 个")
        targets += [(cls, o[0], o[1]) for o in objs]
        time.sleep(1)

    # 去重（resume 模式下跳过已抓对象）
    seen, uniq = set(), []
    for cls, ztf, tns in targets:
        if ztf in done or ztf in seen:
            continue
        seen.add(ztf)
        uniq.append((cls, ztf, tns))
    if args.batch_limit > 0:
        uniq = uniq[: args.batch_limit]
    print(f"\n去重后待抓 {len(uniq)} 个对象（已跳过 {len(done)} 个已抓），"
          f"{args.workers} 并发抓取对象页 ...")

    out_rows, ok, fail = [], 0, 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, ztf): (cls, ztf, tns) for cls, ztf, tns in uniq}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            cls, ztf, tns = futs[fut]
            ztf2, meta = fut.result()
            if meta is None:
                fail += 1
            else:
                ra, dec, z, tns_obj = meta
                # Name 列优先用对象页 TNS 名，退回 filter 页 tns_name
                name = tns_obj or tns
                out_rows.append([ztf, name, f"{ra:.6f}", f"{dec:.6f}", z, cls])
                ok += 1
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(uniq)}] ok={ok} fail={fail} "
                      f"({(i+1)/el:.1f} obj/s, ETA {(len(uniq)-i-1)/((i+1)/el):.0f}s)")

    # resume 模式：已有数据 + 新抓数据累加写回
    all_rows = existing_rows + out_rows
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(all_rows)

    el = time.time() - t0
    print(f"\n完成: 本次 ok={ok} fail={fail} 耗时 {el:.0f}s，"
          f"总计 {len(all_rows)} 行 → {args.out}")


if __name__ == "__main__":
    main()
