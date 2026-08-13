#!/usr/bin/env python3
"""对已下载的 19 个 AGN 批量生成 analysis.md + 分类"""
import sys, os, subprocess, csv, json

PROJECT = "/home/cyan/AppData/VScode/TDeck/ZTF_prompt"
FLUX_DIR = f"{PROJECT}/data/downloaded_AGN/flux"
AGN_DIR = "/home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/AGN"

# 从 CSV 读取所有坐标
coords = {}
with open(f"{PROJECT}/tns_search_AGN_n.csv") as f:
    for row in csv.DictReader(f):
        ztf = row.get("Disc. Internal Name", "").strip()
        z = row.get("Redshift", "").strip()
        if not ztf.startswith("ZTF"):
            continue
        ra_str = row["RA"]; dec_str = row["DEC"]
        try:
            h, m, s = ra_str.split(":")
            ra_d = (float(h) + float(m)/60 + float(s)/3600) * 15
            sign = -1 if dec_str.startswith("-") else 1
            d = dec_str.lstrip("+-").split(":")
            dec_d = sign * (float(d[0]) + float(d[1])/60 + float(d[2])/3600)
        except:
            ra_d = dec_d = None
        coords[ztf] = {
            "at": row["Name"], "z": z,
            "ra": ra_d, "dec": dec_d
        }

# 所有已下载的 npy
npy_files = sorted([f for f in os.listdir(FLUX_DIR) if f.endswith(".npy")])
results = []

for npy in npy_files:
    ztf = npy.replace("_difference_photometry_flux.npy", "")
    info = coords.get(ztf, {})
    at = info.get("at", "?")
    z = info.get("z", "?")
    ra = info.get("ra")
    dec = info.get("dec")

    print(f"\n{'─'*50}")
    print(f"  {ztf}  ({at}, z={z})")
    print(f"{'─'*50}")

    # Step 1: copy npy
    subprocess.run(["cp", f"{FLUX_DIR}/{npy}", f"{AGN_DIR}/"], check=False)

    # Step 2: ztf_adapter
    ra_s = f"{ra:.4f}" if ra else None
    dec_s = f"{dec:.4f}" if dec else None
    cmd = ["python", "ztf_adapter.py", ztf, "--category", "AGN"]
    if ra_s and dec_s:
        cmd += ["--ra", ra_s, "--dec", dec_s]

    ret = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True, timeout=60)
    adapter_out = ret.stdout.strip()
    print(adapter_out[:200])

    # Step 3: classify
    sid = f"{ztf}_PRF"
    ret = subprocess.run(
        ["python", "classify.py", sid, "--mode", "multimodal", "--n-shot", "1", "--force"],
        cwd=PROJECT, capture_output=True, text=True, timeout=120
    )
    classify_out = ret.stdout.strip()
    if "error" in classify_out.lower() and "analysis.md not found" in classify_out:
        print(f"  ❌ no analysis.md")
        results.append({"ztf": ztf, "at": at, "z": z, "error": "no analysis.md"})
        continue

    # Step 4: get results JSON
    result_path = f"{PROJECT}/results/{sid}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            r = json.load(f)
        label = r.get("classification", {}).get("label", "?")
        conf = r.get("classification", {}).get("confidence", "?")
        score = r.get("classification", {}).get("score", 0)
        summary = r.get("reasoning", {}).get("summary", "")
        indicators = r.get("reasoning", {}).get("indicators", [])
        flags = r.get("quality", {}).get("flags", [])

        print(f"  → {label} ({conf}, score={score:.2f})")
        print(f"     {summary[:120]}")

        results.append({
            "ztf": ztf, "at": at, "z": z,
            "label": label, "conf": conf, "score": score,
            "summary": summary, "indicators": indicators, "flags": flags
        })
    else:
        results.append({"ztf": ztf, "at": at, "z": z, "error": "no result"})

# 汇总报告
print(f"\n\n{'='*70}")
print(f"                     AGN CLASSIFICATION REPORT")
print(f"{'='*70}")

correct = sum(1 for r in results if r.get("label") == "AGN")
wrong = sum(1 for r in results if r.get("label") not in ("AGN", None) and "error" not in r)
errors = sum(1 for r in results if "error" in r)

print(f"\nTotal: {len(results)}  |  AGN ✓: {correct}  |  ✗: {wrong}  |  Err: {errors}")
print(f"\n{'Name':<25s} {'AT':<15s} {'z':>6s}  {'Result':>6s}  {'Score':>5s}  {'Conf':>7s}")
print(f"{'─'*25} {'─'*15} {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}")

for r in results:
    if "error" in r:
        print(f"{r['ztf']:<25s} {r['at']:<15s} {r['z']:>6s}  {'ERROR':>6s}")
    else:
        mark = "✓" if r["label"] == "AGN" else "✗"
        print(f"{r['ztf']:<25s} {r['at']:<15s} {r['z']:>6s}  {r['label']:>6s} {mark}  {r['score']:.2f}  {r['conf']:>7s}")

# 标志物分析
print(f"\n{'─'*70}")
print("Key indicators per source:")
print(f"{'─'*70}")
for r in results:
    if "indicators" in r and r["indicators"]:
        print(f"\n  {r['ztf']} → {r.get('label','?')}")
        for ind in r["indicators"][:3]:
            direction = f" →{ind.get('direction','')}" if ind.get('direction') else ""
            print(f"    [{ind.get('weight',0):.2f}] {ind.get('name','?')}: {ind.get('value','?')}{direction}")