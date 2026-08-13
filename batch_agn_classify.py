#!/usr/bin/env python3
"""批量：download → ztf_adapter → classify，每个源跑完输出结果"""
import sys, os, subprocess, time, csv

PROJECT = "/home/cyan/AppData/VScode/TDeck/ZTF_prompt"
FLUX_DIR = f"{PROJECT}/data/downloaded_AGN/flux"
AGN_DIR = "/home/cyan/AppData/VScode/TDeck/ZTF_TDE/data/TS/Flux/AGN"

# 读取目标列表
targets = []
with open(f"{PROJECT}/tns_search_AGN_n.csv") as f:
    for row in csv.DictReader(f):
        ztf = row.get("Disc. Internal Name", "").strip()
        z = row.get("Redshift", "").strip()
        if not ztf.startswith("ZTF") or not z:
            continue
        ra = row["RA"]; dec = row["DEC"]
        h, m, s = ra.split(":")
        ra_d = (float(h) + float(m)/60 + float(s)/3600) * 15
        sign = -1 if dec.startswith("-") else 1
        d = dec.lstrip("+-").split(":")
        dec_d = sign * (float(d[0]) + float(d[1])/60 + float(d[2])/3600)
        targets.append({
            "ztf": ztf, "at": row["Name"], "z": z, "ra": ra_d, "dec": dec_d
        })

# 跳过已处理
done_flux = {f.replace("_difference_photometry_flux.npy", "") for f in os.listdir(FLUX_DIR) if f.endswith(".npy")} if os.path.isdir(FLUX_DIR) else set()
pending = [t for t in targets if t["ztf"] not in done_flux]

print(f"Total: {len(targets)}, already downloaded: {len(done_flux)}, to download: {len(pending)}\n")

# 逐个处理（直到收集到 10 个成功的）
success = 0
target = 10

for i, t in enumerate(pending):
    if success >= target:
        break

    ztf = t["ztf"]
    at = t["at"]
    print(f"{'='*60}")
    print(f"  [{success+1}/{target}] {ztf}  ({at}, z={t['z']})")
    print(f"{'='*60}")

    # Step 1: Download
    npy_file = f"{FLUX_DIR}/{ztf}_difference_photometry_flux.npy"
    if not os.path.exists(npy_file):
        time.sleep(1.5)
        ret = subprocess.run(
            ["python", "download_agn.py", "--start", str(i), "--test", "1"],
            cwd=PROJECT, capture_output=True, text=True, timeout=60
        )
        if "[done] flux" not in ret.stdout:
            print(f"  ❌ SKIP — no forced flux data\n")
            continue

    # Step 2: Copy to AGN dir
    subprocess.run(["cp", npy_file, f"{AGN_DIR}/"], check=False)

    # Step 3: ztf_adapter
    ra_str = f"{t['ra']:.4f}"
    dec_str = f"{t['dec']:.4f}"
    ret = subprocess.run(
        ["python", "ztf_adapter.py", ztf, "--ra", ra_str, "--dec", dec_str, "--category", "AGN"],
        cwd=PROJECT, capture_output=True, text=True, timeout=60
    )
    print(ret.stdout.strip())
    if ret.stderr.strip():
        print(f"  stderr: {ret.stderr.strip()[:200]}")

    # Step 4: classify
    sid = f"{ztf}_PRF"
    ret = subprocess.run(
        ["python", "classify.py", sid, "--mode", "multimodal", "--n-shot", "1"],
        cwd=PROJECT, capture_output=True, text=True, timeout=120
    )
    print(ret.stdout.strip())
    if "error" in ret.stdout.lower():
        print(f"  ❌ CLASSIFY FAILED\n")
        continue

    # Step 5: results
    ret = subprocess.run(
        ["python", "classify.py", sid, "--results"],
        cwd=PROJECT, capture_output=True, text=True, timeout=30
    )
    print(ret.stdout.strip())

    success += 1
    print()

print(f"\n{'='*60}")
print(f"Done: {success}/{target} sources classified")
print(f"{'='*60}")