#!/usr/bin/env python3
"""Download SDSS cutout images for all sources with valid radec.txt."""

import os, re, time, urllib.request
from pathlib import Path

SOURCES_DIR = Path("/home/cyan/AppData/VScode/TDeck/ZTF_prompt/sources")
SDSS_URL = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"


def hmsdms_to_deg(radec_str):
    """Parse 'HH:MM:SS.S ±DD:MM:SS.S' → (ra_deg, dec_deg)."""
    parts = radec_str.strip().split()
    if len(parts) != 2:
        return None, None
    ra_s, dec_s = parts[0], parts[1]
    # RA
    h, m, s = ra_s.split(":")
    ra_deg = (float(h) + float(m)/60 + float(s)/3600) * 15
    # Dec
    sign = -1 if dec_s.startswith("-") else 1
    d_parts = dec_s.lstrip("+-").split(":")
    dec_deg = sign * (float(d_parts[0]) + float(d_parts[1])/60 + float(d_parts[2])/3600)
    return ra_deg, dec_deg


def download_cutout(ra_deg, dec_deg, out_path, scale=0.4, size=300):
    """Download SDSS cutout. Returns True on success."""
    url = f"{SDSS_URL}?ra={ra_deg}&dec={dec_deg}&scale={scale}&width={size}&height={size}&opt=G"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 500:
            return False  # too small, probably error page
        with open(out_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def main():
    total, ok, skip, fail = 0, 0, 0, 0
    for src_dir in sorted(SOURCES_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        radec_file = src_dir / "radec.txt"
        if not radec_file.exists():
            continue
        total += 1

        out_path = src_dir / "cutout.png"
        if out_path.exists():
            skip += 1
            continue

        content = radec_file.read_text().strip()
        if content == "unknown":
            fail += 1
            continue

        ra_deg, dec_deg = hmsdms_to_deg(content)
        if ra_deg is None:
            fail += 1
            continue

        success = download_cutout(ra_deg, dec_deg, out_path)
        if success:
            ok += 1
            if ok <= 3 or ok % 30 == 0:
                print(f"  [{ok}] {src_dir.name} → cutout.png")
        else:
            fail += 1
            print(f"  [fail] {src_dir.name}")
        time.sleep(0.3)  # be nice to SDSS

    print(f"\nDone: {ok} downloaded, {skip} skipped (exists), {fail} failed/missing, {total} total")


if __name__ == "__main__":
    main()