#!/usr/bin/env python3
"""
cutout.py - Download sky survey cutout images from radec.txt

Usage:
  python cutout.py ZTF19aadolpe_difference_photometry
  python cutout.py --all
  python cutout.py --all --survey SDSS --size 400
  python cutout.py ZTF19aadolpe --force

Output: sources/{source_id}/cutout.png
"""

import sys, os, re, time, argparse, urllib.request
from pathlib import Path

import config

# ── Survey URL templates ────────────────────────────────────
SURVEY_URLS = {
    "SDSS": (
        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        "?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}&opt=G"
    ),
    "DSS": (
        "https://archive.stsci.edu/cgi-bin/dss_search"
        "?v=poss2ukstu_red&r={ra}&d={dec}&e=J2000&h={size_arcmin}&w={size_arcmin}&f=gif"
    ),
}


def hmsdms_to_deg(radec_str):
    """Parse 'HH:MM:SS.S ±DD:MM:SS.S' → (ra_deg, dec_deg)."""
    parts = radec_str.strip().split()
    if len(parts) != 2:
        return None, None
    try:
        ra_s, dec_s = parts[0], parts[1]
        h, m, s = ra_s.split(":")
        ra_deg = (float(h) + float(m) / 60 + float(s) / 3600) * 15
        sign = -1 if dec_s.startswith("-") else 1
        d_parts = dec_s.lstrip("+-").split(":")
        dec_deg = sign * (float(d_parts[0]) + float(d_parts[1]) / 60 + float(d_parts[2]) / 3600)
        return ra_deg, dec_deg
    except (ValueError, IndexError):
        return None, None


def download_cutout(ra_deg, dec_deg, out_path, survey="SDSS", size=300, scale=0.4, timeout=60):
    """Download cutout from sky survey. Returns True on success."""
    url_tpl = SURVEY_URLS.get(survey)
    if url_tpl is None:
        print(f"  [error] Unknown survey: {survey}")
        return False

    size_arcmin = round(size * scale / 60.0, 2)
    url = url_tpl.format(ra=ra_deg, dec=dec_deg, scale=scale,
                         size=size, size_arcmin=size_arcmin)

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 500:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return False
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  [error] {e}")
                return False
    return False


def process_one(source_id, survey="SDSS", size=300, scale=0.4, force=False):
    """Download cutout for a single source."""
    src_dir = config.SOURCES_DIR / source_id
    if not src_dir.is_dir():
        print(f"  [error] {source_id} - no source directory")
        return False

    radec_file = src_dir / "radec.txt"
    if not radec_file.exists():
        print(f"  [error] {source_id} - no radec.txt (run extract_radec.py first)")
        return False

    out_path = src_dir / "cutout.png"
    if out_path.exists() and not force:
        print(f"  [skip] {source_id} - cutout.png exists (use --force to overwrite)")
        return None

    content = radec_file.read_text().strip()
    if content == "unknown":
        print(f"  [skip] {source_id} - radec unknown")
        return False

    ra_deg, dec_deg = hmsdms_to_deg(content)
    if ra_deg is None:
        print(f"  [error] {source_id} - cannot parse radec: {content}")
        return False

    success = download_cutout(ra_deg, dec_deg, out_path, survey=survey, size=size, scale=scale)
    if success:
        print(f"  [done] {source_id} → cutout.png ({survey} {size}px)")
    else:
        print(f"  [fail] {source_id} - download failed or outside survey footprint")
    return success


def process_all(survey="SDSS", size=300, scale=0.4, force=False, max_files=None):
    """Download cutouts for all sources with valid radec.txt."""
    if not config.INDEX_FILE.exists():
        print("No index.json found")
        return

    import json
    with open(config.INDEX_FILE) as f:
        idx = json.load(f)

    sources = sorted(idx.keys())
    if max_files:
        sources = sources[:max_files]

    ok, skipped, failed = 0, 0, 0
    for i, source_id in enumerate(sources):
        result = process_one(source_id, survey=survey, size=size, scale=scale, force=force)
        if result is None:
            skipped += 1
        elif result:
            ok += 1
        else:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(sources)} (ok={ok} skip={skipped} fail={failed})")

    print(f"\nDone: {ok} ok, {skipped} skipped, {failed} failed (total {len(sources)})")


def main():
    parser = argparse.ArgumentParser(
        description="cutout.py - download sky survey cutout images from radec.txt")
    parser.add_argument("source_id", nargs="?", help="Source ID (omit for --all)")
    parser.add_argument("--all", action="store_true", help="Process all sources in index.json")
    parser.add_argument("--survey", default="SDSS",
                        choices=list(SURVEY_URLS.keys()),
                        help="Sky survey (default: SDSS)")
    parser.add_argument("--size", type=int, default=300,
                        help="Image size in pixels (default: 300)")
    parser.add_argument("--scale", type=float, default=0.4,
                        help="Pixel scale in arcsec/pixel (default: 0.4)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cutout.png")
    parser.add_argument("--max", type=int,
                        help="Max sources in --all mode")
    args = parser.parse_args()

    if args.all:
        process_all(survey=args.survey, size=args.size, scale=args.scale,
                    force=args.force, max_files=args.max)
    elif args.source_id:
        process_one(args.source_id, survey=args.survey, size=args.size,
                    scale=args.scale, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()