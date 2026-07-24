#!/usr/bin/env python3
"""Extract RA/Dec for all sources and save as radec.txt in HMS/DMS format.
V2: Fixed TNS CSV parsing and ZTF18 naming."""

import json, os, re, csv, glob
from pathlib import Path

PROJECT = Path("/home/cyan/AppData/VScode/TDeck/ZTF_prompt")
ZTF_DATA = Path("/home/cyan/AppData/VScode/TDeck/ZTF_TDE/data")

INDEX_FILE = PROJECT / "index.json"
SOURCES_DIR = PROJECT / "sources"
ZTFSNIA_DIR = ZTF_DATA / "ZTFSNIa"
WYB_DIR = ZTF_DATA / "WYB_ZTF-basecorr"
TNS_CSV = ZTF_DATA / "TNS_TDE_search_results.csv"


def deg_to_hmsdms(ra_deg, dec_deg):
    """Convert decimal degrees to HMS/DMS string."""
    ra_h = ra_deg / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = (ra_h - h - m / 60) * 3600
    ra_str = f"{h:02d}:{m:02d}:{s:05.2f}"
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    dd = int(d)
    mm = int((d - dd) * 60)
    ss = (d - dd - mm / 60) * 3600
    dec_str = f"{sign}{dd:02d}:{mm:02d}:{ss:04.1f}"
    return f"{ra_str} {dec_str}"


def hmsdms_to_deg(ra_str, dec_str):
    """Parse HMS/DMS to decimal degrees."""
    # RA: "12:22:01.130"
    parts = ra_str.split(":")
    ra_deg = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
    # Dec: "+16:59:44.02"
    parts2 = dec_str.split(":")
    sign = -1 if parts2[0].startswith("-") else 1
    dec_deg = sign * (abs(float(parts2[0])) + float(parts2[1]) / 60 + float(parts2[2]) / 3600)
    return ra_deg, dec_deg


def build_tns_map():
    """Build mapping from IAU name suffix (e.g. '2022czy') → (ra_deg, dec_deg).
    
    TNS CSV columns (by index, because header has missing field):
      0=id, 1=prefix(TDE/AT/SN), 2=reps(yymmmm), 3=class, 4=extra, 5=ra, 6=decl
    """
    tns_map = {}
    if not TNS_CSV.exists():
        return tns_map
    with open(TNS_CSV) as f:
        f.readline()  # skip header
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 7:
                continue
            prefix = cols[1].strip()
            reps = cols[2].strip()
            ra_str = cols[5].strip()
            dec_str = cols[6].strip()
            if reps and ra_str and dec_str:
                try:
                    ra_deg, dec_deg = hmsdms_to_deg(ra_str, dec_str)
                    # Map by multiple keys
                    full_name = f"{prefix}{reps}"  # e.g. TDE2022czy
                    tns_map[full_name] = (ra_deg, dec_deg)
                    tns_map[reps] = (ra_deg, dec_deg)  # e.g. 2022czy
                except (ValueError, IndexError):
                    continue
    return tns_map


def build_wyb_map():
    """Build mapping from IAU name → (ra_deg, dec_deg) from WYB_ZTF-basecorr."""
    wyb_map = {}
    if not WYB_DIR.exists():
        return wyb_map
    for txt_file in WYB_DIR.glob("*.txt"):
        try:
            with open(txt_file) as f:
                ra_deg = dec_deg = None
                for line in f:
                    if line.startswith("# # ra:"):
                        ra_deg = float(line.split(":")[1].strip())
                    elif line.startswith("# # dec:"):
                        dec_deg = float(line.split(":")[1].strip())
                        if ra_deg is not None:
                            break
                if ra_deg is not None and dec_deg is not None:
                    iau_name = txt_file.stem  # e.g. "AT2022czy"
                    wyb_map[iau_name] = (ra_deg, dec_deg)
        except Exception:
            pass
    return wyb_map


def build_ztf_json_map():
    """Build mapping from ZTF name → (ra_deg, dec_deg) from ZTFSNIa JSONs."""
    ztf_map = {}
    if not ZTFSNIA_DIR.exists():
        return ztf_map
    for json_file in ZTFSNIA_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            cands = data.get("candidates", [])
            if cands and "ra" in cands[0] and "dec" in cands[0]:
                ztf_name = json_file.stem  # e.g. "ZTF19aadolpe_difference_photometry"
                ra = cands[0]["ra"]
                dec = cands[0]["dec"]
                ztf_map[ztf_name] = (ra, dec)
        except Exception:
            pass
    return ztf_map


def get_wfst_coords(source_id):
    """Get coordinates from WFST CSV filenames."""
    if source_id.startswith("WFST_J"):
        coord_part = source_id.replace("WFST_J", "")
        for csv_file in ZTF_DATA.rglob("*.csv"):
            if coord_part in csv_file.name:
                m = re.search(r'J(\d+\.?\d*)\+(\d+\.?\d*)', csv_file.stem)
                if m:
                    ra_s = m.group(1)
                    dec_s = m.group(2)
                    ra_deg = (float(ra_s[:2]) + float(ra_s[2:4])/60 + float(ra_s[4:])/3600) * 15
                    dec_deg = float(dec_s[:2]) + float(dec_s[2:4])/60 + float(dec_s[4:])/3600
                    return ra_deg, dec_deg
    return None


def main():
    with open(INDEX_FILE) as f:
        idx = json.load(f)

    tns_map = build_tns_map()
    wyb_map = build_wyb_map()
    ztf_map = build_ztf_json_map()
    print(f"TNS: {len(tns_map)} entries, WYB: {len(wyb_map)} entries, ZTF JSON: {len(ztf_map)} entries")

    done = 0
    missing = 0
    results = {}

    for source_id in sorted(idx.keys()):
        ra_deg, dec_deg = None, None

        # Strategy 1: ZTF JSON (for ZTF SN sources)
        if source_id in ztf_map:
            ra_deg, dec_deg = ztf_map[source_id]

        # Strategy 2: WYB (for AT/SN named sources)
        if ra_deg is None:
            # Try as-is, and with AT prefix
            base = source_id.replace("wmx_", "")
            for name in [base, base.replace("TDE_", "AT"), f"AT{base.replace('TDE_', '')}",
                         source_id.replace("wmx_", "AT")]:
                if name in wyb_map:
                    ra_deg, dec_deg = wyb_map[name]
                    break

        # Strategy 3: TNS CSV
        if ra_deg is None:
            base = source_id.replace("wmx_TDE_", "").replace("wmx_", "")
            for key in [base, base.replace("_", ""), f"TDE{base}"]:
                if key in tns_map:
                    ra_deg, dec_deg = tns_map[key]
                    break

        # Strategy 4: WFST CSV
        if ra_deg is None and source_id.startswith("WFST"):
            result = get_wfst_coords(source_id)
            if result:
                ra_deg, dec_deg = result

        # Write result
        src_dir = SOURCES_DIR / source_id
        src_dir.mkdir(parents=True, exist_ok=True)
        out_path = src_dir / "radec.txt"

        if ra_deg is not None:
            hmsdms = deg_to_hmsdms(ra_deg, dec_deg)
            with open(out_path, "w") as f:
                f.write(f"{hmsdms}\n")
            done += 1
            if done <= 5 or done % 50 == 0:
                print(f"  [{done}] {source_id} → {hmsdms}")
        else:
            with open(out_path, "w") as f:
                f.write("unknown\n")
            missing += 1
            if missing <= 10:
                print(f"  [missing] {source_id}")

    print(f"\nDone: {done} with coords, {missing} unknown (total {len(idx)})")


if __name__ == "__main__":
    main()