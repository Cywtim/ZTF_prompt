#!/usr/bin/env python3
"""
enrich_analysis.py — 往 analysis.md 的 Section 1 补入 Gaia Sep / WISE 字段。

可作为脚本独立运行（扫描 sources/ 目录），也可作为模块被 app_rlhl.py 导入调用。
"""

import os, sys, sqlite3, argparse


# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_DIR = os.path.join(SCRIPT_DIR, "sources")
REGISTRY_PATH = os.path.join(SCRIPT_DIR, "..", "TDEweb", "instance", "wfst_registry.db")


# ═══════════════════════════════════════════════════
# 核心函数：可被外部导入
# ═══════════════════════════════════════════════════

def enrich_analysis_md(md_path, gaia_sep=None, w1mag=None, w2mag=None, w1_w2=None):
    """往 analysis.md 的 Section 1 补入 Gaia Sep 和 WISE 字段。

    Args:
        md_path: analysis.md 的完整路径
        gaia_sep: Gaia 星间距 (arcsec)，None 表示无数据
        w1mag, w2mag, w1_w2: WISE 红外星等和颜色，任一为 None 则跳过 WISE 行
    Returns:
        True 如果写入了内容，False 如果无数据可写或文件不存在
    """
    if not os.path.isfile(md_path):
        return False

    has_gaia = gaia_sep is not None
    has_wise = any(v is not None for v in [w1mag, w2mag, w1_w2])

    if not has_gaia and not has_wise:
        return False

    with open(md_path, "r") as f:
        lines = f.read().split("\n")

    # 检查是否已有 gaia/wise 行
    for line in lines:
        if line.startswith("| Gaia Sep ") or line.startswith("| WISE "):
            return False  # 已补过，跳过

    # 清理 + 重建：在 Source ID 之后、Label 之前插入
    cleaned = [l for l in lines
               if not l.startswith("| Gaia Sep ") and not l.startswith("| WISE ")]

    new_lines = []
    state = "before_sid"
    for line in cleaned:
        if state == "before_sid" and line.startswith("| Source ID "):
            new_lines.append(line)
            state = "after_sid"
            continue

        if state == "after_sid" and line.startswith("| Label "):
            if has_gaia:
                new_lines.append(f"| Gaia Sep | {gaia_sep:.3f} arcsec |")
            if has_wise:
                w1s = f"{w1mag:.3f}" if w1mag is not None else "-"
                w2s = f"{w2mag:.3f}" if w2mag is not None else "-"
                w12s = f"{w1_w2:.3f}" if w1_w2 is not None else "-"
                new_lines.append(f"| WISE W1/W2/W1-W2 | {w1s} / {w2s} / {w12s} |")
            new_lines.append(line)
            state = "after_insert"
            continue

        new_lines.append(line)

    with open(md_path, "w") as f:
        f.write("\n".join(new_lines))

    return True


# ═══════════════════════════════════════════════════
# CLI 模式：扫描 sources/ 目录批量补入
# ═══════════════════════════════════════════════════

def _list_collections(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r[0] for r in cur.fetchall()]


def _fetch_meta(conn, collections, objectID):
    for tbl in collections:
        row = conn.execute(
            f"SELECT gaia_sep, w1mag, w2mag, w1_w2 FROM [{tbl}] WHERE objectID=?",
            (objectID,)
        ).fetchone()
        if row is not None:
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description="补入 WFST registry 元数据到 analysis.md")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写盘")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有的 gaia_sep/WISE 行")
    args = parser.parse_args()

    if not os.path.isfile(REGISTRY_PATH):
        print(f"ERROR: registry 不存在: {REGISTRY_PATH}")
        sys.exit(1)
    if not os.path.isdir(SOURCES_DIR):
        print(f"ERROR: sources 目录不存在: {SOURCES_DIR}")
        sys.exit(1)

    conn = sqlite3.connect(REGISTRY_PATH)
    collections = _list_collections(conn)

    source_dirs = sorted([
        d for d in os.listdir(SOURCES_DIR)
        if d.startswith("WFST_") and os.path.isdir(os.path.join(SOURCES_DIR, d))
    ])

    print(f"Registry: {len(collections)} tables, Sources: {len(source_dirs)} WFST dirs")
    if args.dry_run:
        print("[DRY RUN]")
    if args.force:
        print("[FORCE]")
    print()

    updated = 0
    skipped = 0
    no_meta = 0

    for source_id in source_dirs:
        md_path = os.path.join(SOURCES_DIR, source_id, "analysis.md")
        if not os.path.isfile(md_path):
            no_meta += 1
            continue

        objectID = source_id.replace("WFST_", "", 1)

        # 如果 force，先清除已有的 gaia/wise 行
        if args.force:
            with open(md_path, "r") as f:
                content = f.read()
            lines = content.split("\n")
            cleaned = [l for l in lines
                       if not l.startswith("| Gaia Sep ") and not l.startswith("| WISE ")]
            with open(md_path, "w") as f:
                f.write("\n".join(cleaned))

        meta = _fetch_meta(conn, collections, objectID)
        if meta is None:
            no_meta += 1
            continue

        gaia_sep, w1mag, w2mag, w1_w2 = meta

        if args.dry_run:
            print(f"  [dry-run] {source_id}: Gaia={gaia_sep}, W1={w1mag}, W2={w2mag}, W1-W2={w1_w2}")
        else:
            ok = enrich_analysis_md(md_path, gaia_sep, w1mag, w2mag, w1_w2)
            if ok:
                print(f"  [ok] {source_id}")
                updated += 1
            else:
                skipped += 1

    conn.close()
    print(f"\nDone: {updated} updated, {skipped} skipped, {no_meta} no meta")


if __name__ == "__main__":
    main()