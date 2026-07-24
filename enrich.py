#!/usr/bin/env python3
"""
enrich.py - Add TDE physics context to classification results.

Reads results/{id}.json, matches each indicator against INDICATOR_WIKI_MAP.json,
writes enriched result to results_enriched/{id}.json.

Usage:
  python enrich.py                     # enrich all results
  python enrich.py ZTF19aadolpe        # enrich single source
  python enrich.py --force             # re-enrich all (overwrite existing)
  python enrich.py --dry-run           # show what would be added, don't write
"""

import sys, os, json, argparse
from pathlib import Path

PROJECT = Path(__file__).parent
RESULTS_DIR = PROJECT / "results"
ENRICHED_DIR = PROJECT / "results_enriched"
MAP_FILE = PROJECT / "INDICATOR_WIKI_MAP.json"


def load_map():
    with open(MAP_FILE) as f:
        return json.load(f)


def enrich_result(result, indicator_map):
    """Add physical_context field to a classification result. Returns modified result."""
    indicators = result.get("reasoning", {}).get("indicators", [])
    classification = result.get("classification", {})
    label = classification.get("label", "?")
    unsure_pref = classification.get("unsure_preference")

    effective_label = label
    if label == "Unsure" and unsure_pref:
        effective_label = unsure_pref

    physical_context = []

    for ind in indicators:
        name = ind.get("name", "")
        direction = ind.get("direction", "")
        value = ind.get("value", "")
        weight = ind.get("weight", 0)

        # Look up in map
        indicator_defs = indicator_map.get("indicators", {}).get(name, {})
        if not indicator_defs:
            continue

        # Try direction-specific, then general
        entry = indicator_defs.get(direction) or indicator_defs.get("general")
        if not entry:
            continue

        context = {
            "indicator": name,
            "value": value,
            "direction": direction,
            "weight": weight,
            "phenomenon": entry.get("phenomenon", ""),
            "mechanism": entry.get("mechanism", ""),
            "wiki_ref": entry.get("wiki_ref", ""),
            "expected": entry.get("expected", ""),
            "caveat": entry.get("caveat", ""),
        }
        # Remove empty fields
        context = {k: v for k, v in context.items() if v}
        physical_context.append(context)

    # Add spectral guidance if TDE (or Unsure→TDE)
    if effective_label == "TDE":
        spectral = indicator_map.get("spectral_guidance", {})
        physical_context.append({
            "indicator": "Spectral Follow-up",
            "mechanism": (
                f"TDE subtypes: {spectral.get('TDE-H','')}, {spectral.get('TDE-He','')}, "
                f"{spectral.get('TDE-H+He','')}. "
                f"Key lines at host z: {spectral.get('key_lines','')}"
            ),
            "wiki_ref": spectral.get("wiki_ref", ""),
        })

    result["physical_context"] = physical_context
    result["physical_summary"] = _generate_summary(result, physical_context, effective_label)
    return result


def _generate_summary(result, physical_context, effective_label):
    """Synthesize a Chinese diagnostic paragraph from all indicators + wiki context."""
    classification = result.get("classification", {})
    label = classification.get("label", "?")
    confidence = classification.get("confidence", "?")
    score = classification.get("score", 0)
    primary = result.get("reasoning", {}).get("primary_signal", "")
    summary = result.get("reasoning", {}).get("summary", "")

    lines = []

    # ── 1. 诊断标题 ──
    if label == "TDE":
        lines.append(f"## 物理诊断：TDE 候选体 (置信度: {confidence}, score={score:.2f})")
    elif label == "Unsure":
        pref = classification.get("unsure_preference", "?")
        lines.append(f"## 物理诊断：不确定 (倾向 {pref}, score={score:.2f})")
    else:
        lines.append(f"## 物理诊断：{label} (置信度: {confidence}, score={score:.2f})")

    lines.append("")

    # ── 2. 逐指标物理解释 ──
    for ctx in physical_context:
        name = ctx.get("indicator", "")
        if name == "Spectral Follow-up":
            continue  # handled separately at end

        direction = ctx.get("direction", "")
        value = ctx.get("value", "")
        phenomenon = ctx.get("phenomenon", "")
        mechanism = ctx.get("mechanism", "")
        wiki_ref = ctx.get("wiki_ref", "")
        expected = ctx.get("expected", "")
        caveat = ctx.get("caveat", "")

        dir_mark = "→" + direction if direction else ""
        line = f"**{name}**{dir_mark}：{value}。"
        if phenomenon:
            line += f" {phenomenon}——{mechanism}"
        elif mechanism:
            line += f" {mechanism}"
        if expected:
            line += f" 预期范围：{expected}。"
        if caveat:
            line += f" ⚠️ {caveat}"
        if wiki_ref:
            line += f" ({wiki_ref})"
        lines.append(line)
        lines.append("")

    # ── 3. 综合判断 ──
    lines.append(f"**综合判断**：{summary}")
    lines.append("")

    # ── 4. TDE 专属：实体对比 + 光谱建议 ──
    if effective_label == "TDE":
        # Find spectral guidance
        spec_ctx = [c for c in physical_context if c.get("indicator") == "Spectral Follow-up"]
        if spec_ctx:
            lines.append(f"**光谱建议**：{spec_ctx[0].get('mechanism','')}")
            lines.append("")

        # Add entity comparison based on indicator pattern
        color_ctx = [c for c in physical_context if c.get("indicator") == "Color Evolution" and c.get("direction") == "TDE"]
        decline_ctx = [c for c in physical_context if c.get("indicator") == "Decline Rate" and c.get("direction") == "TDE"]
        if color_ctx and decline_ctx:
            lines.append("**相似已知源**：该源同时具备强颜色演化和陡衰减，类似于 AT2019qiz (TDE-He 原型) 和 ASASSN-14li 的光变特征。若光谱确认 He II 4686Å 发射线 → TDE-He 亚型。")
        elif color_ctx:
            lines.append("**相似已知源**：颜色演化特征与 AT2019dsg 类似，但需光谱确认是否有 H+He 混合线。")
        elif decline_ctx:
            lines.append("**相似已知源**：陡衰减与 ASASSN-14li 的早期光变类似，需核实是否有 IR echo。")

    elif effective_label == "SN":
        # Check if there are conflicting TDE indicators
        tde_indicators = [c for c in physical_context if c.get("direction") == "TDE"]
        if tde_indicators:
            names = [c["indicator"] for c in tde_indicators]
            lines.append(f"**注意**：该源有 {len(tde_indicators)} 个 TDE 方向指标 ({', '.join(names)})，但综合判断为 SN。可能为 IIb/IIn/SLSN-I 等显示 TDE-like 颜色演化的 SN 亚型。建议核实光谱分类。")

    return "\n".join(lines)


def process_one(source_id, indicator_map, force=False, dry_run=False):
    src = RESULTS_DIR / f"{source_id}.json"
    if not src.exists():
        print(f"  [skip] {source_id} - no result file")
        return False

    dst = ENRICHED_DIR / f"{source_id}.json"
    if dst.exists() and not force:
        print(f"  [skip] {source_id} - already enriched (use --force)")
        return None

    with open(src) as f:
        result = json.load(f)

    enrich_result(result, indicator_map)

    if dry_run:
        ctx_count = len(result.get("physical_context", []))
        label = result.get("classification", {}).get("label", "?")
        print(f"  [dry-run] {source_id} ({label}): {ctx_count} context entries")
        for ctx in result.get("physical_context", []):
            print(f"    - {ctx.get('indicator','?')}: {ctx.get('phenomenon','') or ctx.get('mechanism','')[:80]}")
        return True

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    ctx_count = len(result.get("physical_context", []))
    print(f"  [done] {source_id} → enriched ({ctx_count} contexts)")
    return True


def process_all(indicator_map, force=False, dry_run=False, max_files=None):
    files = sorted(RESULTS_DIR.glob("*.json"))
    if max_files:
        files = files[:max_files]

    ok, skipped, failed = 0, 0, 0
    for i, f in enumerate(files):
        sid = f.stem
        result = process_one(sid, indicator_map, force=force, dry_run=dry_run)
        if result is None:
            skipped += 1
        elif result:
            ok += 1
        else:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(files)} (ok={ok} skip={skipped} fail={failed})")

    print(f"\nDone: {ok} enriched, {skipped} skipped, {failed} failed (total {len(files)})")


def main():
    parser = argparse.ArgumentParser(
        description="enrich.py - add TDE physics context to classification results")
    parser.add_argument("source_id", nargs="?", help="Source ID (omit for --all)")
    parser.add_argument("--all", action="store_true", help="Process all results")
    parser.add_argument("--force", action="store_true", help="Re-enrich existing")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    parser.add_argument("--max", type=int, help="Max files in --all mode")
    args = parser.parse_args()

    indicator_map = load_map()
    print(f"Loaded {len(indicator_map.get('indicators',{}))} indicator definitions")

    if args.source_id:
        process_one(args.source_id, indicator_map, force=args.force, dry_run=args.dry_run)
    elif args.all or not args.source_id:
        process_all(indicator_map, force=args.force, dry_run=args.dry_run, max_files=args.max)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()