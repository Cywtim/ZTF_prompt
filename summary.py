#!/usr/bin/env python3
"""
summary.py - Summarize existing classification results (read-only, no API calls).

Usage:
  python summary.py                                    # all results → summary/all.json
  python summary.py --exemplar-set boundary            # boundary set only → summary/boundary.json
  python summary.py --mode multimodal                  # multimodal only → summary/mode_multimodal.json
  python summary.py --exemplar-set default --mode multimodal  # → summary/default_multimodal.json
  python summary.py --verbose                          # list individual sources
"""

import sys, json, argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

import config

SUMMARY_DIR = config.PROJECT_ROOT / "summary"


# ── I/O ────────────────────────────────────────────

def load_results(mode=None, exemplar_set=None):
    """Load result JSONs, optionally filtered by mode and exemplar-set."""

    # Load exemplar template for matching (if filtering by exemplar-set)
    exemplar_ids = None
    if exemplar_set:
        tmpl_path = config.TEMPLATES_DIR / f"fewshot_{exemplar_set}.json"
        if exemplar_set == "default":
            tmpl_path = config.TEMPLATES_DIR / "fewshot.json"
        if tmpl_path.exists():
            tmpl = json.loads(tmpl_path.read_text())
            exemplar_ids = set()
            for ids in tmpl.values():
                if isinstance(ids, list):
                    exemplar_ids.update(ids)

    results = {}
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        sid = path.stem
        try:
            r = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  [warn] corrupt JSON: {path}")
            continue

        # Filter by mode
        if mode and r.get("mode") != mode:
            continue

        # Filter by exemplar-set: match stored few_shot IDs against template
        if exemplar_ids is not None:
            stored_ids = set(fs["id"] for fs in r.get("few_shot", []))
            if stored_ids != exemplar_ids:
                continue

        results[sid] = r

    return results


def load_index():
    if config.INDEX_FILE.exists():
        return json.loads(config.INDEX_FILE.read_text())
    return {}


def save_summary(data, exemplar_set=None, mode=None):
    """Save summary JSON to summary/ directory with auto-generated filename."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename from options
    parts = []
    if exemplar_set:
        parts.append(exemplar_set)
    else:
        parts.append("all")
    if mode:
        parts.append(mode)

    fname = "_".join(parts) + ".json"
    path = SUMMARY_DIR / fname

    data["_meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exemplar_set": exemplar_set,
        "mode": mode,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")
    return path


# ── Classification ─────────────────────────────────

def classify_known_unknown(results, idx):
    """Split results into known (has label ≠ unknown) and unknown."""
    known = {}
    unknown = {}
    for sid, r in results.items():
        info = idx.get(sid, {})
        label = info.get("label", "unknown")
        if label == "unknown" or label not in config.CLASSES:
            unknown[sid] = r
        else:
            known[sid] = (r, label)
    return known, unknown


# ── Known ──────────────────────────────────────────

def summary_known(known, verbose=False, min_conf=None):
    """Return metrics dict + print. Returns None if no data."""
    if not known:
        print("No labeled results found.\n")
        return None

    results_list = []
    for sid, (r, true_label) in known.items():
        cls_info = r.get("classification", {})
        pred_label = cls_info.get("label", "?")
        conf = cls_info.get("confidence", "?")
        score = cls_info.get("score", 0)
        unsure_pref = cls_info.get("unsure_preference")

        is_correct = (pred_label == true_label)
        results_list.append({
            "source_id": sid,
            "true": true_label,
            "pred": pred_label,
            "confidence": conf,
            "score": score,
            "unsure_preference": unsure_pref,
            "correct": is_correct,
        })

    if min_conf:
        results_list = [r for r in results_list if r["confidence"] == min_conf]

    # Confusion matrix
    cm = defaultdict(Counter)
    for r in results_list:
        cm[r["true"]][r["pred"]] += 1

    all_labels = sorted(set(r["true"] for r in results_list) | set(r["pred"] for r in results_list))
    all_labels = [l for l in all_labels if l in config.CLASSES]

    print("═══ Known Sources (ground truth available) ═══")
    print(f"Total: {len(results_list)} results\n")

    if not results_list:
        return None

    # Print confusion matrix
    print("Confusion Matrix:")
    header = "           " + "".join(f"{l:>10s}" for l in all_labels)
    print(header)
    for true_l in all_labels:
        row = f"  {true_l:>8s}  "
        for pred_l in all_labels:
            row += f"{cm[true_l][pred_l]:>10d}"
        print(row)

    # Per-class metrics
    print(f"\nPer-Class:")
    print(f"  {'Class':>8s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    per_class = {}
    total_correct = 0
    total_support = 0
    for cls_name in all_labels:
        tp = cm[cls_name][cls_name]
        support = sum(cm[cls_name].values())
        predicted = sum(cm[l][cls_name] for l in all_labels)
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_correct += tp
        total_support += support
        per_class[cls_name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }
        print(f"  {cls_name:>8s}  {precision:>10.3f}  {recall:>10.3f}  {f1:>10.3f}  {support:>8d}")

    acc = total_correct / total_support if total_support > 0 else 0.0
    print(f"\nAccuracy: {acc:.3f} ({total_correct}/{total_support})")

    # Unsure
    unsure = [r for r in results_list if r["pred"] == "Unsure"]
    if unsure:
        print(f"Unsure rate: {len(unsure)}/{len(results_list)} ({100*len(unsure)/len(results_list):.0f}%)")
        for r in unsure:
            pref = f"→{r['unsure_preference']}" if r.get("unsure_preference") else ""
            print(f"  {r['source_id']}  true={r['true']}  Unsure{pref}")

    # Errors
    errors = [r for r in results_list if not r["correct"] and r["pred"] != "Unsure"]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in sorted(errors, key=lambda x: x["score"], reverse=True):
            print(f"  {r['source_id']}  true={r['true']}  pred={r['pred']}  "
                  f"conf={r['confidence']}  score={r['score']:.2f}")

    high_err = [r for r in errors if r["score"] > 0.8]
    if high_err:
        print(f"\n⚠ High-confidence WRONG (score > 0.8): {len(high_err)}")
        for r in high_err:
            print(f"  {r['source_id']}  true={r['true']}  pred={r['pred']}  score={r['score']:.2f}")

    if verbose:
        print(f"\n── All Results ──")
        for r in sorted(results_list, key=lambda x: x["score"], reverse=True):
            mark = "✓" if r["correct"] else ("?" if r["pred"] == "Unsure" else "✗")
            pref = f"→{r['unsure_preference']}" if r.get("unsure_preference") else ""
            print(f"  {mark} {r['source_id']:45s} true={r['true']:5s}  pred={r['pred']+pref:12s}  "
                  f"score={r['score']:.2f}  conf={r['confidence']}")

    print()

    # Build serializable confusion matrix
    cm_serializable = {tl: dict(pc) for tl, pc in cm.items()}

    return {
        "total": len(results_list),
        "accuracy": round(acc, 3),
        "confusion_matrix": cm_serializable,
        "per_class": per_class,
        "unsure_rate": round(len(unsure) / len(results_list), 3) if results_list else 0,
        "unsure_count": len(unsure),
        "error_count": len(errors),
        "high_confidence_errors": len(high_err),
        "errors": errors,
    }


# ── Unknown ────────────────────────────────────────

def summary_unknown(unknown, verbose=False, min_conf=None):
    """Return distribution dict + print. Returns None if no data."""
    if not unknown:
        print("═══ Unknown Sources: no results ═══\n")
        return None

    results_list = []
    for sid, r in unknown.items():
        cls_info = r.get("classification", {})
        results_list.append({
            "source_id": sid,
            "label": cls_info.get("label", "?"),
            "confidence": cls_info.get("confidence", "?"),
            "score": cls_info.get("score", 0),
            "unsure_preference": cls_info.get("unsure_preference"),
        })

    if min_conf:
        results_list = [r for r in results_list if r["confidence"] == min_conf]

    print("═══ Unknown Sources ═══")
    print(f"Total: {len(results_list)} results\n")

    if not results_list:
        return None

    # Distribution
    dist = Counter(r["label"] for r in results_list)
    print("Classification Distribution:")
    for label in ["TDE", "SN", "Unsure"]:
        cnt = dist.get(label, 0)
        pct = 100 * cnt / len(results_list) if results_list else 0
        bar = "█" * int(pct / 5)
        print(f"  {label:>8s}: {cnt:>4d} ({pct:5.1f}%)  {bar}")

    # Confidence distribution
    conf_dist = Counter(r["confidence"] for r in results_list)
    print(f"\nConfidence:")
    for level in ["high", "medium", "low"]:
        cnt = conf_dist.get(level, 0)
        pct = 100 * cnt / len(results_list) if results_list else 0
        print(f"  {level:>6s}: {cnt:>4d} ({pct:5.1f}%)")

    # Unsure detail
    unsure_list = [r for r in results_list if r["label"] == "Unsure"]
    if unsure_list:
        print(f"\nUnsure Details ({len(unsure_list)}):")
        for r in sorted(unsure_list, key=lambda x: x["score"]):
            pref = f"→{r['unsure_preference']}" if r.get("unsure_preference") else ""
            print(f"  {r['source_id']}  pref={pref}  score={r['score']:.2f}")

    if verbose:
        print(f"\n── All Results ──")
        for r in sorted(results_list, key=lambda x: x["score"], reverse=True):
            pref = f"→{r['unsure_preference']}" if r.get("unsure_preference") else ""
            print(f"  {r['source_id']:45s}  pred={r['label']+pref:12s}  "
                  f"score={r['score']:.2f}  conf={r['confidence']}")

    print()

    return {
        "total": len(results_list),
        "distribution": {label: dist.get(label, 0) for label in ["TDE", "SN", "Unsure"]},
        "confidence": {level: conf_dist.get(level, 0) for level in ["high", "medium", "low"]},
        "unsure_count": len(unsure_list),
        "items": results_list,
    }


# ── Overview ───────────────────────────────────────

def summary_all(known, unknown):
    """Print combined overview."""
    n_known = len(known)
    n_unknown = len(unknown)

    all_results = {}
    for sid, (r, label) in known.items():
        all_results[sid] = r
    all_results.update(unknown)

    conf_dist = Counter()
    cls_dist = Counter()
    for r in all_results.values():
        cls_info = r.get("classification", {})
        conf_dist[cls_info.get("confidence", "?")] += 1
        cls_dist[cls_info.get("label", "?")] += 1

    print("═══ Overview ═══")
    print(f"Results: {len(all_results)} total  |  Known: {n_known}  |  Unknown: {n_unknown}")
    print(f"Classes: " + ", ".join(f"{l}: {cls_dist.get(l, 0)}" for l in ["TDE", "SN", "Unsure"]))
    print(f"Confidence: " + ", ".join(f"{l}: {conf_dist.get(l, 0)}" for l in ["high", "medium", "low"]))
    print()

    return {
        "total": len(all_results),
        "known": n_known,
        "unknown": n_unknown,
        "classes": {l: cls_dist.get(l, 0) for l in ["TDE", "SN", "Unsure"]},
        "confidence": {l: conf_dist.get(l, 0) for l in ["high", "medium", "low"]},
    }


# ── CLI ────────────────────────────────────────────

def plot_confusion_matrix(cm_dict, labels, accuracy, save_path):
    """Plot confusion matrix heatmap and save to PNG (academic style)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed. Install with: pip install matplotlib")
        return

    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            matrix[i, j] = cm_dict.get(true_l, {}).get(pred_l, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            row_total = matrix[i].sum()
            pct = f"{100*val/row_total:.0f}%" if row_total > 0 else ""
            text = f"{val}\n({pct})" if pct else str(val)
            color = "white" if val > matrix.max() * 0.5 else "#333"
            ax.text(j, i, text, ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.1%})",
                 fontsize=14, fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=10)

    ax.spines[:].set_visible(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[plot]  {save_path}")


def plot_unknown_distribution(unknown_data, save_path):
    """Plot classification distribution for unknown sources (grouped bar chart)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    items = unknown_data.get("items", [])
    if not items:
        return

    classes = ["TDE", "SN", "Unsure"]
    conf_levels = ["high", "medium", "low"]
    colors = ["#2e86ab", "#a23b72", "#f18f01"]

    # Cross-tab: class × confidence
    cross = {cls: {c: 0 for c in conf_levels} for cls in classes}
    for r in items:
        label = r.get("label", "?")
        conf = r.get("confidence", "?")
        if label in cross and conf in conf_levels:
            cross[label][conf] += 1

    # Build grouped bar data
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (level, color) in enumerate(zip(conf_levels, colors)):
        vals = [cross[cls][level] for cls in classes]
        bars = ax.bar(x + i * width, vals, width, label=level.capitalize(),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Unknown Sources  (n={len(items)})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, edgecolor="#ddd")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[plot]  {save_path}")


def main():
    parser = argparse.ArgumentParser(description="summary.py - summarize classification results")
    parser.add_argument("--mode", choices=["text", "multimodal"],
                        help="filter results by mode")
    parser.add_argument("--exemplar-set", dest="exemplar_set",
                        help="filter by exemplar set (matches templates/fewshot_NAME.json)")
    parser.add_argument("--verbose", action="store_true", help="list individual sources")
    parser.add_argument("--min-conf", choices=["high", "medium", "low"],
                        help="filter by confidence level")
    parser.add_argument("--known-only", action="store_true", help="only labeled sources")
    parser.add_argument("--unknown-only", action="store_true", help="only unlabeled sources")
    parser.add_argument("--no-save", action="store_true", help="skip saving to summary/")
    parser.add_argument("--plot", action="store_true", help="also save confusion matrix plot (PNG)")
    args = parser.parse_args()

    results = load_results(mode=args.mode, exemplar_set=args.exemplar_set)

    if not results:
        print("No results found (check --mode / --exemplar-set filters)")
        return

    idx = load_index()
    known, unknown = classify_known_unknown(results, idx)

    overview = summary_all(known, unknown)

    known_data = None
    if not args.unknown_only:
        known_data = summary_known(known, verbose=args.verbose, min_conf=args.min_conf)

    unknown_data = None
    if not args.known_only:
        unknown_data = summary_unknown(unknown, verbose=args.verbose, min_conf=args.min_conf)

    # Save
    if not args.no_save:
        report = {
            "overview": overview,
            "known": known_data,
            "unknown": unknown_data,
        }
        es = None if args.exemplar_set == "default" else args.exemplar_set
        saved_path = save_summary(report, exemplar_set=es, mode=args.mode)

        # Plot confusion matrix
        if args.plot and known_data and known_data.get("confusion_matrix"):
            cm = known_data["confusion_matrix"]
            labels = sorted(set(cm.keys()) | set(k for v in cm.values() for k in v))
            labels = [l for l in labels if l in config.CLASSES]
            if labels:
                plot_path = saved_path.with_suffix(".png")
                plot_confusion_matrix(cm, labels, known_data.get("accuracy", 0), plot_path)

        # Plot unknown distribution
        if args.plot and unknown_data and unknown_data.get("items"):
            unknown_plot_path = Path(str(saved_path).replace(".json", "_unknown.png"))
            plot_unknown_distribution(unknown_data, unknown_plot_path)


if __name__ == "__main__":
    main()