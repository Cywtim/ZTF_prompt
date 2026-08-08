#!/usr/bin/env python3
"""
eval.py - Evaluate LLM classification accuracy against ground truth

Usage:
  python eval.py                    # default: 30 per class, 3-shot
  python eval.py --test-size 50 --n-shot 5
  python eval.py --classes TDE,SN   # binary only
  python eval.py --verbose          # show per-source results
"""

import sys, json, random, argparse
from pathlib import Path
from collections import defaultdict, Counter

import config
from classify import classify_one, promt_load_index, parse_response


def evaluate(test_size=30, n_shot=3, classes=None, mode="text", model=None, verbose=False, cot=False):
    """Run evaluation on labeled sources.
    
    Args:
        cot: if True, enable Chain-of-Thought reasoning.
    """
    if classes is None:
        classes = config.CLASSES

    idx = promt_load_index()
    if not idx:
        print("index.json is empty. Run promt.py first.")
        return

    # Split: hold out test set per class, rest = few-shot pool
    test_ids = []
    for cls_name in classes:
        pool = [sid for sid, info in idx.items() if info["label"] == cls_name]
        if not pool:
            print(f"  [warn] No labeled sources for class '{cls_name}'")
            continue
        n = min(test_size, len(pool))
        chosen = random.sample(pool, n)
        test_ids.extend([(sid, cls_name) for sid in chosen])

    test_set = set(sid for sid, _ in test_ids)

    if not test_ids:
        print("No test sources available.")
        return

    mode_str = f"{n_shot}-shot, mode={mode}"
    if cot:
        mode_str += ", CoT"
    print(f"Evaluation: {len(test_ids)} test sources ({len(classes)} classes), "
          f"{mode_str}")
    print(f"Few-shot pool: {len(idx) - len(test_set)} sources "
          f"(test set excluded)\n")

    # Run classification for each test source
    results = []
    correct = 0
    errors = []

    for i, (sid, true_label) in enumerate(test_ids):
        print(f"[{i+1}/{len(test_ids)}] {sid} (true={true_label}) ...", end=" ", flush=True)

        # Build few-shot from pool (exclude test set + current source)
        exclude = test_set.copy()

        from classify import sample_few_shot, build_prompt, call_api
        few_shot = sample_few_shot(idx, n_per_class=n_shot, exclude=exclude)
        if n_shot > 0 and not few_shot:
            print("FAIL (no few-shot available)")
            continue

        messages = build_prompt(sid, few_shot, mode, cot=cot)
        try:
            raw_text, usage = call_api(messages, model=model)
            parsed = parse_response(raw_text)
        except Exception as e:
            print(f"FAIL ({e})")
            errors.append({"source_id": sid, "true": true_label, "error": str(e)})
            continue

        pred_label = parsed.get("classification", {}).get("label", "?")
        conf = parsed.get("classification", {}).get("confidence", "?")
        score = parsed.get("classification", {}).get("score", 0)
        unsure_pref = parsed.get("classification", {}).get("unsure_preference")

        is_correct = (pred_label == true_label)
        if pred_label == "Unsure":
            status = f"UNSURE→{unsure_pref}" if unsure_pref else "UNSURE"
        elif is_correct:
            correct += 1
            status = "OK"
        else:
            status = "WRONG"

        print(f"pred={pred_label} conf={conf} [{status}]")

        results.append({
            "source_id": sid,
            "true": true_label,
            "pred": pred_label,
            "confidence": conf,
            "score": score,
            "unsure_preference": unsure_pref,
            "correct": is_correct,
            "tokens": usage.total_tokens if usage else 0,
        })

    # ---- Metrics ----
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")

    # Confusion matrix
    cm = defaultdict(Counter)
    for r in results:
        cm[r["true"]][r["pred"]] += 1

    all_labels = sorted(set(r["true"] for r in results) | set(r["pred"] for r in results))
    print(f"\n  Confusion Matrix:")
    header = "           " + "".join(f"{l:>10s}" for l in all_labels)
    print(header)
    for true_l in all_labels:
        row = f"  {true_l:>8s}  "
        for pred_l in all_labels:
            row += f"{cm[true_l][pred_l]:>10d}"
        print(row)

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    macro_p, macro_r, macro_f1 = [], [], []
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

        if support > 0:
            macro_p.append(precision)
            macro_r.append(recall)
            macro_f1.append(f1)

        print(f"  {cls_name:>10s}  {precision:>10.3f}  {recall:>10.3f}  {f1:>10.3f}  {support:>10d}")

    accuracy = total_correct / total_support if total_support > 0 else 0.0
    print(f"\n  Accuracy:  {accuracy:.3f} ({total_correct}/{total_support})")
    if macro_f1:
        print(f"  Macro F1:  {sum(macro_f1)/len(macro_f1):.3f}")

    # Unsure analysis
    unsure = [r for r in results if r["pred"] == "Unsure"]
    unsure_by_true = Counter()
    unsure_by_pref = Counter()
    if unsure:
        for r in unsure:
            key = f"true={r['true']}"
            if r.get("unsure_preference"):
                key += f"→{r['unsure_preference']}"
            unsure_by_true[key] += 1
            unsure_by_pref[r.get("unsure_preference", "none")] += 1
        print(f"\n  Unsure rate: {len(unsure)}/{len(results)} ({100*len(unsure)/len(results):.0f}%)")
        for k, v in sorted(unsure_by_true.items()):
            print(f"    {v:>3d}x {k}")
        # Show preference distribution
        prefs = [(k, v) for k, v in unsure_by_pref.items() if k != "none"]
        if prefs:
            pref_str = ", ".join(f"{k}: {v}" for k, v in sorted(prefs))
            print(f"    Preference: {pref_str}")

    # Error analysis
    wrong = [r for r in results if not r["correct"] and r["pred"] != "Unsure"]
    if wrong:
        print(f"\n  Errors ({len(wrong)}):")
        for r in sorted(wrong, key=lambda x: x["score"], reverse=True):
            print(f"    {r['source_id']}  true={r['true']}  pred={r['pred']}  "
                  f"score={r['score']:.2f}  conf={r['confidence']}")

    # Low-confidence cases (score < 0.5 regardless of correctness)
    low_conf = [r for r in results if r["score"] < 0.5]
    if low_conf:
        print(f"\n  Low-confidence (score < 0.5, {len(low_conf)}):")
        for r in low_conf:
            mark = "correct" if r["correct"] else "WRONG"
            print(f"    {r['source_id']}  true={r['true']}  pred={r['pred']}  "
                  f"score={r['score']:.2f}  [{mark}]")

    # High-confidence errors
    high_err = [r for r in wrong if r["score"] > 0.8]
    if high_err:
        print(f"\n  High-confidence WRONG (score > 0.8, {len(high_err)}):")
        for r in high_err:
            print(f"    {r['source_id']}  true={r['true']}  pred={r['pred']}  score={r['score']:.2f}")

    # Token stats
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n  Total tokens: {total_tokens} (~{total_tokens/len(results):.0f}/source)")

    # Save evaluation report
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": {
            "test_size": test_size, "n_shot": n_shot,
            "classes": classes, "mode": mode, "model": model or config.MODEL,
        },
        "accuracy": accuracy,
        "macro_f1": sum(macro_f1)/len(macro_f1) if macro_f1 else 0,
        "confusion_matrix": {tl: dict(pc) for tl, pc in cm.items()},
        "per_class": {
            cls_name: {
                "precision": tp / predicted if (predicted := sum(cm[l][cls_name] for l in all_labels)) > 0 else 0,
                "recall": tp / support if (support := sum(cm[cls_name].values())) > 0 else 0,
                "support": support,
            }
            for cls_name in all_labels
            if (tp := cm[cls_name][cls_name]) >= 0
        },
        "errors": wrong,
        "unsure": {
            "count": len(unsure),
            "rate": len(unsure) / len(results) if results else 0,
            "by_true": dict(unsure_by_true) if unsure else {},
            "preference": dict(unsure_by_pref) if unsure else {},
            "items": unsure,
        },
        "low_confidence": low_conf,
        "total_tokens": total_tokens,
    }

    report_path = config.PROJECT_ROOT / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="eval.py - evaluate LLM classification")
    parser.add_argument("--test-size", type=int, default=30, help="sources per class for testing")
    parser.add_argument("--n-shot", type=int, default=3, help="few-shot examples per class")
    parser.add_argument("--classes", default=",".join(config.CLASSES),
                        help="comma-separated class list")
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text")
    parser.add_argument("--model", help="model override")
    parser.add_argument("--verbose", action="store_true", help="show detailed per-source results")
    parser.add_argument("--cot", action="store_true", help="enable Chain-of-Thought reasoning")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible test sets")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    classes = [c.strip() for c in args.classes.split(",")]
    evaluate(
        test_size=args.test_size,
        n_shot=args.n_shot,
        classes=classes,
        mode=args.mode,
        model=args.model,
        verbose=args.verbose,
        cot=args.cot,
    )


if __name__ == "__main__":
    main()