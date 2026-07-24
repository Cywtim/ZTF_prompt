#!/usr/bin/env python3
"""Quick eval: only sources WITH cutout.png, as both test & few-shot."""

import sys, os, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from classify import sample_few_shot, build_prompt, call_api, parse_response, promt_load_index

os.environ.setdefault("NO_PROXY", "api.llm.ustc.edu.cn")

MODEL = "qwen3.6-chat"
N_SHOT = 1
TEST_PER_CLASS = 3
CLASSES = ["TDE", "SN"]

def has_cutout(sid):
    return (config.SOURCES_DIR / sid / "cutout.png").exists()

def main():
    idx = promt_load_index()
    
    # Filter: only sources with cutout
    test_ids = []
    for cls in CLASSES:
        pool = [s for s, v in idx.items() if v["label"] == cls and has_cutout(s)]
        n = min(TEST_PER_CLASS, len(pool))
        chosen = random.sample(pool, n)
        test_ids.extend([(s, cls) for s in chosen])
        print(f"{cls}: {n}/{len(pool)} selected for test")
    
    # For few-shot pool: ALL cutout sources (exclude test set)
    test_set = set(s for s, _ in test_ids)
    
    print(f"\nEval: {len(test_ids)} sources, multimodal {N_SHOT}-shot (cutout-only)\n")
    
    correct = 0
    results = []
    
    for i, (sid, true_label) in enumerate(test_ids):
        print(f"[{i+1}/{len(test_ids)}] {sid} (true={true_label}) ...", end=" ", flush=True)
        
        # Build few-shot from cutout-only pool
        few_shot = []
        for cls in CLASSES:
            pool = [s for s, v in idx.items() if v["label"] == cls and has_cutout(s) and s not in test_set]
            chosen = random.sample(pool, min(N_SHOT, len(pool)))
            for s in sorted(chosen):
                few_shot.append((s, cls))
        
        try:
            messages = build_prompt(sid, few_shot, "multimodal")
            raw, usage = call_api(messages, model=MODEL)
            parsed = parse_response(raw)
        except Exception as e:
            print(f"FAIL ({e})")
            continue
        
        pred = parsed.get("classification", {}).get("label", "?")
        conf = parsed.get("classification", {}).get("confidence", "?")
        score = parsed.get("classification", {}).get("score", 0)
        ok = (pred == true_label)
        if ok: correct += 1
        status = "OK" if ok else ("UNSURE" if pred == "Unsure" else "WRONG")
        print(f"pred={pred} conf={conf} score={score:.2f} [{status}]")
        results.append({"sid": sid, "true": true_label, "pred": pred, "ok": ok, "score": score})
    
    # Summary
    acc = correct / len(results) * 100 if results else 0
    unsure = sum(1 for r in results if r["pred"] == "Unsure")
    wrong = sum(1 for r in results if not r["ok"] and r["pred"] != "Unsure")
    print(f"\nAccuracy: {correct}/{len(results)} ({acc:.0f}%)")
    print(f"Unsure: {unsure}, Errors: {wrong}")
    for r in results:
        if not r["ok"]:
            print(f"  {r['sid']}: true={r['true']} pred={r['pred']} score={r['score']:.2f}")

if __name__ == "__main__":
    main()