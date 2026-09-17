#!/usr/bin/env python3
"""补跑 cadence5 失败的源（只重新分类，不重新生成 md/plot）。

失败根因是 LLM API 网络错误（upstream connect error / timeout），
故只重跑 classify_pipeline，并加重试（最多 5 次，指数退避）。
"""
import sys, csv, time, argparse
sys.path.insert(0, "/home/cyan/AppData/VScode/TDeck/ZTF_prompt")

CSV_PATH = "/home/cyan/AppData/VScode/TDeck/ZTF_prompt/ztf_quality_selection.csv"
SUMMARY = "/tmp/cadence5_summary.csv"

# 全局 label 映射（子进程 fork 后可见）
LABEL_MAP = {}


def classify_with_retry(source_id, max_retry=5):
    from classify import classify_pipeline
    import config
    last_err = None
    for attempt in range(max_retry):
        try:
            r = classify_pipeline(source_id, model=config.MODEL, cot=False)
            return r, None
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt * 3)
    return None, last_err


def worker(sid):
    """模块级函数，可被 pickle。"""
    src = sid.replace("_PRF_cadence5", "")
    true_lbl = LABEL_MAP.get(src, "?")
    r, err = classify_with_retry(sid)
    if r is None:
        return {"id": sid, "status": "fail", "true": true_lbl, "reason": str(err)[:80]}
    return {"id": sid, "status": "ok", "true": true_lbl,
            "pred": r.get("label"), "score": r.get("score"), "conf": r.get("confidence")}


def main():
    global LABEL_MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    fails = [r["id"] for r in csv.DictReader(open(SUMMARY)) if r["status"] == "fail"]
    print(f"待补跑 fail 源数: {len(fails)}", flush=True)

    LABEL_MAP = {r["source_id"]: r["label"] for r in csv.DictReader(open(CSV_PATH))}

    ok = fail = 0
    results = []
    import multiprocessing as mp

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, fails), 1):
            results.append(res)
            if res["status"] == "ok": ok += 1
            else: fail += 1
            el = (time.time() - t0) / 60
            line = f"[{i}/{len(fails)}] {res['status']} {res['id']}"
            if res.get("pred"): line += f" {res['true']}->{res['pred']} score={res.get('score')}"
            if res.get("reason"): line += f" ({res['reason'][:50]})"
            print(f"{line} | ok={ok} fail={fail} {el:.1f}min", flush=True)

    out = "/tmp/cadence5_refill.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "true", "pred", "score", "conf", "reason"])
        w.writeheader()
        w.writerows(results)
    print(f"\n补跑完成: ok={ok} fail={fail}")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
