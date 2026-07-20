#!/usr/bin/env python3
"""
classify.py - Classify light curves using LLM via API

Usage:
  python classify.py WFST_J101658
  python classify.py --all-unlabeled
  python classify.py WFST_J101658 --n-shot 3 --mode text
"""

import sys, os, json, random, argparse, base64
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI

import config


def _get_client():
    """Create a fresh API client (avoids SSL session caching issues)."""
    return OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY, timeout=300)


# ═══════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════

def read_md(source_id, include_raw=False):
    """Read analysis.md. If include_raw=False, strip Section 4 (raw data) to save tokens."""
    path = config.SOURCES_DIR / source_id / "analysis.md"
    if not path.exists():
        return f"[analysis.md not found for {source_id}]"
    content = path.read_text()
    if not include_raw:
        # Keep only Sections 1-3 + 5 (drop raw data Section 4)
        parts = content.split("## Section 4:")
        if len(parts) >= 2:
            # Remove Section 4, keep Section 5
            after_s4 = parts[1]
            sec5_start = after_s4.find("## Section 5:")
            if sec5_start >= 0:
                content = parts[0] + after_s4[sec5_start:]
            else:
                content = parts[0]
    return content


def read_image_b64(source_id):
    path = config.SOURCES_DIR / source_id / "lightcurve.png"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def has_image(source_id):
    return (config.SOURCES_DIR / source_id / "lightcurve.png").exists()


def sample_few_shot(idx, n_per_class=None, exclude=None):
    """Sample few-shot examples from labeled sources, excluding certain IDs."""
    if n_per_class is None:
        n_per_class = config.N_SHOT_TEXT
    if exclude is None:
        exclude = set()
    selected = []
    for cls_name in config.CLASSES:
        pool = [sid for sid, info in idx.items()
                if info["label"] == cls_name and sid not in exclude]
        if not pool:
            continue
        n = min(n_per_class, len(pool))
        chosen = random.sample(pool, n)
        for sid in sorted(chosen):
            selected.append((sid, cls_name))
    return selected


def _make_system_prompt():
    """Build system prompt dynamically from config.CLASSES."""
    classes_str = "|".join(config.CLASSES)
    class_list = ", ".join(config.CLASSES)

    return (
        f"You are an astronomical transient classifier specializing in ZTF light curves. "
        f"Classify each source as one of: {class_list}.\n\n"
        "## Physical Discriminators (check in priority order)\n\n"
        "### 1. Color Evolution (g − r) — STRONGEST signal\n"
        "- delta_g-r > 10 uJy (strong red→blue evolution): STRONG TDE indicator\n"
        "- delta_g-r < 5 uJy (flat/mild evolution): typical SN\n"
        "- Color staying RED throughout: favors steady sources, flag as uncertain\n\n"
        "### 2. Timescale\n"
        "- Rise time < 60 days: favors TDE\n"
        "- Rise time 30−200 days: favors SN\n"
        "- Total time span < 200 days: favors TDE\n"
        "- Total time span > 1000 days: unusual — flag as uncertain\n\n"
        "### 3. Decline Shape\n"
        "- Steep power-law decline (>1 uJy/d sustained): favors TDE\n"
        "- Plateau or very slow decline (<0.1 uJy/d): favors SN\n\n"
        "### 4. Data Quality\n"
        "- Total points < 10: inherently LOW confidence\n"
        "- Single-band only: no color information, be cautious\n"
        "- If quality is poor and signals are ambiguous, prefer medium/low confidence\n\n"
        "## Decision Logic\n"
        "1. Check color evolution FIRST. If delta_g-r > 10 → TDE (unless rise >200d).\n"
        "2. If color is ambiguous, use timescale + decline as tiebreakers.\n"
        "3. IGNORE the auto-generated hints in Section 3 when they contradict "
        "these physical rules — those hints can be wrong.\n"
        "4. If conflicting signals: flag as medium confidence and explain why.\n\n"
        "## Response Format\n"
        "Output ONLY a JSON object (no markdown, no thinking process):\n"
        f'{{"classification":{{"label":"{classes_str}","confidence":"high|medium|low","score":0.0-1.0}},'
        f'"reasoning":{{"primary_signal":"...","indicators":[{{'
        f'"name":"...","value":"...","weight":0.0-1.0,"direction":"{classes_str}"}}]}},'
        '"quality":{"flags":[]}}'
    )


def build_prompt(target_id, few_shot, mode="text"):
    """Build the API prompt. Returns messages list."""
    system_text = _make_system_prompt()

    user_content = []

    # Few-shot header
    user_content.append({"type": "text", "text":
                         "## Few-Shot Examples\n"
                         "Study these labeled examples using the physical rules above.\n"})

    # Few-shot examples
    for fs_id, fs_label in few_shot:
        user_content.append({"type": "text", "text": f"### Example: {fs_label} ({fs_id})\n"})

        if mode == "multimodal" and has_image(fs_id):
            b64 = read_image_b64(fs_id)
            if b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        md_text = read_md(fs_id)
        user_content.append({"type": "text", "text": md_text + "\n"})

    # Target source
    user_content.append({"type": "text", "text": "\n---\n\n## Target: Classify This Source\n\n"})

    if mode == "multimodal" and has_image(target_id):
        b64 = read_image_b64(target_id)
        if b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    target_md = read_md(target_id)
    user_content.append({"type": "text", "text": target_md})

    if mode == "text":
        # Flatten to single string for text-only API
        flat = "\n".join(
            item["text"] for item in user_content if item["type"] == "text"
        )
        return [{"role": "system", "content": system_text},
                {"role": "user", "content": flat}]

    return [{"role": "user", "content": user_content}]


# ═══════════════════════════════════════════════════
# API call
# ═══════════════════════════════════════════════════

MAX_RETRIES = 3

def call_api(messages, model=None):
    """Call the LLM API and return response text."""
    if model is None:
        model = config.MODEL
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                temperature=config.TEMPERATURE,
                messages=messages,
                max_tokens=2000,
            )
            msg = response.choices[0].message
            # Handle USTC API proxy: content may be None, fall back to reasoning_content
            text = msg.content or getattr(msg, "reasoning_content", None) or ""
            return text, response.usage
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e}")
                import time
                time.sleep(2 ** attempt)
    raise last_error


# ═══════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════

def parse_response(raw_text):
    """Extract JSON from LLM response. Handles markdown code blocks."""
    text = raw_text.strip()
    # Try to extract from ```json ... ``` block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find first { and last }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    return {"error": "json_parse_failed", "raw": raw_text}


# ═══════════════════════════════════════════════════
# Result saving
# ═══════════════════════════════════════════════════

def save_result(source_id, parsed, raw_response, usage, mode, model, few_shot):
    result = {
        "source_id": source_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "tokens": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
        "classification": parsed.get("classification", {}),
        "reasoning": parsed.get("reasoning", {}),
        "quality": parsed.get("quality", {}),
        "few_shot": [{"id": fs[0], "label": fs[1]} for fs in few_shot],
        "_raw_response": raw_response,
    }
    out_path = config.RESULTS_DIR / f"{source_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ═══════════════════════════════════════════════════
# Classify one source
# ═══════════════════════════════════════════════════

def classify_one(source_id, mode="text", n_shot=None, model=None, force=False):
    """Classify a single source."""
    # Check if already done
    result_path = config.RESULTS_DIR / f"{source_id}.json"
    if result_path.exists() and not force:
        print(f"  [skip] {source_id} -- result already exists (use --force to redo)")
        return None

    # Check source exists
    md_path = config.SOURCES_DIR / source_id / "analysis.md"
    if not md_path.exists():
        print(f"  [error] {source_id} -- analysis.md not found. Run promt.py first.")
        return None

    # Load index
    idx = promt_load_index()

    # Sample few-shot (exclude the target itself)
    if n_shot is None:
        n_shot = config.N_SHOT_MULTIMODAL if mode == "multimodal" else config.N_SHOT_TEXT

    few_shot = sample_few_shot(idx, n_per_class=n_shot, exclude={source_id})
    if not few_shot:
        print(f"  [error] {source_id} -- no labeled examples available in index.json")
        return None

    # Build prompt
    print(f"  {source_id}: building prompt ({mode}, {len(few_shot)} few-shot)")
    messages = build_prompt(source_id, few_shot, mode)

    # Call API
    raw_text, usage = call_api(messages, model=model)

    # Parse
    parsed = parse_response(raw_text)

    # Save
    result = save_result(source_id, parsed, raw_text, usage, mode, model or config.MODEL, few_shot)

    # Print summary
    cls_info = result.get("classification", {})
    label = cls_info.get("label", "?")
    conf = cls_info.get("confidence", "?")
    score = cls_info.get("score", 0)
    print(f"  [done] {source_id} -> {label} ({conf}, score={score:.2f}) "
          f"[{result['tokens']['total']} tokens]")

    return result


def classify_all_unlabeled(mode="text", n_shot=None, model=None, force=False):
    """Classify all sources labeled 'unknown' in index.json."""
    idx = promt_load_index()
    unlabeled = [sid for sid, info in idx.items() if info["label"] == "unknown"]
    if not unlabeled:
        print("No unlabeled sources found in index.json")
        return
    print(f"Classifying {len(unlabeled)} unlabeled sources...")
    done = 0
    for sid in sorted(unlabeled):
        result = classify_one(sid, mode=mode, n_shot=n_shot, model=model, force=force)
        if result:
            done += 1
    print(f"\nDone: {done}/{len(unlabeled)} classified")


def show_result(source_id):
    path = config.RESULTS_DIR / f"{source_id}.json"
    if not path.exists():
        print(f"No result found for {source_id}")
        return
    r = json.loads(path.read_text())
    c = r.get("classification", {})
    q = r.get("quality", {})
    reasoning = r.get("reasoning", {})
    print(f"\n{'='*60}")
    print(f"  {source_id}")
    print(f"{'='*60}")
    print(f"  Classification: {c.get('label', '?')}  ({c.get('confidence', '?')}, score={c.get('score', 0):.2f})")
    print(f"  Model: {r.get('model', '?')}  |  Mode: {r.get('mode', '?')}  |  {r['tokens']['total']} tokens")
    if reasoning.get("summary"):
        print(f"\n  Summary: {reasoning['summary']}")
    if reasoning.get("indicators"):
        print(f"\n  Key Indicators:")
        for ind in reasoning["indicators"]:
            direction_mark = "->TDE" if ind.get("direction") == "TDE" else ("->SN" if ind.get("direction") == "SN" else "")
            print(f"    [{ind.get('weight', 0):.2f}] {ind.get('name', '?')}: {ind.get('value', '?')} {direction_mark}")
    if q.get("flags"):
        print(f"\n  Quality Flags:")
        for f in q["flags"]:
            if isinstance(f, dict):
                print(f"    [{f.get('severity', '?')}] {f.get('flag', '?')}: {f.get('detail', '?')}")
            else:
                print(f"    - {f}")
    print(f"{'='*60}\n")


def promt_load_index():
    if config.INDEX_FILE.exists():
        return json.loads(config.INDEX_FILE.read_text())
    return {}


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="classify.py - LLM-based light curve classification")
    parser.add_argument("source_id", nargs="?", help="source ID to classify or show results")
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text")
    parser.add_argument("--n-shot", type=int, help="few-shot examples per class")
    parser.add_argument("--model", help="model override")
    parser.add_argument("--all-unlabeled", action="store_true", help="classify all unlabeled")
    parser.add_argument("--results", action="store_true", help="show saved results")
    parser.add_argument("--force", action="store_true", help="reclassify even if result exists")
    args = parser.parse_args()

    if args.all_unlabeled:
        classify_all_unlabeled(mode=args.mode, n_shot=args.n_shot, model=args.model, force=args.force)
    elif args.results and args.source_id:
        show_result(args.source_id)
    elif args.source_id:
        classify_one(args.source_id, mode=args.mode, n_shot=args.n_shot, model=args.model, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()