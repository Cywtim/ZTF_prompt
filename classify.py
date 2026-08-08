#!/usr/bin/env python3
"""
classify.py - Classify light curves using LLM via API

Usage:
  python classify.py WFST_J101658
  python classify.py --all-unlabeled
  python classify.py WFST_J101658 --n-shot 3 --mode text
"""

import sys, os, json, random, argparse, base64, re
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
        # Keep Sections 1-2 + 4 (drop raw data Section 3)
        parts = content.split("## Section 3:")
        if len(parts) >= 2:
            # Remove Section 4, keep Section 5
            after_s4 = parts[1]
            sec5_start = after_s4.find("## Section 4:")
            if sec5_start >= 0:
                content = parts[0] + after_s4[sec5_start:]
            else:
                content = parts[0]
    return content


def read_image_b64(source_id, filename="lightcurve.png"):
    path = config.SOURCES_DIR / source_id / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def has_image(source_id, filename="lightcurve.png"):
    return (config.SOURCES_DIR / source_id / filename).exists()



def _peak_flux(source_id):
    """Extract peak flux (uJy) from analysis.md. Returns None if not found."""
    md_path = config.SOURCES_DIR / source_id / "analysis.md"
    if not md_path.exists():
        return None
    text = md_path.read_text()
    m = re.search(r"Peak flux \| ([\d.]+) uJy", text)
    return float(m.group(1)) if m else None

def _load_fewshot_json(exemplar_set=None):
    """Load curated few-shot exemplars from templates/fewshot{_set}.json.
    Returns dict {class_name: [source_id, ...]} or None if file not found.
    """
    if exemplar_set:
        path = config.TEMPLATES_DIR / f"fewshot_{exemplar_set}.json"
    else:
        path = config.TEMPLATES_DIR / "fewshot.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def sample_few_shot(idx, n_per_class=None, exclude=None, exemplar_set=None):
    """Sample few-shot examples from labeled sources, excluding certain IDs.

    Priority:
      1. templates/fewshot.json (or fewshot_{exemplar_set}.json) — curated, reproducible
      2. Fall back to random sampling from index.json
    """
    if exclude is None:
        exclude = set()

    # Try curated fewshot.json first
    fs = _load_fewshot_json(exemplar_set)
    if fs:
        selected = []
        for cls_name in config.CLASSES:
            for sid in fs.get(cls_name, []):
                if sid in exclude:
                    continue
                if sid not in idx:
                    print(f"  [warn] fewshot exemplar '{sid}' not in index.json, skipping")
                    continue
                selected.append((sid, cls_name))
        if selected:
            print(f"  [fewshot] using {len(selected)} curated exemplars from "
                  f"fewshot{'_' + exemplar_set if exemplar_set else ''}.json")
            return selected
        print(f"  [warn] fewshot{'_' + exemplar_set if exemplar_set else ''}.json "
              f"exists but no valid exemplars, falling back to random")

    # Fallback: random sampling
    if n_per_class is None:
        n_per_class = config.N_SHOT_TEXT
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


def _make_system_prompt(cot=False, has_cutout=False):
    """Build system prompt from template files in prompts/.

    Args:
        cot: If True, include Chain-of-Thought reasoning instructions.
        has_cutout: If True, include host galaxy cutout guidance.
    """
    classes_str = "|".join(config.CLASSES)
    class_list = ", ".join(config.CLASSES)
    real_classes = [c for c in config.CLASSES if c != "Unsure"]
    real_str = "|".join(real_classes)

    # Load versioned template
    template_path = config.PROMPTS_DIR / f"system_{config.PROMPT_VERSION}.txt"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}\n"
            f"Available versions: {sorted(f.stem for f in config.PROMPTS_DIR.glob('system_*.txt'))}"
        )
    text = template_path.read_text()

    # Replace static placeholders
    text = text.replace("{__CLASS_LIST__}", class_list)
    text = text.replace("{__CLASSES_STR__}", classes_str)
    text = text.replace("{__REAL_STR__}", real_str)

    # Handle host galaxy section (§5)
    if not has_cutout:
        # Remove §5 from "### 5. Host Galaxy" to next "### 6."
        sec5_start = text.find("### 5. Host Galaxy")
        sec6_start = text.find("### 6. Gaia Separation")
        if sec5_start >= 0 and sec6_start > sec5_start:
            text = text[:sec5_start] + text[sec6_start:]

    # Handle CoT section
    if cot:
        cot_path = config.PROMPTS_DIR / "cot.txt"
        if not cot_path.exists():
            raise FileNotFoundError(f"CoT template not found: {cot_path}")
        cot_text = cot_path.read_text()
        # Replace dynamic step number
        synthesis_step = "Step 7" if has_cutout else "Step 6"
        cot_text = cot_text.replace("{__SYNTHESIS_STEP__}", synthesis_step)
        cot_text = cot_text.replace("{__CLASS_LIST__}", class_list)
        cot_text = cot_text.replace("{__CLASSES_STR__}", classes_str)
        cot_text = cot_text.replace("{__REAL_STR__}", real_str)
        
        # Wrap with Response Format header
        cot_block = (
            "\n## Reasoning Protocol (Chain-of-Thought)\n"
            + cot_text
            + "\n\n## Response Format\n"
        )
    else:
        cot_block = (
            "\n## Response Format\n"
            "Output ONLY a JSON object (no markdown, no thinking process):\n"
        )

    text = text.replace("{__COT_SECTION__}", cot_block)

    return text

def build_prompt(target_id, few_shot, mode="text", cot=False):
    """Build the API prompt. Returns messages list.
    
    Args:
        target_id: source ID to classify.
        few_shot: list of (source_id, label) tuples (empty → zero-shot).
        mode: "text" or "multimodal".
        cot: if True, include Chain-of-Thought reasoning instructions.
    """
    target_has_cutout = has_image(target_id, "cutout.png")
    system_text = _make_system_prompt(cot=cot, has_cutout=target_has_cutout)

    user_content = []

    # Few-shot examples (skip if zero-shot)
    if few_shot:
        user_content.append({"type": "text", "text":
                             "## Few-Shot Examples\n"
                             "Study these labeled examples using the physical rules above.\n"})

        for fs_id, fs_label in few_shot:
            user_content.append({"type": "text", "text": f"### Example: {fs_label} ({fs_id})\n"})

            if mode == "multimodal":
                for img_name in config.MULTIMODAL_IMAGES:
                    if has_image(fs_id, img_name):
                        b64 = read_image_b64(fs_id, img_name)
                        if b64:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            })

            md_text = read_md(fs_id)
            user_content.append({"type": "text", "text": md_text + "\n"})

    # Target source
    target_header = "\n---\n\n"
    
    # Flux-scale note: color should use mag (scale-independent). Only decline rate
    # thresholds need scale adjustment since they're still in uJy/d.
    peak = _peak_flux(target_id)
    if peak and (peak > 500 or peak < 100):
        adj_decline = max(round(peak * 0.005), 1)
        adj_slow = max(round(peak * 0.0005), 0.1)
        target_header += (
            f"**Scale Note:** this source has peak={peak:.0f} uJy. "
            f"Color thresholds are in MAG (scale-independent, see system prompt). "
            f"For decline rate: steep > {adj_decline} uJy/d, slow < {adj_slow} uJy/d.\n\n"
        )


    if cot:
        target_header += (
            "## Reasoning Task\n"
            "Reason through each physical discriminator step by step "
            "(Step 1: Shape Gate → Step 2: Optical Color → "
            "Step 3: WISE → Step 4: Joint Matrix → Step 5: Tiebreakers),\n"
            "then output the classification as JSON.\n\n"
        )

    target_header += "## Target: Classify This Source\n\n"
    user_content.append({"type": "text", "text": target_header})

    if mode == "multimodal":
        for img_name in config.MULTIMODAL_IMAGES:
            if has_image(target_id, img_name):
                b64 = read_image_b64(target_id, img_name)
                if b64:
                    label = img_name.replace(".png", "").replace("_", " ")
                    user_content.append({"type": "text", "text": f"[Image: {label}]\n"})
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

    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user_content}]


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
                max_tokens=6000,
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
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            text = text[start:].strip()  # truncated block, try as-is
    elif "```" in text:
        start = text.index("```") + 3
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            # Single ``` → check if it's a trailing closer
            # If text before ``` is valid JSON, use that
            before = text[:text.index("```")].strip()
            if before.endswith("}") or before.endswith("]"):
                text = before
            else:
                text = text[start:].strip()
    # Strip trailing ``` if present (LLM sometimes adds closing backticks without opening)
    if text.endswith("```"):
        text = text[:-3].strip()
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


def _extract_reasoning(raw_text, parsed):
    """Extract CoT reasoning text by removing the JSON part from raw response."""
    if not raw_text:
        return ""
    # Try to find where the JSON starts and take text before it
    json_str = json.dumps(parsed, ensure_ascii=False)
    # Match by first 60 chars of JSON (robust against whitespace variations)
    json_head = json_str[:min(60, len(json_str))]
    idx = raw_text.find(json_head)
    if idx >= 0:
        return raw_text[:idx].strip()
    # Fallback: everything before the first {
    for i, ch in enumerate(raw_text):
        if ch == '{':
            return raw_text[:i].strip()
    return raw_text.strip()


# ═══════════════════════════════════════════════════
# Result saving
# ═══════════════════════════════════════════════════

def save_result(source_id, parsed, raw_response, usage, mode, model, few_shot, cot=False):
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
        "cot": cot,
        "cot_reasoning": _extract_reasoning(raw_response, parsed),
        "_raw_response": raw_response,
    }
    out_path = config.RESULTS_DIR / f"{source_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ═══════════════════════════════════════════════════
# Classify one source
# ═══════════════════════════════════════════════════

def classify_one(source_id, mode="text", n_shot=None, model=None, force=False, cot=False,
                exemplar_set=None):
    """Classify a single source.
    
    Args:
        cot: if True, enable Chain-of-Thought reasoning.
        exemplar_set: name of curated exemplar set (e.g. "boundary", "textbook").
                      Loads templates/fewshot_{exemplar_set}.json.
    """
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

    few_shot = sample_few_shot(idx, n_per_class=n_shot, exclude={source_id},
                               exemplar_set=exemplar_set)
    if n_shot > 0 and not few_shot:
        print(f"  [error] {source_id} -- no labeled examples available in index.json")
        return None

    # Build prompt
    mode_str = f"{mode}, {len(few_shot)}-shot"
    if cot:
        mode_str += ", CoT"
    print(f"  {source_id}: building prompt ({mode_str})")
    messages = build_prompt(source_id, few_shot, mode, cot=cot)

    # Call API
    raw_text, usage = call_api(messages, model=model)

    # Parse
    parsed = parse_response(raw_text)

    # Save
    result = save_result(source_id, parsed, raw_text, usage, mode,
                         model or config.MODEL, few_shot, cot=cot)

    # Print summary
    cls_info = result.get("classification", {})
    label = cls_info.get("label", "?")
    conf = cls_info.get("confidence", "?")
    score = cls_info.get("score", 0)
    print(f"  [done] {source_id} -> {label} ({conf}, score={score:.2f}) "
          f"[{result['tokens']['total']} tokens]")

    return result


def classify_all_unlabeled(mode="text", n_shot=None, model=None, force=False, cot=False,
                          exemplar_set=None):
    """Classify all sources labeled 'unknown' in index.json."""
    idx = promt_load_index()
    unlabeled = [sid for sid, info in idx.items() if info["label"] == "unknown"]
    if not unlabeled:
        print("No unlabeled sources found in index.json")
        return
    print(f"Classifying {len(unlabeled)} unlabeled sources...")
    done = 0
    for sid in sorted(unlabeled):
        result = classify_one(sid, mode=mode, n_shot=n_shot, model=model, force=force, cot=cot,
                              exemplar_set=exemplar_set)
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
    parser.add_argument("--cot", action="store_true", help="enable Chain-of-Thought reasoning")
    parser.add_argument("--exemplar-set", help="curated exemplar set name (loads templates/fewshot_NAME.json)")
    args = parser.parse_args()

    if args.all_unlabeled:
        classify_all_unlabeled(mode=args.mode, n_shot=args.n_shot, model=args.model,
                               force=args.force, cot=args.cot,
                               exemplar_set=args.exemplar_set)
    elif args.results and args.source_id:
        show_result(args.source_id)
    elif args.source_id:
        classify_one(args.source_id, mode=args.mode, n_shot=args.n_shot, model=args.model,
                     force=args.force, cot=args.cot,
                     exemplar_set=args.exemplar_set)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()