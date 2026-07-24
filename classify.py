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


def _make_system_prompt(cot=False, has_cutout=False):
    """Build system prompt dynamically from config.CLASSES.
    
    Args:
        cot: If True, include Chain-of-Thought reasoning instructions.
        has_cutout: If True, include host galaxy cutout guidance.
    """
    classes_str = "|".join(config.CLASSES)
    class_list = ", ".join(config.CLASSES)
    # For indicator direction: only real physical classes (not Unsure)
    real_classes = [c for c in config.CLASSES if c != "Unsure"]
    real_str = "|".join(real_classes)

    base = (
        f"You are an astronomical transient classifier specializing in ZTF light curves. "
        f"Classify each source as one of: {class_list}.\n\n"
        "## Physical Discriminators (check in priority order)\n\n"
        "### 1. Color Evolution (g − r) — STRONGEST signal\n"
        "- delta_g-r > 10 uJy (strong red→blue evolution): STRONG TDE indicator\n"
        "- delta_g-r < 5 uJy (flat/mild evolution): typical SN\n"
        "- Color staying RED throughout: favors steady sources, flag as uncertain\n"
        "- Caveat: some SN subtypes (IIb, IIn, SLSN-I) can also show red→blue\n"
        "  color evolution, though less common. If color is the ONLY TDE signal\n"
        "  and rise shape / duration point to SN, consider SN with medium confidence.\n\n"
        "### 2. Rise Morphology — SHAPE matters more than duration\n"
        "\n"
        "Concave rise (decelerating, d²F/dt² < 0):\n"
        "  → STRONG TDE indicator\n"
        "  Physical: fallback accretion — most bound debris returns first,\n"
        "  giving rapid initial flux increase that decelerates toward peak.\n"
        "  Look for: steepest slope in early rise, flattening near peak.\n"
        "  Hint: in per-phase Phase column (Section 2.3),\n"
        "  negative early → approaching zero = concave (TDE-like).\n"
        "\n"
        "Convex rise (accelerating, d²F/dt² > 0):\n"
        "  → favors SN\n"
        "  Physical: shock cooling — post-breakout envelope is UV-hot,\n"
        "  optical flux is initially dim and accelerates as temperature drops\n"
        "  into the optical band.\n"
        "  Look for: slope increasing through the rise, steepest just before peak.\n"
        "\n"
        "Rise duration as secondary check:\n"
        "  < 10 days  → favors SN (shock-cooling timescale)\n"
        "  10–60 days → typical TDE (fallback timescale)\n"
        "  > 60 days  → check shape: concave still possible TDE; convex = unusual\n"
        "\n"
        "Total span:\n"
        "  < 200 days  → favors TDE (t^(-5/3) decay ~months)\n"
        "  > 1000 days → favors SN (Ni decay tail can last years)\n"
        "\n"
        "### 3. Decline Shape\n"
        "- Steep power-law decline (>1 uJy/d sustained): favors TDE\n"
        "- Plateau or very slow decline (<0.1 uJy/d): favors SN\n\n"
        "### 4. Data Quality\n"
        "- Total points < 10: inherently LOW confidence\n"
        "- Single-band only: no color information, be cautious\n"
        "- If quality is poor and signals are ambiguous, prefer medium/low confidence\n\n" +
        (
        "### 5. Host Galaxy Context (from cutout image)\n"
        "The SDSS cutout shows the host galaxy. Use as a SECONDARY check:\n"
        "- Red, smooth, elliptical host → favors TDE (TDEs prefer quiescent galaxies)\n"
        "- Blue, spiral, or irregular host → weakly favors SN (star-forming hosts)\n"
        "- Source centered on galaxy nucleus → favors TDE (nuclear events)\n"
        "- Source clearly in disk/arm → favors SN (traces star formation)\n"
        "- No obvious host (faint/blended): this indicator is UNINFORMATIVE, skip it\n"
        "IMPORTANT: host morphology is a WEAK prior (weight ≤ 0.2). NEVER override\n"
        "strong color/rise signals based on host alone.\n\n"
        if has_cutout else ""
        ) +
        "## Decision Logic\n"
        "1. Check color evolution FIRST. If clear, follow it.\n"
        "2. Check rise SHAPE: concave = TDE; convex = SN.\n"
        "   Use rise duration + total span as secondary tiebreakers.\n"
        "3. If conflicting signals: flag as medium confidence and explain why.\n"
        "4. WHEN TO USE 'Unsure': if 2+ physical indicators point in opposite\n"
        "   directions (e.g. color=TDE but rise shape=SN), classify as 'Unsure'.\n"
        "   Set 'unsure_preference' to the class you slightly favor.\n"
        "   Unsure is for genuinely torn cases — prefer committing if any\n"
        "   clear signal exists.\n" +
        (
        "5. Host galaxy (if cutout available): use as weak tiebreaker ONLY.\n"
        "   Never let host morphology override clear light curve signals.\n"
        if has_cutout else ""
        )
    )

    if cot:
        cot_section = (
            "\n## Reasoning Protocol (Chain-of-Thought)\n"
            "Before outputting the JSON, reason through each step in plain text.\n"
            "Label each step clearly with 'Step 1:', 'Step 2:', etc.\n\n"
            "Step 1 — Color Evolution: extract delta_g-r from Section 2.2. "
            "Compare against thresholds (>10 = TDE, <5 = SN). State your assessment.\n"
            "Step 2 — Timescale: extract rise time and total span from Section 1. "
            "Compare against thresholds (<60d rise = TDE, <200d span = TDE).\n"
            "Step 3 — Decline Shape: extract decline rate from Section 1. "
            "Compare (>1 uJy/d = TDE, <0.1 = SN).\n" +
            (
            "Step 4 — Host Galaxy (from cutout): describe the host. "
            "Red/smooth/elliptical → favors TDE. Blue/spiral → favors SN. "
            "No obvious host → skip this step.\n"
            if has_cutout else ""
            ) +
            "Step " + ("5" if has_cutout else "4") + " — Synthesis: weigh all indicators. If they conflict, "
            "use 'Unsure' with unsure_preference (e.g. Unsure→TDE). "
            "State final classification with confidence level.\n\n"
            "Then output the JSON on a new line after your reasoning.\n\n"
            "## Response Format\n"
        )
    else:
        cot_section = (
            "\n## Response Format\n"
            "Output ONLY a JSON object (no markdown, no thinking process):\n"
        )

    json_template = (
        f'{{"classification":{{"label":"{classes_str}","confidence":"high|medium|low",'
        f'"score":0.0-1.0,"unsure_preference":"{real_str}|null"}},'
        f'"reasoning":{{"primary_signal":"...","indicators":[{{'
        f'"name":"...","value":"...","weight":0.0-1.0,"direction":"{real_str}"}}]}},'
        '"quality":{"flags":[]}}'
    )

    return base + cot_section + json_template


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
    
    # Flux-scale threshold adjustment for bright sources (e.g. WFST vs ZTF)
    peak = _peak_flux(target_id)
    if peak and peak > 500:
        adj_color = max(round(peak * 0.05), 10)
        adj_decline = max(round(peak * 0.005), 1)
        adj_flat = max(round(peak * 0.02), 5)
        adj_slow = max(round(peak * 0.0005), 0.1)
        target_header += (
            f"**Flux Scale Note:** this source is bright (peak={peak:.0f} uJy). "
            f"Use RELATIVE thresholds:\n"
            f"- Significant color evolution: delta_g-r > {adj_color} uJy "
            f"({adj_color/peak*100:.0f}% of peak)\n"
            f"- Flat color: delta_g-r < {adj_flat} uJy "
            f"({adj_flat/peak*100:.0f}% of peak)\n"
            f"- Significant decline: > {adj_decline} uJy/d "
            f"({adj_decline/peak*100:.1f}%/d of peak)\n"
            f"- Slow decline: < {adj_slow} uJy/d\n\n"
        )
    

    if cot:
        target_header += (
            "## Reasoning Task\n"
            "Reason through each physical discriminator step by step "
            "(Step 1: Color Evolution → Step 2: Timescale → "
            "Step 3: Decline Shape → Step 4: Synthesis),\n"
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
                max_tokens=3000,
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
            text = text[start:].strip()
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

def classify_one(source_id, mode="text", n_shot=None, model=None, force=False, cot=False):
    """Classify a single source.
    
    Args:
        cot: if True, enable Chain-of-Thought reasoning.
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

    few_shot = sample_few_shot(idx, n_per_class=n_shot, exclude={source_id})
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


def classify_all_unlabeled(mode="text", n_shot=None, model=None, force=False, cot=False):
    """Classify all sources labeled 'unknown' in index.json."""
    idx = promt_load_index()
    unlabeled = [sid for sid, info in idx.items() if info["label"] == "unknown"]
    if not unlabeled:
        print("No unlabeled sources found in index.json")
        return
    print(f"Classifying {len(unlabeled)} unlabeled sources...")
    done = 0
    for sid in sorted(unlabeled):
        result = classify_one(sid, mode=mode, n_shot=n_shot, model=model, force=force, cot=cot)
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
    args = parser.parse_args()

    if args.all_unlabeled:
        classify_all_unlabeled(mode=args.mode, n_shot=args.n_shot, model=args.model,
                               force=args.force, cot=args.cot)
    elif args.results and args.source_id:
        show_result(args.source_id)
    elif args.source_id:
        classify_one(args.source_id, mode=args.mode, n_shot=args.n_shot, model=args.model,
                     force=args.force, cot=args.cot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()