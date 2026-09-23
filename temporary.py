"""
temporary.py — Temporary classification labels layered on top of the base pipeline.

Design intent
=============
temporary_* labels are OPTIONAL ADDITIONAL categories layered on the MAIN
classification (which stays mutually-exclusive single-label: TDE/SN/AGN/Others/
Unsure). Each temporary label uses a POSITIVE criterion, so a source may match
zero, one, or several temporary labels at once — independently of its main class
(a TDE can also be a "temporary_OVI", for example).

The temporary layer is storage-driven and additive:
  * No temporary config for a collection  ->  apply_temporary() returns inputs
    unchanged, so the pipeline degrades 100% to the base version.
  * Temporary labels are managed (add/remove/edit) through
    temporary_registry.json (an array — same registry pattern as TDEweb's
    policy_registry.json, so multiple labels are naturally supported and a
    label is removed by deleting its object).

Layout
======
  temporary_registry.json   — the registry (array of label objects)
  temporary.py              — this module (functions used by CLI and TDEweb)

All temp labels carry a "temporary_" prefix to keep them distinct from the
fixed base classes.

Author: CyanWhite
"""

import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "temporary_registry.json"
TEMP_PREFIX = "temporary_"

# ---------------------------------------------------------------------------
# Registry I/O (mirrors TDEweb's policy_registry.json pattern)
# ---------------------------------------------------------------------------


def _load_registry():
    """Load the temporary-label registry.

    Returns a dict {label_id: {...}} (normalised from the on-disk array),
    or {} if the file is missing / empty.
    """
    if not REGISTRY_PATH.is_file():
        return {}
    try:
        raw = json.loads(REGISTRY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Accept both list [{id:...}] and dict {id:{...}} forms
    if isinstance(raw, list):
        return {e.get("id"): e for e in raw if e.get("id")}
    if isinstance(raw, dict):
        return raw
    return {}


def _save_registry(reg):
    """Write the registry back to disk as an array, with a .bak backup."""
    arr = list(reg.values())
    if REGISTRY_PATH.is_file():
        try:
            shutil.copy2(REGISTRY_PATH, str(REGISTRY_PATH) + ".bak")
        except OSError:
            pass
    REGISTRY_PATH.write_text(json.dumps(arr, indent=2, ensure_ascii=False))
    return arr


# ---------------------------------------------------------------------------
# Query / mutate
# ---------------------------------------------------------------------------


def _normalise_id(label_id):
    """Normalise a label id to the temporary_ prefix convention."""
    label_id = label_id.strip()
    if not label_id.startswith(TEMP_PREFIX):
        label_id = TEMP_PREFIX + label_id
    return label_id


def list_temporary_labels(collection=None):
    """Return the list of temporary labels.

    If `collection` is given, only labels bound to it (or bound to nothing =
    global) are returned.
    """
    reg = _load_registry()
    labels = []
    for entry in reg.values():
        if collection is not None:
            colls = entry.get("collections") or []
            if colls and collection not in colls:
                continue  # label is scoped to other collections
        labels.append(entry)
    return labels


def add_temporary_label(label_id, name="", positive_criteria="",
                        fewshot=None, collections=None, overwrite=False):
    """Add a temporary label to the registry. Raises if id already exists
    unless overwrite=True."""
    label_id = _normalise_id(label_id)
    reg = _load_registry()
    if label_id in reg and not overwrite:
        raise ValueError(f"temporary label '{label_id}' already exists "
                         f"(use overwrite=True to replace)")
    reg[label_id] = {
        "id": label_id,
        "name": name or label_id,
        "positive_criteria": positive_criteria,
        "fewshot": list(fewshot or []),
        "collections": list(collections or []),
    }
    _save_registry(reg)
    return reg[label_id]


def remove_temporary_label(label_id):
    """Remove a temporary label. Returns True if it was removed."""
    label_id = _normalise_id(label_id)
    reg = _load_registry()
    if label_id not in reg:
        return False
    del reg[label_id]
    _save_registry(reg)
    return True


def update_temporary_label(label_id, name=None, positive_criteria=None,
                           fewshot=None, collections=None):
    """Update fields of an existing temporary label (id is immutable).

    Only the fields passed as non-None are updated; omitted fields keep their
    current values. The label id (the registry key, also referenced by saved
    classification results) is intentionally NOT changeable.
    """
    label_id = _normalise_id(label_id)
    reg = _load_registry()
    if label_id not in reg:
        raise ValueError(f"temporary label '{label_id}' does not exist")
    entry = reg[label_id]
    if name is not None:
        entry["name"] = name
    if positive_criteria is not None:
        entry["positive_criteria"] = positive_criteria
    if fewshot is not None:
        entry["fewshot"] = list(fewshot)
    if collections is not None:
        entry["collections"] = list(collections)
    _save_registry(reg)
    return entry


# ---------------------------------------------------------------------------
# The layering core (used by classify_pipeline / CLI / TDEweb)
# ---------------------------------------------------------------------------


def build_system_section(collection=None):
    """Return the temporary-classes system-prompt section string.

    Empty string if no temporary labels apply to `collection` — callers can
    then skip appending entirely (graceful fallback to the base prompt).
    """
    labels = list_temporary_labels(collection)
    if not labels:
        return ""
    lines = [
        "\n\n## Temporary Classes (independent, positive criteria)",
        "These are OPTIONAL ADDITIONAL labels — they DO NOT replace the base "
        "classification above. Independently check each criterion and list ALL "
        "matching labels in the output field \"temporary_label\" (an empty list "
        "if none match). A source may match multiple temporary labels "
        "simultaneously, including on top of its base class.",
    ]
    for lb in labels:
        lines.append(f"- {lb['id']}: POSITIVE — {lb['positive_criteria'] or 'n/a'}")
    lines.append("Output format: \"temporary_label\": [\"temporary_xxx\", ...]")
    return "\n".join(lines)


def build_temporary_few_shot(collection=None, exclude=None):
    """Return [(source_id, temporary_label_id), ...] for `collection`."""
    exclude = exclude or set()
    out = []
    for lb in list_temporary_labels(collection):
        for fs_id in (lb.get("fewshot") or []):
            if fs_id in exclude:
                continue
            out.append((fs_id, lb["id"]))
    return out


def apply_temporary(system_text, few_shot, collection=None):
    """Layer temporary labels onto the base system prompt + few-shot.

    Args:
        system_text: base system prompt str (already version-rendered).
        few_shot:    base few-shot list of (source_id, label) tuples.
        collection:  optional remote WFST collection name to scope labels to.

    Returns:
        (new_system_text, new_few_shot). If there are no temporary labels for
        the given collection, returns (system_text, few_shot) UNCHANGED — i.e.
        the pipeline degrades to the base version.
    """
    section = build_system_section(collection)
    temp_few = build_temporary_few_shot(collection,
                                        exclude={fs[0] for fs in few_shot})
    if not section and not temp_few:
        return system_text, few_shot
    new_system = system_text + section if section else system_text
    new_few_shot = list(few_shot) + temp_few if temp_few else list(few_shot)
    return new_system, new_few_shot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Manage temporary labels")
    sub = ap.add_subparsers(dest="action")

    p_list = sub.add_parser("list")
    p_list.add_argument("--collection", default=None)

    p_add = sub.add_parser("add")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--name", default="")
    p_add.add_argument("--criteria", default="")
    p_add.add_argument("--fewshot", nargs="*", default=[])
    p_add.add_argument("--collections", nargs="*", default=[])
    p_add.add_argument("--overwrite", action="store_true")

    p_rm = sub.add_parser("remove")
    p_rm.add_argument("--id", required=True)

    args = ap.parse_args()

    if args.action == "list":
        for lb in list_temporary_labels(args.collection):
            print(f"- {lb['id']}: {lb['name']} | collections={lb['collections']}")
        exit(0)
    elif args.action == "add":
        e = add_temporary_label(args.id, args.name, args.criteria,
                                args.fewshot, args.collections, args.overwrite)
        print(f"added: {e['id']}")
        exit(0)
    elif args.action == "remove":
        ok = remove_temporary_label(args.id)
        print(f"removed: {ok}")
        exit(0)
    else:
        ap.print_help()
