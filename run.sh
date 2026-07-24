#!/bin/bash
# run.sh - ZTF_prompt full pipeline
# Usage:
#   bash run.sh                        # multimodal, default exemplar set
#   bash run.sh --set boundary         # boundary exemplar set
#   bash run.sh --mode text --set text # text mode, text exemplar set
#   bash run.sh --skip-plot            # skip plot generation
#   bash run.sh --summary-only         # only run summary (no classification)

set -e
cd "$(dirname "$0")"

MODE="multimodal"
EXEMPLAR_SET=""
SKIP_PLOT=false
SUMMARY_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --set)  EXEMPLAR_SET="$2"; shift 2 ;;
        --skip-plot) SKIP_PLOT=true; shift ;;
        --summary-only) SUMMARY_ONLY=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════"
echo "  ZTF_prompt Pipeline"
echo "  Mode: $MODE"
echo "  Exemplar set: ${EXEMPLAR_SET:-default}"
echo "═══════════════════════════════════════"
echo ""

# ── Step 1: Generate lightcurve plots ──
if ! $SKIP_PLOT && ! $SUMMARY_ONLY; then
    echo "── Step 1: Generating lightcurve plots ──"
    python plot.py --all 2>&1 | tail -3
    echo ""
fi

# ── Step 2: Classify all unknowns ──
if ! $SUMMARY_ONLY; then
    echo "── Step 2: Classifying unknown sources ──"
    CMD="python classify.py --all-unlabeled --mode $MODE"
    if [ -n "$EXEMPLAR_SET" ]; then
        CMD="$CMD --exemplar-set $EXEMPLAR_SET"
    fi
    if [ "$MODE" = "multimodal" ]; then
        CMD="$CMD --model qwen3.6-chat"
    fi
    echo "  $CMD"
    echo ""
    $CMD
    echo ""
fi

# ── Step 3: Summary + plot ──
echo "── Step 3: Summarizing results ──"
CMD="python summary.py --plot"
if [ -n "$EXEMPLAR_SET" ]; then
    CMD="$CMD --exemplar-set $EXEMPLAR_SET"
fi
if ! $SUMMARY_ONLY || [ "$MODE" != "multimodal" ]; then
    CMD="$CMD --mode $MODE"
fi
echo "  $CMD"
echo ""
$CMD

echo ""
echo "Done. Check summary/ for results."