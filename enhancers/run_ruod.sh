#!/usr/bin/env bash
# Enhance ruod640/orig with SCNet and Semi-UIR (the two run in parallel).
# Usage (Kaggle, after setup.sh, with dataset ruod640 added as input):
#   [SPLITS="train test"] [ENHANCERS="scnet semiuir"] bash enhancers/run_ruod.sh [WORK=/kaggle/working] [OUT=$WORK/enhanced]
# Examples:
#   SPLITS=train bash enhancers/run_ruod.sh
#   SPLITS=test  bash enhancers/run_ruod.sh
#   ENHANCERS=scnet bash enhancers/run_ruod.sh
# Safe to re-run: images already enhanced are skipped (resume after a session timeout).
set -euo pipefail
WORK=${1:-/kaggle/working}
OUT=${2:-$WORK/enhanced}
SPLITS=${SPLITS:-"train test"}
ENHANCERS=${ENHANCERS:-"scnet semiuir"}
INPUT_ROOT=${INPUT_ROOT:-/kaggle/input}
HERE=$(cd "$(dirname "$0")" && pwd)

# Locate the ruod640 root by its orig/images/train folder (does not depend on data_orig.yaml being present)
TRAIN_DIR=$(find "$INPUT_ROOT" -type d -path '*/orig/images/train' -print -quit)
[ -n "$TRAIN_DIR" ] || { echo "orig/images/train not found under $INPUT_ROOT"; exit 1; }
D=$(dirname "$(dirname "$(dirname "$TRAIN_DIR")")") # .../orig/images/train -> ruod640 root
echo "Input:  $D"
echo "Output: $OUT"
echo "Splits: $SPLITS | Enhancers: $ENHANCERS"
for s in $SPLITS; do echo "  $s: $(ls "$D/orig/images/$s" | wc -l) images"; done

NGPU=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)
G1=$(( NGPU > 1 ? 1 : 0 )) # second GPU if available, otherwise both enhancers share GPU 0

P1=""; P2=""
case " $ENHANCERS " in *" scnet "*)
  ( for s in $SPLITS; do
      CUDA_VISIBLE_DEVICES=0 python "$HERE/enhance_scnet.py" --repo "$WORK/computer-vision-SCNet" \
        --input "$D/orig/images/$s" --output "$OUT/scnet/images/$s"
    done ) > "$WORK/scnet.log" 2>&1 &
  P1=$! ;;
esac
case " $ENHANCERS " in *" semiuir "*)
  ( for s in $SPLITS; do
      CUDA_VISIBLE_DEVICES=$G1 python "$HERE/enhance_semiuir.py" --repo "$WORK/computer-vision-Semi-UIR" \
        --input "$D/orig/images/$s" --output "$OUT/semiuir/images/$s"
    done ) > "$WORK/semiuir.log" 2>&1 &
  P2=$! ;;
esac

rc=0 # wait for both before judging, so a failure in one never leaves the other orphaned
if [ -n "$P1" ]; then wait "$P1" || { echo "SCNet failed, see $WORK/scnet.log"; tail -20 "$WORK/scnet.log"; rc=1; }; fi
if [ -n "$P2" ]; then wait "$P2" || { echo "Semi-UIR failed, see $WORK/semiuir.log"; tail -20 "$WORK/semiuir.log"; rc=1; }; fi
[ "$rc" -eq 0 ] || exit 1

# Same YOLO labels for every version (Ultralytics finds labels by replacing images -> labels in the path)
for v in $ENHANCERS; do
  mkdir -p "$OUT/$v/labels" && cp -r "$D/orig/labels/." "$OUT/$v/labels/"
  find "$OUT/$v" -name '*.part' -delete # leftovers of an interrupted save (only this run's enhancers, so a
done                                    # run for another enhancer in parallel is never disturbed)

for s in $SPLITS; do
  outs=(); for v in $ENHANCERS; do outs+=("$OUT/$v/images/$s"); done
  CL="$WORK/check_${ENHANCERS// /_}_$s.log" # one log per enhancer set, so parallel runs do not overwrite each other
  python "$HERE/check_outputs.py" --orig "$D/orig/images/$s" --out "${outs[@]}" | tee "$CL" | tail -n 8
  grep -q "ALL OK" "$CL" || { echo "check_outputs failed for split $s, see $CL"; exit 1; }
done
du -sh "$OUT"/*
