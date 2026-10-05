#!/usr/bin/env bash
# Enhance the whole ruod640 dataset with SCNet (GPU 0) and Semi-UIR (GPU 1) in parallel.
# Usage (Kaggle, after setup.sh, with dataset ruod640 added as input):
#   bash enhancers/run_ruod.sh [WORK=/kaggle/working] [OUT=$WORK/enhanced]
# Safe to re-run: images already enhanced are skipped.
set -euo pipefail
WORK=${1:-/kaggle/working}
OUT=${2:-$WORK/enhanced}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(dirname "$(find /kaggle/input -name data_orig.yaml | head -1)") # ruod640 root
[ -d "$D/orig/images" ] || { echo "ruod640 not found under /kaggle/input"; exit 1; }
echo "Input: $D"
echo "Output: $OUT"

NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
G1=$(( NGPU > 1 ? 1 : 0 )) # second GPU if available, otherwise share GPU 0

( for s in train test; do
    CUDA_VISIBLE_DEVICES=0 python "$HERE/enhance_scnet.py" --repo "$WORK/computer-vision-SCNet" \
      --input "$D/orig/images/$s" --output "$OUT/scnet/images/$s"
  done ) > "$WORK/scnet.log" 2>&1 &
P1=$!
( for s in train test; do
    CUDA_VISIBLE_DEVICES=$G1 python "$HERE/enhance_semiuir.py" --repo "$WORK/computer-vision-Semi-UIR" \
      --input "$D/orig/images/$s" --output "$OUT/semiuir/images/$s"
  done ) > "$WORK/semiuir.log" 2>&1 &
P2=$!
wait $P1 || { echo "SCNet failed, see $WORK/scnet.log"; tail -20 "$WORK/scnet.log"; exit 1; }
wait $P2 || { echo "Semi-UIR failed, see $WORK/semiuir.log"; tail -20 "$WORK/semiuir.log"; exit 1; }

# Same YOLO labels for every version (Ultralytics finds labels by replacing images -> labels in the path)
for v in scnet semiuir; do
  mkdir -p "$OUT/$v/labels" && cp -r "$D/orig/labels/." "$OUT/$v/labels/"
done

for s in train test; do
  python "$HERE/check_outputs.py" --orig "$D/orig/images/$s" --out "$OUT/scnet/images/$s" "$OUT/semiuir/images/$s"
done
du -sh "$OUT"/*
