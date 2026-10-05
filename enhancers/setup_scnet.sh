#!/usr/bin/env bash
# enhancers/setup_scnet.sh
set -e
REPO=${1:-/kaggle/working/computer-vision-SCNet}
mkdir -p "$REPO/weights"
if [ ! -s "$REPO/weights/scnet.pth" ]; then
  pip install -q gdown
  gdown 1kHOPSUObw7FafI6_xgaD9kCZ5U5JzIsc -O "$REPO/weights/scnet.pth"
fi
echo "eedbdd5f851876d77e24b5048ccf790f93bcca4f1f48d99efa85187f0eb76ff5  $REPO/weights/scnet.pth" | sha256sum -c -