#!/usr/bin/env bash
# Setup Semi-UIR + SCNet.
set -euo pipefail
WORK=${1:-/kaggle/working}
GH=https://github.com/bao-nguyen-quoc

# 1. Check if GPU is enabled.
python - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    sys.exit('Kaggle GPU is not enabled. Enable GPU in Settings -> Accelerator.')
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))
EOF

# 2. Clone repositories.
for r in computer-vision-Semi-UIR computer-vision-SCNet; do
  [ -d "$WORK/$r" ] || git clone -q "$GH/$r" "$WORK/$r"
done

# 3. Install dependencies.
pip install -q -r "$WORK/computer-vision-Semi-UIR/requirements.txt" gdown

# 4. Download SCNet model weights (model weights is too heavy so it is not included in the repository).
W="$WORK/computer-vision-SCNet/weights"; mkdir -p "$W"
[ -s "$W/scnet.pth" ] || gdown 1kHOPSUObw7FafI6_xgaD9kCZ5U5JzIsc -O "$W/scnet.pth"
echo "eedbdd5f851876d77e24b5048ccf790f93bcca4f1f48d99efa85187f0eb76ff5  $W/scnet.pth" | sha256sum -c -
echo "Setup OK"