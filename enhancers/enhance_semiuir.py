# enhancers/enhance_semiuir.py
import argparse, os, sys, time
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--repo', default='computer-vision-Semi-UIR')   # thư mục fork
p.add_argument('--input', required=True)                       # 1 ảnh hoặc 1 thư mục ảnh
p.add_argument('--output', required=True)                      # thư mục xuất
args = p.parse_args()

sys.path.append(args.repo)
from model import AIMnet

def luminance_estimation(img):
    img = np.uint8(np.array(img))
    L = np.ones_like(img).astype(np.float32)
    for sigma in [15, 60, 90]:
        L += np.clip(np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8), 0, 255)
    L = L / 3
    L = (L - L.min()) / (L.max() - L.min() + 1e-6)
    return np.uint8(L * 255)

ckpt = torch.load(f'{args.repo}/pretrained/model.pth', map_location='cuda', weights_only=False)
state = {k.replace('module.', '', 1): v for k, v in ckpt['state_dict'].items()}
net = AIMnet().cuda().eval()
net.load_state_dict(state)
to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32) / 255.).permute(2, 0, 1)[None].cuda()

def enhance(img):
    w, h = img.size
    x, la = to_t(img), to_t(luminance_estimation(img))
    ph, pw = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    la = F.pad(la, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        out, _ = net(x, la)
    y = (out[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(y)

exts = ('.png', '.jpg', '.jpeg')
files = [args.input] if os.path.isfile(args.input) else \
        [os.path.join(args.input, n) for n in sorted(os.listdir(args.input)) if n.lower().endswith(exts)]
os.makedirs(args.output, exist_ok=True)
for f in files:
    t = time.time()
    enhance(Image.open(f).convert('RGB')).save(os.path.join(args.output, os.path.basename(f)))
    parent_dir = os.path.basename(os.path.dirname(f))
    file_name = os.path.basename(f)
    print(f'{parent_dir}/{file_name}: {time.time() - t:.2f}s')