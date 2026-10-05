# enhancers/enhance_scnet.py
import argparse, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--repo', default='computer-vision-SCNet') # fork repository
p.add_argument('--weights', default=None) # default weights path
p.add_argument('--input', required=True) # input image or directory
p.add_argument('--output', required=True)
args = p.parse_args()

sys.path.append(args.repo)
from net import Decoder # only use enhance branch, remove VGG16 (only serve loss when training)
from utils import quantize

device = torch.device('cuda:0')
ckpt = torch.load(args.weights or f'{args.repo}/weights/scnet.pth', map_location='cpu')
state = {}
for k, v in ckpt.items():
    k = k.replace('module.', '', 1)
    if k.startswith('decoder.'):
        state[k[len('decoder.'):]] = v
net = Decoder(device=device).to(device).eval()
net.load_state_dict(state) # strict: report error if missing/extra key

@torch.no_grad()
def enhance(img):
    w, h = img.size
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    ph, pw = (-h) % 8, (-w) % 8 # 3 times MaxPool2d(2) -> must divide by 8
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    out = quantize(net(x), 1) # rgb_range=1 like eval.py
    y = (out[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
    return Image.fromarray(y)

exts = ('.png', '.jpg', '.jpeg')
files = [args.input] if os.path.isfile(args.input) else \
        [os.path.join(args.input, n) for n in sorted(os.listdir(args.input)) if n.lower().endswith(exts)]
os.makedirs(args.output, exist_ok=True)
for f in files:
    out_path = os.path.join(args.output, os.path.basename(f))
    if os.path.exists(out_path): # skip images already enhanced (safe to re-run after an interruption)
        continue
    t = time.time()
    # JPEG quality 95 to match ruod640/orig (Pillow default 75 would add extra compression artifacts)
    img = enhance(Image.open(f).convert('RGB'))
    tmp = out_path + '.part' # write then rename: an interrupted save never leaves a half-written image
    img.save(tmp, format=Image.registered_extensions()[os.path.splitext(out_path)[1].lower()], quality=95)
    os.replace(tmp, out_path)
    parent_dir = os.path.basename(os.path.dirname(f))
    file_name = os.path.basename(f)
    print(f'{parent_dir}/{file_name}: {time.time() - t:.2f}s')