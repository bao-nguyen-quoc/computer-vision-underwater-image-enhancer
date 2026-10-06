# PSNR/SSIM on test dataset of UIED.

import argparse, csv, os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
 
p = argparse.ArgumentParser()
p.add_argument('--ref', required=True)                 # thư mục label (ảnh tham chiếu)
p.add_argument('--pred', nargs='+', required=True)     # danh sách tên=thư_mục
p.add_argument('--csv', default='uieb_metrics.csv')    # kết quả từng ảnh
args = p.parse_args()
 
exts = ('.png', '.jpg', '.jpeg', '.bmp')
index = lambda d: {os.path.splitext(n)[0]: os.path.join(d, n)
                   for n in sorted(os.listdir(d)) if n.lower().endswith(exts)}   # ghép theo tên không đuôi
ref = index(args.ref)
load = lambda path: np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
 
rows, summary = [], {}
for spec in args.pred:
    name, d = spec.split('=', 1)
    pred = index(d)
    missing = sorted(set(ref) - set(pred))
    if missing:
        raise SystemExit(f'[{name}] missing {len(missing)} images, e.g: {missing[:3]}')
    ps, ss = [], []
    for stem in sorted(ref):
        a, b = load(ref[stem]), load(pred[stem])
        # No implicit resize: size mismatch is an error to be known
        if a.shape != b.shape:
            raise SystemExit(f'[{name}] {stem}: ref {a.shape} != pred {b.shape}')
        ps.append(peak_signal_noise_ratio(a, b, data_range=255))
        ss.append(structural_similarity(a, b, channel_axis=2, data_range=255))
        rows.append({'method': name, 'image': stem, 'psnr': ps[-1], 'ssim': ss[-1]})
    summary[name] = (len(ps), np.mean(ps), np.mean(ss))
 
with open(args.csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'image', 'psnr', 'ssim'])
    w.writeheader(); w.writerows(rows)
 
print(f'\n| Method | Number of images | PSNR | SSIM |\n|---|---|---|---|')
for name, (n, ps, ss) in summary.items():
    print(f'| {name} | {n} | {ps:.2f} | {ss:.4f} |')