import argparse, os, numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--orig', required=True) # original directory
p.add_argument('--out', nargs='+', required=True) # 1 or more output directories
args = p.parse_args()

names = sorted(n for n in os.listdir(args.orig) if n.lower().endswith(('.png', '.jpg', '.jpeg')))
bad = 0
for d in args.out:
    for n in names:
        path = os.path.join(d, n)
        if not os.path.isfile(path):
            print(f'MISSING {path}'); bad += 1; continue
        a, b = Image.open(os.path.join(args.orig, n)), Image.open(path)
        arr = np.asarray(b.convert('RGB'), dtype=np.float32)
        if a.size != b.size or not np.isfinite(arr).all() or arr.std() < 1:
            print(f'ERROR {path}: size {a.size}->{b.size}, std={arr.std():.1f}'); bad += 1
    print(f'{d}: checked {len(names)} images')
print('ALL OK' if bad == 0 else f'{bad} ERRORS')