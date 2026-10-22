"""R1-5: sensitivity of segmentation accuracy to the fixed-point format of conv_ip.
Arithmetic isolated (preprocess_ip bypassed, base model), DRIVE test set, 20 images."""
import sys, json, numpy as np, torch
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model')
import common, hwemu
m, sd = common.load_model(); data = common.load_split('DRIVE', 'test')
# enc1 SQNR reference: 200 random FOV patches
rng = np.random.default_rng(0); P = []
for d in data:
    x = np.floor(common.train_preprocess(d['rgb']) * 255)
    for _ in range(10):
        r, c = rng.integers(0, x.shape[0] - 64), rng.integers(0, x.shape[1] - 64); P.append(x[r:r+64, c:c+64])
P = torch.tensor(np.stack(P)[:, None], dtype=torch.float64, device=common.DEV)
with torch.no_grad(): ref = m.enc1.conv((P / 255).float()).double()
def run(tag, wb, ab, fr, accb=32, rnd=False):
    q, _ = hwemu.fold_enc1(sd, frac=fr, wbits=wb); hw = common.HWFront(q, frac=fr, acc_bits=accb, act_bits=ab, shift_round=rnd, preprocess=False)
    # emulate input quantisation at the same fractional precision (x_q = floor(px*2^F/255))
    with torch.no_grad(): e = hw(P) / (1 << fr)
    sqnr = 10 * np.log10(float((ref ** 2).sum() / ((e - ref) ** 2).sum()))
    D = []
    for d in data:
        p = common.infer_image(m, d['rgb'], 'hw', hw); D.append(common.metrics(p > 0.5, d['gt'], d['fov'], p))
    r = {k: float(np.mean([x[k] for x in D])) for k in D[0]}; r.update(tag=tag, wbits=wb, abits=ab, frac=fr, accbits=accb, round=rnd, sqnr=sqnr)
    r['dice_per_image'] = [float(x['Dice']) for x in D]
    print(f"{tag:22s} W{wb} A{ab} F{fr} acc{accb} rnd={int(rnd)}  SQNR={sqnr:6.2f} dB  Dice={r['Dice']:.4f} AUC={r['AUC']:.4f} Sens={r['Sens']:.4f}", flush=True)
    return r
R = []
for wb in (16, 12, 10, 8):
    for fr in (4, 6, 8, 10, 12):
        if fr >= wb: continue
        R.append(run('grid', wb, wb, fr))
for accb in (20, 24, 28):
    R.append(run('acc_width', 16, 16, 10, accb))
R.append(run('round_shift', 16, 16, 10, 32, True))
R.append(run('round_shift', 8, 8, 4, 32, True))
R.append(run('round_shift', 8, 8, 6, 32, True))
json.dump(R, open('results/sweep_precision.json', 'w'), indent=1)
