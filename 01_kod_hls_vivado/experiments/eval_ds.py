"""Evaluate FP32 and deployed (bit-exact PL front-end) pipelines on a dataset test split."""
import sys, json, numpy as np, torch
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model')
import common, hwemu
ds, fp32_ck, hw_ck, tag = sys.argv[1:5]
_, sd0 = common.load_model('weights/best_model.pth'); q, _ = hwemu.fold_enc1(sd0)
hw = common.HWFront(q, preprocess=True)
data = common.load_split(ds, 'test'); res = {}
for name, ck, mode in (('fp32', fp32_ck, 'fp32'), ('fpga', hw_ck, 'hw')):
    m, _ = common.load_model(ck); R = []
    for d in data:
        p = common.infer_image(m, d['rgb'], mode, hw); mt = common.metrics(p > 0.5, d['gt'], d['fov'], p); mt['id'] = d['id']; R.append({k: (float(v) if k != 'id' else v) for k, v in mt.items()})
    res[name] = R
    print(f"{ds:6s} {tag:10s} {name:5s} n={len(R)} " + ' '.join(f"{k}={np.mean([x[k] for x in R]):.4f}±{np.std([x[k] for x in R], ddof=1):.4f}" for k in ['Dice', 'Sens', 'Spec', 'Acc', 'AUC']), flush=True)
json.dump(res, open(f'results/ds_{ds}_{tag}.json', 'w'), indent=1)
