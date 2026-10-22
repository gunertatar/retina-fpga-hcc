import sys, json, numpy as np, torch, time
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model')
import common, hwemu
dsname = sys.argv[1] if len(sys.argv) > 1 else 'DRIVE'
ckpt = sys.argv[2] if len(sys.argv) > 2 else 'weights/best_model.pth'
tag = sys.argv[3] if len(sys.argv) > 3 else 'base'
m, sd = common.load_model(ckpt); q, _ = hwemu.fold_enc1(sd)
cfgs = {'A_fp32': ('fp32', None),
        'B_fp32+preprocess_ip': ('fp32_pre', common.HWFront(q, preprocess=True)),
        'D_conv_ip_only(Q10)': ('hw', common.HWFront(q, preprocess=False)),
        'C_deployed(pre+conv_ip)': ('hw', common.HWFront(q, preprocess=True))}
data = common.load_split(dsname, 'test')
res = {k: [] for k in cfgs}
t0 = time.time()
for d in data:
    for k, (mode, hw) in cfgs.items():
        p = common.infer_image(m, d['rgb'], mode, hw)
        mt = common.metrics(p > 0.5, d['gt'], d['fov'], p); mt['id'] = d['id']; res[k].append(mt)
print(f'{dsname} ({len(data)} test images, {tag}) {time.time()-t0:.0f}s')
for k, v in res.items():
    s = ' '.join(f"{mm}={np.mean([x[mm] for x in v]):.4f}±{np.std([x[mm] for x in v], ddof=1):.4f}" for mm in ['Dice', 'IoU', 'Sens', 'Spec', 'Acc', 'AUC'])
    print(f'  {k:26s} {s}')
json.dump({k: [{kk: float(vv) if kk != 'id' else vv for kk, vv in x.items()} for x in v] for k, v in res.items()}, open(f'results/acc_{dsname}_{tag}.json', 'w'), indent=1)
