"""Rigorous per-stage GPU profile of LightUNet (base_ch=32, trained weights), batch 1, 64x64.
Stage timing: CUDA events, 200 warm-up, 5 repetitions x 1000 iterations; median reported.
Residual decomposition: torch.profiler kernel-level breakdown by operator class."""
import torch, numpy as np, json, sys, torch.nn.functional as F
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model')
import common
m, _ = common.load_model(); torch.backends.cudnn.benchmark = False
x = torch.randn(1, 1, 64, 64, device='cuda')
def stages(x):
    s1 = m.enc1.conv(x); yield 'enc1 DoubleConv', s1
    p = m.enc1.pool(s1); yield 'pool1', p
    s2 = m.enc2.conv(p); yield 'enc2 DoubleConv', s2
    p = m.enc2.pool(s2); yield 'pool2', p
    s3 = m.enc3.conv(p); yield 'enc3 DoubleConv', s3
    p = m.enc3.pool(s3); yield 'pool3', p
    b = m.bridge(p); yield 'bridge DoubleConv', b
    y = b
    for name, d, s in (('dec3', m.dec3, s3), ('dec2', m.dec2, s2), ('dec1', m.dec1, s1)):
        u = d.up(y); yield f'{name} ConvTranspose', u
        c = torch.cat([s, u], 1); yield f'{name} concat', c
        y = d.conv(c); yield f'{name} DoubleConv', y
    o = m.out_conv(y); yield 'out 1x1 conv', o
    yield 'sigmoid', torch.sigmoid(o)
names = [n for n, _ in stages(x)]
res = {n: [] for n in names}; tot = []
with torch.no_grad():
    for _ in range(200): m(x)
    torch.cuda.synchronize()
    for rep in range(5):
        acc = {n: [] for n in names}
        for it in range(1000):
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
            evs[0].record(); k = 1
            for n, _ in stages(x):
                evs[k].record(); k += 1
            torch.cuda.synchronize()
            for i, n in enumerate(names): acc[n].append(evs[0].elapsed_time(evs[i + 1]) - (evs[0].elapsed_time(evs[i]) if i else 0))
        for n in names: res[n].append(float(np.median(acc[n])))
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
    for _ in range(3000):
        e0.record(); m(x); e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1))
    t = np.array(t)
    # batched throughput (full-image use-case: 306 patches of a DRIVE image)
    xb = torch.randn(306, 1, 64, 64, device='cuda'); tb = []
    for _ in range(20): m(xb)
    torch.cuda.synchronize()
    for _ in range(50):
        e0.record(); m(xb); e1.record(); torch.cuda.synchronize(); tb.append(e0.elapsed_time(e1))
med = {n: float(np.median(v)) for n, v in res.items()}; S = sum(med.values())
print(f"{'stage':22s} {'median ms':>10s} {'share %':>8s}  (rep spread)")
for n in names: print(f"{n:22s} {med[n]:10.4f} {100*med[n]/S:8.2f}  [{min(res[n]):.4f}-{max(res[n]):.4f}]")
print(f"sum of stages {S:.4f} ms | whole forward median {np.median(t):.4f} ms (p5 {np.percentile(t,5):.4f}, p95 {np.percentile(t,95):.4f}) -> {1000/np.median(t):.1f} patch/s")
print(f"batched 306 patches: median {np.median(tb):.2f} ms -> {306e3/np.median(tb):.0f} patch/s")
json.dump(dict(stage_median=med, reps=res, forward_median=float(np.median(t)), forward_p5=float(np.percentile(t,5)), forward_p95=float(np.percentile(t,95)), batched306_ms=float(np.median(tb))), open('results/results_gpu_profile.json', 'w'), indent=1)
