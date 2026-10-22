import json, numpy as np, sys
from scipy import stats
rng = np.random.default_rng(0)
def paired(a, b, name):
    a, b = np.asarray(a), np.asarray(b); d = b - a
    w = stats.wilcoxon(a, b, zero_method='pratt') if np.any(d != 0) else None
    t = stats.ttest_rel(b, a)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); dz = d.mean() / d.std(ddof=1) if d.std() > 0 else 0
    # TOST equivalence, margin +-0.01
    m = 0.01; t1 = stats.ttest_1samp(d, -m, alternative='greater').pvalue; t2 = stats.ttest_1samp(d, m, alternative='less').pvalue
    r = dict(name=name, n=len(d), mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), ci=[lo, hi], wilcoxon_p=w.pvalue if w else 1.0, t_p=t.pvalue, dz=dz, tost_p=max(t1, t2))
    print(f"{name:48s} n={len(d):2d} {a.mean():.4f} -> {b.mean():.4f}  Δ={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  Wilcoxon p={r['wilcoxon_p']:.3g}  t p={t.pvalue:.3g}  dz={dz:+.2f}  TOST(±0.01) p={r['tost_p']:.3g}")
    return r
out = []
B = json.load(open('results/acc_DRIVE_base.json')); H = json.load(open('results/acc_DRIVE_hwaware.json'))
for met in ('Dice', 'AUC', 'Sens', 'Spec', 'Acc'):
    ref = [x[met] for x in B['A_fp32']]
    out.append(paired(ref, [x[met] for x in B['D_conv_ip_only(Q10)']], f'DRIVE {met}: FP32 vs Q-format conv_ip only'))
    out.append(paired(ref, [x[met] for x in B['C_deployed(pre+conv_ip)']], f'DRIVE {met}: FP32 vs deployed (original PS)'))
    out.append(paired(ref, [x[met] for x in H['C_deployed(pre+conv_ip)']], f'DRIVE {met}: FP32 vs deployed (HW-aware PS)'))
for ds in ('STARE', 'CHASE', 'HRF'):
    try: R = json.load(open(f'results/ds_{ds}_trained.json'))
    except FileNotFoundError: continue
    for met in ('Dice', 'AUC'):
        out.append(paired([x[met] for x in R['fp32']], [x[met] for x in R['fpga']], f'{ds} {met}: FP32 vs FPGA (trained)'))
json.dump(out, open('results/stats.json', 'w'), indent=1, default=float)
