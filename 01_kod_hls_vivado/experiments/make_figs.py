import sys, json, re, numpy as np, torch
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
RED, BLU, YEL, GRN, VIO, ORG, INK = '#E34948', '#2A78D6', '#EDA100', '#1BAF7A', '#4A3AA7', '#EB6834', '#1F2933'
OUT = 'figures/'
def parse(f):
    d = {}
    for l in open(f):
        m = re.match(r'\s+(\S+)\s+([\d.]+) ms', l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
kv = parse('results/board_raw/kv260/latency.log')
pz = parse('results/board_raw/pynqz1/lat_mar26.log')
gp = json.load(open('results/results_gpu_profile.json'))['stage_median']
cx = {s['name']: s['macs'] for s in json.load(open('results/complexity.json'))['stages']}
groups = ['enc1', 'enc2', 'enc3', 'bridge', 'dec3', 'dec2', 'dec1', 'other']
def arm_groups(d):   # all-software profile: enc1 executed in software (ps_only_enc1_numpy)
    g = dict(enc1=d['ps_only_enc1_numpy'], enc2=d['enc2'], enc3=d['enc3'], bridge=d['bridge'])
    for k in ('dec3', 'dec2', 'dec1'): g[k] = d[k + '.up'] + d[k + '.concat'] + d[k + '.conv']
    g['other'] = d['pool1'] + d['pool2'] + d['pool3'] + d['out'] + d['sigmoid']; return g
def gpu_groups(d):
    g = {'enc1': d['enc1 DoubleConv'], 'enc2': d['enc2 DoubleConv'], 'enc3': d['enc3 DoubleConv'], 'bridge': d['bridge DoubleConv']}
    for k in ('dec3', 'dec2', 'dec1'): g[k] = d[f'{k} ConvTranspose'] + d[f'{k} concat'] + d[f'{k} DoubleConv']
    g['other'] = d['pool1'] + d['pool2'] + d['pool3'] + d['out 1x1 conv'] + d['sigmoid']; return g
mac = {'enc1': cx['enc1'], 'enc2': cx['enc2'], 'enc3': cx['enc3'], 'bridge': cx['bridge'],
       'dec3': cx['dec3'] + cx['dec3.up'], 'dec2': cx['dec2'] + cx['dec2.up'], 'dec1': cx['dec1'] + cx['dec1.up'], 'other': cx['out']}
series = [('MAC count', mac, '#9AA5B1'), ('GPU GTX 1050 (batch 1)', gpu_groups(gp), VIO),
          ('KV260 Cortex-A53 ×4', arm_groups(kv), BLU), ('PYNQ-Z1 Cortex-A9 ×2', arm_groups(pz), GRN)]
fig, ax = plt.subplots(figsize=(7.0, 2.35))
w = 0.2; x = np.arange(len(groups))
for i, (lbl, d, col) in enumerate(series):
    tot = sum(d.values()); v = [100 * d[g] / tot for g in groups]
    b = ax.bar(x + (i - 1.5) * w, v, w, label=lbl, color=col, edgecolor='white', linewidth=0.3)
    if i == 0: pass
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('Share of forward pass (%)')
ax.axvspan(-0.5, 0.5, color=RED, alpha=0.08, lw=0); ax.text(0, ax.get_ylim()[1] * 0.97, 'offloaded\nto PL', ha='center', va='top', fontsize=6.5, color=RED)
ax.legend(ncol=4, frameon=False, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, 1.16)); ax.grid(axis='y', lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(OUT + 'fig_profile.pdf'); plt.close(fig)
json.dump({lbl: d for lbl, d, _ in series}, open('results/profile_groups.json', 'w'), indent=1)
print('profile groups:'); [print(' ', lbl, {g: round(100 * d[g] / sum(d.values()), 1) for g in groups}, 'total', round(sum(d.values()), 3)) for lbl, d, _ in series]
# ---------- sensitivity ----------
S = json.load(open('results/sweep_precision.json')); H = json.load(open('hls/sweep.json'))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
for wb, col, mk in ((16, RED, 'o'), (12, ORG, 's'), (10, BLU, '^'), (8, GRN, 'D')):
    pts = sorted([(r['frac'], r['Dice']) for r in S if r['tag'] == 'grid' and r['wbits'] == wb])
    a1.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.5, lw=1, color=col, label=f'{wb}-bit')
a1.axhline(0.8154, color=INK, lw=0.7, ls='--'); a1.text(4.1, 0.8154 + 0.012, 'FP32 reference (0.815)', fontsize=6.5)
a1.scatter([10], [0.8156], s=60, facecolors='none', edgecolors=INK, lw=0.9, zorder=5); a1.annotate('deployed\n(16-bit, F=10)', (10, 0.8156), (10.4, 0.62), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
a1.set_xlabel('Fractional bits F'); a1.set_ylabel('Dice (DRIVE test, 20 images)'); a1.set_ylim(0.3, 0.86); a1.legend(frameon=False, fontsize=6.8, title='word length', title_fontsize=6.8, loc='lower left')
a1.set_title('(a) Fixed-point format of conv_ip', fontsize=8)
for dev, col, mk in (('Zynq-7020', GRN, 'o'), ('ZU5EV (KV260)', RED, 's')):
    clk = 100.0 if dev == 'Zynq-7020' else 250.0
    pts = sorted([(r['T'], r['lat_max'] / clk / 1e3, r['dsp'], r['lut']) for r in H if r['device'] == dev and r['T'] < 32])
    if dev == 'ZU5EV (KV260)':
        pts.append((32, 3149862 / clk / 1e3, 88, 10315))          # deployed conv_ip (Vitis HLS report of the shipped IP)
    a2.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.5, lw=1, color=col, label=f'{dev} @ {clk:.0f} MHz')
    for p in pts:
        if p[0] in (8, 32) and not (dev == 'Zynq-7020' and p[0] == 32):
            a2.annotate(f'{p[2]} DSP, {p[3]/1e3:.1f}k LUT', (p[0], p[1]), (-8, 14) if p[0] == 32 else ((7, -9) if col == GRN else (-6, 8)), textcoords='offset points', fontsize=5.8, color=col, ha='right' if (p[0] == 32 or col == RED) else 'left')
    if dev == 'Zynq-7020':
        a2.scatter([32], [790579 / clk / 1e3], marker='x', color=col, s=18); a2.annotate('T=32: 326 DSP > 220 (infeasible)', (32, 790579 / clk / 1e3), (-6, -3), textcoords='offset points', fontsize=5.8, color=col, ha='right')
a2.scatter([8], [101.456], marker='*', s=40, color=GRN, zorder=5); a2.annotate('on-board 101.5 ms', (8, 101.456), (6, 3), textcoords='offset points', fontsize=5.8, color=GRN)
a2.scatter([32], [11.633], marker='*', s=40, color=RED, zorder=5); a2.annotate('on-board 11.6 ms', (32, 11.633), (-6, -9), textcoords='offset points', fontsize=5.8, color=RED, ha='right')
a2.set_xscale('log', base=2); a2.set_yscale('log'); a2.set_xticks([1, 2, 4, 8, 16, 32]); a2.set_xticklabels(['1', '2', '4', '8', '16', '32'])
a2.set_xlabel('ICH_TILE'); a2.set_ylabel('conv_ip latency (ms)'); a2.legend(frameon=False, fontsize=6.8); a2.set_title('(b) Input-channel tiling (Vitis HLS 2022.1)', fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'fig_sensitivity.pdf'); plt.close(fig)
# ---------- qualitative ----------
import common, hwemu
d = common.load_split('DRIVE', 'test')[0]; pb = np.load('results/board_raw/kv260/image_prob.npy')
m, _ = common.load_model(); pf = common.infer_image(m, d['rgb'], 'fp32', None, bs=128)
fov = d['fov']; gt = d['gt']
def err(p):
    img = np.zeros(p.shape + (3,)); tp = p & gt; fp = p & ~gt; fn = ~p & gt
    img[tp] = (1, 1, 1); img[fp] = (0.89, 0.29, 0.28); img[fn] = (0.16, 0.47, 0.84); img[~fov] = 0.15; return img
bf, bb = (pf > 0.5) & fov, (pb > 0.5) & fov
df = common.metrics(pf > 0.5, gt, fov)['Dice']; db = common.metrics(pb > 0.5, gt, fov)['Dice']
fig, axs = plt.subplots(1, 5, figsize=(7.1, 1.75))
panels = [(common.train_preprocess(d['rgb']), 'gray', '(a) Input (green channel)'), (gt, 'gray', '(b) Manual annotation'),
          (bf, 'gray', f'(c) FP32 GPU, Dice {df:.3f}'), (bb, 'gray', f'(d) KV260 on-board, Dice {db:.3f}'), (err(bb), None, '(e) On-board errors')]
for a, (im, cm, t) in zip(axs, panels):
    a.imshow(im, cmap=cm, interpolation='nearest'); a.set_title(t, fontsize=6.6); a.axis('off')
fig.text(0.99, 0.01, 'white: TP   red: FP   blue: FN', ha='right', fontsize=6)
fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_qualitative.pdf', dpi=300); plt.close(fig)
print('qual dice fp32', df, 'board', db)
# ---------- quantisation fidelity from on-board data (KV260, 200 patches) ----------
V = {k: np.concatenate([np.load(f'results/board_raw/kv260/validate_out_part{i}.npz')[k] for i in (1, 2)]) for k in ('pre', 'conv')}
m, _ = common.load_model(); m = m.cpu()
with torch.no_grad(): ref = m.enc1.conv(torch.tensor(V['pre'][:, None] / 255.0, dtype=torch.float32)).numpy()
hwv = V['conv'].astype(np.float64) / 1024.0
r_ch = np.array([np.corrcoef(hwv[:, c].ravel(), ref[:, c].ravel())[0, 1] for c in range(32)])
r_pool = np.corrcoef(hwv.ravel(), ref.ravel())[0, 1]; mae = np.abs(hwv - ref).mean(); mx = np.abs(hwv - ref).max()
lsb = 1 / 1024
print(f'fidelity: pooled r={r_pool:.6f} per-channel min={r_ch.min():.6f} mean={r_ch.mean():.6f} MAE={mae:.5f} ({mae/lsb:.2f} LSB) max={mx:.4f} ref range {ref.max():.2f}')
json.dump(dict(r_pool=r_pool, r_min=r_ch.min(), r_mean=r_ch.mean(), mae=mae, maxerr=mx, r_ch=r_ch.tolist()), open('results/fidelity.json', 'w'), indent=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
o = np.argsort(-r_ch); a1.bar(np.arange(32), 1 - r_ch[o], color=RED, width=0.75)
a1.set_yscale('log'); a1.set_xlabel('enc1 output channel (sorted)'); a1.set_ylabel('1 − Pearson r'); a1.set_title(f'(a) Per-channel fidelity (min r = {r_ch.min():.5f})', fontsize=8)
idx = np.random.default_rng(0).choice(ref.size, 20000, replace=False)
a2.scatter(ref.ravel()[idx], hwv.ravel()[idx], s=1, color=BLU, alpha=0.4, rasterized=True); lim = [0, float(ref.max())]
a2.plot(lim, lim, color=INK, lw=0.6, ls='--'); a2.set_xlabel('FP32 activation'); a2.set_ylabel('conv_ip (on-board) activation')
a2.set_title(f'(b) pooled r = {r_pool:.6f}, MAE = {mae/lsb:.2f} LSB', fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'fig_fidelity.pdf', dpi=300); plt.close(fig)
