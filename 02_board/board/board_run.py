"""On-board measurements for the revision (PYNQ-Z1 and Kria KV260, PYNQ framework).
usage: python3 board_run.py <board: pynqz1|kv260> <mode: all|validate|latency|power|image>
Outputs -> results_<board>/ (copy back to host)."""
import sys, os, time, json, threading, glob
import numpy as np
from ps_numpy import PS
BOARD = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else 'all'
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results_' + BOARD + os.environ.get('TAG', '')); os.makedirs(OUT, exist_ok=True)
if BOARD == 'kv260':   # PetaLinux, no PYNQ: /dev/mem + dma-heap driver (kvdma.py)
    import kvdma; A = kvdma.Accel(); ib, mb, ob = A.ib.a, A.mb.a, A.ob.a
    def pre(p): A.pre(p)
    def conv(): A.conv()
else:
    from pynq import Overlay, allocate
    ol = Overlay(os.path.join(os.environ.get('BITDIR', HERE), 'retina_system_wrapper.bit')); dma0, dma1 = ol.axi_dma_0, ol.axi_dma_1
    ib = allocate((64, 64), np.uint8); mb = allocate((64, 64), np.uint8); ob = allocate((32, 64, 64), np.int16)
D = np.load(os.path.join(HERE, 'board_inputs.npz'))
ps = PS(os.path.join(HERE, 'w_hwaware.npz'))
if os.environ.get('CGEMM', '1') == '1':
    import ps_numpy; ps_numpy.use_c_gemm(os.path.join(HERE, 'libgemm.so')); print('PS GEMM: C/OpenMP', flush=True)
if BOARD != 'kv260':
    def pre(p):
        ib[:] = p; dma0.recvchannel.transfer(mb); dma0.sendchannel.transfer(ib); dma0.sendchannel.wait(); dma0.recvchannel.wait()
    def conv():
        dma1.recvchannel.transfer(ob); dma1.sendchannel.transfer(mb); dma1.sendchannel.wait(); dma1.recvchannel.wait()
log = {}
def save(): json.dump(log, open(os.path.join(OUT, 'log.json'), 'w'), indent=1)
if MODE in ('all', 'validate'):
    P = D['val_patches']; PRE = np.zeros_like(P); CONV = np.zeros((len(P), 32, 64, 64), np.int16)
    for i, p in enumerate(P):
        pre(p); PRE[i] = mb; conv(); CONV[i] = ob
    np.savez_compressed(os.path.join(OUT, 'validate_out.npz'), pre=PRE, conv=CONV); print('validate: saved', len(P), 'patches', flush=True)
if MODE in ('all', 'latency'):
    P = D['val_patches'][:int(os.environ.get('NLAT', '20'))]; T = {}
    def tm(name, f):
        t0 = time.perf_counter(); r = f(); T.setdefault(name, []).append(time.perf_counter() - t0); return r
    for _ in range(5): pre(P[0]); conv(); ps.forward(ob.astype(np.float32) / 1024)   # warm-up
    for p in P:
        t0 = time.perf_counter(); tm('pl_preprocess', lambda: pre(p)); tm('pl_enc1', conv)
        s1 = tm('ps_readback', lambda: ob.astype(np.float32) / 1024)
        ps.forward(s1, tm); T.setdefault('total', []).append(time.perf_counter() - t0)
        tm('ps_only_enc1_numpy', lambda: ps.dconv((p.astype(np.float32) / 255)[None], 'enc1.conv'))   # same stage in software, for the on-device speedup
    log['latency_ms'] = {k: dict(mean=1e3*float(np.mean(v)), std=1e3*float(np.std(v)), median=1e3*float(np.median(v)), n=len(v)) for k, v in T.items()}
    for k, v in log['latency_ms'].items(): print(f"  {k:14s} {v['mean']:10.3f} ms ± {v['std']:.3f}", flush=True)
    save()
if MODE in ('all', 'power'):
    hw = [f for f in glob.glob('/sys/class/hwmon/hwmon*/power1_input')]
    samples = []; stop = [False]; phase = ['idle']
    def sampler():
        while not stop[0]:
            if hw: samples.append((time.time(), phase[0], int(open(hw[0]).read()) / 1e6))
            time.sleep(0.05)
    th = threading.Thread(target=sampler); th.start()
    def run_phase(name, dur, fn):
        phase[0] = name; print(f'PHASE {name} START {time.strftime("%H:%M:%S")} ({dur}s)  <-- read the external meter now if applicable', flush=True)
        t0 = time.time(); n = 0
        while time.time() - t0 < dur: fn(); n += 1
        el = time.time() - t0; print(f'PHASE {name} END   n={n} ({el/n*1e3 if n else 0:.2f} ms/iter)', flush=True); return n, el
    P = D['val_patches']
    PD = int(os.environ.get('PHASE_S', '40'))
    n_idle, _ = run_phase('idle', PD, lambda: time.sleep(0.1))
    n_pl, t_pl = run_phase('pl_only', PD, lambda: (pre(P[0]), conv()))
    n_ps, t_ps = run_phase('ps_only', max(PD, 60), lambda: ps.forward(ob.astype(np.float32) / 1024))
    n_full, t_full = run_phase('full', max(PD, 60), lambda: (pre(P[0]), conv(), ps.forward(ob.astype(np.float32) / 1024)))
    phase[0] = 'end'; stop[0] = True; th.join()
    res = {'iters': dict(pl_only=[n_pl, t_pl], ps_only=[n_ps, t_ps], full=[n_full, t_full])}
    if hw:
        for ph in ('idle', 'pl_only', 'ps_only', 'full'):
            v = [s[2] for s in samples if s[1] == ph][10:]   # drop first 0.5 s of each phase
            res[ph + '_W'] = dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
        print(json.dumps(res, indent=1))
    log['power'] = res; save()
if MODE in ('all', 'image'):
    I = D['img_patches']; G = D['img_grid']; H, W = D['img_shape']; prob = np.zeros((H, W)); cnt = np.zeros((H, W)); t0 = time.time()
    for p, (r, c) in zip(I, G):
        pre(p); conv(); o = ps.forward(ob.astype(np.float32) / 1024); prob[r:r+64, c:c+64] += o; cnt[r:r+64, c:c+64] += 1
    el = time.time() - t0; np.save(os.path.join(OUT, 'image_prob.npy'), prob / np.maximum(cnt, 1))
    log['full_image'] = dict(seconds=el, patches=len(I), ms_per_patch=1e3 * el / len(I)); save(); print('full image', el, 's', len(I), 'patches', flush=True)
print('DONE', BOARD, MODE)
