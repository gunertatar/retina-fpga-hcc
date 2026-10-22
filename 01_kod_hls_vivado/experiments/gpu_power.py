"""GPU/host platform power from the laptop battery discharge rate (AC adapter must be unplugged).
Phases: idle, batch-1 patch inference, batched full-image inference. Reports mean system power and
throughput; the workload power is (phase - idle)."""
import time, threading, numpy as np, torch, json, sys
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model'); import common
def pw():
    try: return int(open('/sys/class/power_supply/BAT0/power_now').read()) / 1e6
    except Exception: return float('nan')
st = open('/sys/class/power_supply/BAT0/status').read().strip(); print('battery status:', st)
if st not in ('Discharging',): print('!! unplug the AC adapter first'); sys.exit(1)
m, _ = common.load_model(); x1 = torch.randn(1, 1, 64, 64, device='cuda'); xb = torch.randn(306, 1, 64, 64, device='cuda')
S = []; ph = ['idle']; stop = [False]
def smp():
    while not stop[0]: S.append((ph[0], pw())); time.sleep(0.5)
th = threading.Thread(target=smp); th.start(); R = {}
def phase(name, dur, f):
    ph[0] = name; t0 = time.time(); n = 0
    with torch.no_grad():
        while time.time() - t0 < dur: f(); n += 1
        torch.cuda.synchronize()
    R[name + '_iters_per_s'] = n / (time.time() - t0); print(name, 'done', n)
phase('idle', 90, lambda: time.sleep(0.2))
phase('batch1', 90, lambda: (m(x1), torch.cuda.synchronize()))
phase('batched_image', 90, lambda: (m(xb), torch.cuda.synchronize()))
stop[0] = True; th.join()
for p in ('idle', 'batch1', 'batched_image'):
    v = [s[1] for s in S if s[0] == p][6:]; R[p + '_W'] = float(np.mean(v)); R[p + '_W_std'] = float(np.std(v))
print(json.dumps(R, indent=1)); json.dump(R, open('results/gpu_power.json', 'w'), indent=1)
