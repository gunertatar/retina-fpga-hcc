"""Run one power-measurement phase on the PYNQ-Z1 for a fixed duration.
usage: pz_phase.py <idle|pl|ps|full> <seconds>"""
import sys, os, time, numpy as np
from pynq import Overlay, allocate
from ps_numpy import PS
import ps_numpy
ph, dur = sys.argv[1], float(sys.argv[2]); H = os.path.dirname(os.path.abspath(__file__))
ol = Overlay(os.path.join(H, 'retina_system_wrapper.bit')); d0, d1 = ol.axi_dma_0, ol.axi_dma_1
ib = allocate((64, 64), np.uint8); mb = allocate((64, 64), np.uint8); ob = allocate((32, 64, 64), np.int16)
P = np.load(os.path.join(H, 'board_inputs.npz'))['val_patches']
ps_numpy.use_c_gemm(os.path.join(H, 'libgemm.so')); ps = PS(os.path.join(H, 'w_hwaware.npz'))
def pre():
    ib[:] = P[0]; d0.recvchannel.transfer(mb); d0.sendchannel.transfer(ib); d0.sendchannel.wait(); d0.recvchannel.wait()
def conv():
    d1.recvchannel.transfer(ob); d1.sendchannel.transfer(mb); d1.sendchannel.wait(); d1.recvchannel.wait()
fn = {'idle': lambda: time.sleep(0.1), 'pl': lambda: (pre(), conv()),
      'ps': lambda: ps.forward(ob.astype(np.float32) / 1024),
      'full': lambda: (pre(), conv(), ps.forward(ob.astype(np.float32) / 1024))}[ph]
pre(); conv()
print(f'PHASE {ph} RUNNING from {time.strftime("%H:%M:%S")} for {dur:.0f}s', flush=True)
t0 = time.time(); n = 0
while time.time() - t0 < dur: fn(); n += 1
el = time.time() - t0
print(f'PHASE {ph} DONE n={n} {1e3*el/n:.2f} ms/iter', flush=True)
