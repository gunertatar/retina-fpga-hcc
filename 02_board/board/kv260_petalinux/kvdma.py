"""Minimal AXI-DMA driver for PetaLinux (no PYNQ): physically contiguous buffers from
/dev/dma_heap/reserved, cache maintenance with DMA_BUF_IOCTL_SYNC, registers via /dev/mem."""
import os, mmap, fcntl, struct, time, numpy as np
from devmem import Mem
DMA_HEAP_IOCTL_ALLOC = 0xC0184800; DMA_BUF_IOCTL_SYNC = 0x40086200
SYNC_READ, SYNC_WRITE, SYNC_START, SYNC_END = 1, 2, 0, 4
class Buf:
    def __init__(self, shape, dtype, heap='/dev/dma_heap/reserved'):
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize; sz = (self.nbytes + 4095) & ~4095
        hf = os.open(heap, os.O_RDWR); d = bytearray(struct.pack('<QIIQ', sz, 0, os.O_RDWR | os.O_CLOEXEC, 0))
        fcntl.ioctl(hf, DMA_HEAP_IOCTL_ALLOC, d); self.fd = struct.unpack('<QIIQ', d)[1]
        self.mm = mmap.mmap(self.fd, sz, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.a = np.frombuffer(self.mm, dtype, int(np.prod(shape))).reshape(shape)
        sig = np.frombuffer(os.urandom(64), np.uint64); u = np.frombuffer(self.mm, np.uint64)
        self.sync(SYNC_WRITE | SYNC_START); u[:8] = sig; self.sync(SYNC_WRITE | SYNC_END)
        self.phys = Buf.find(sig)
    def sync(self, fl): fcntl.ioctl(self.fd, DMA_BUF_IOCTL_SYNC, struct.pack('<Q', fl))
    @staticmethod
    def find(sig):
        f = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC); step = 0x4000000
        for base in list(range(0x10000000, 0x80000000, step)) + list(range(0x800000000, 0x880000000, step)):
            m = mmap.mmap(f, step, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
            v = np.frombuffer(m, np.uint64)[::512]
            for i in np.nonzero(v == sig[0])[0]:
                if np.array_equal(np.frombuffer(m, np.uint64, 8, int(i) * 4096), sig): return base + int(i) * 4096
            del v; m.close()
        raise RuntimeError('buffer not found')
class Pool:
    """One physically contiguous region; CPU access through an uncached /dev/mem mapping (always coherent)."""
    def __init__(self, size=1 << 20, tries=8):
        keep = []
        for _ in range(tries):
            try:
                b = Buf((size,), np.uint8); self.dmabuf = b; break
            except RuntimeError: keep.append(1)
        else: raise RuntimeError('no findable contiguous buffer')
        self.phys = b.phys; f = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(f, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=self.phys); self.off = 0
    def take(self, shape, dtype):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize; o = self.off; self.off = (o + n + 4095) & ~4095
        v = type('V', (), {})(); v.a = np.frombuffer(self.mm, dtype, int(np.prod(shape)), o).reshape(shape); v.phys = self.phys + o; v.nbytes = n
        v.sync = lambda fl: None; return v
class DMA:
    def __init__(self, base): self.r = Mem(base)
    def reset(self):
        for o in (0x00, 0x30): self.r.w(o, 4)
        time.sleep(0.001)
        for o in (0x00, 0x30): self.r.w(o, 1)
    def recv(self, buf):   # S2MM
        self.r.w(0x30, 1); self.r.w(0x48, buf.phys & 0xFFFFFFFF); self.r.w(0x4C, buf.phys >> 32); self.r.w(0x58, buf.nbytes)
    def send(self, buf):   # MM2S
        self.r.w(0x00, 1); self.r.w(0x18, buf.phys & 0xFFFFFFFF); self.r.w(0x1C, buf.phys >> 32); self.r.w(0x28, buf.nbytes)
    def wait(self, ch):    # ch 0x04 (MM2S SR) or 0x34 (S2MM SR); bit1 = idle
        t0 = time.time()
        while not (self.r.r(ch) & 0x2):
            if time.time() - t0 > 2: raise TimeoutError(hex(self.r.r(ch)))
class Accel:
    def __init__(self):
        self.d0, self.d1 = DMA(0xA0000000), DMA(0xA0010000); self.d0.reset(); self.d1.reset()
        self.pool = Pool(); self.ib, self.mb, self.ob = self.pool.take((64, 64), np.uint8), self.pool.take((64, 64), np.uint8), self.pool.take((32, 64, 64), np.int16)
    def pre(self, p):
        self.ib.sync(SYNC_WRITE | SYNC_START); self.ib.a[:] = p; self.ib.sync(SYNC_WRITE | SYNC_END)
        self.mb.sync(SYNC_READ | SYNC_START)
        self.d1.recv(self.mb); self.d0.send(self.ib); self.d0.wait(0x04); self.d1.wait(0x34)   # KV260 wiring: preprocess MM2S=dma0, S2MM=dma1
        self.mb.sync(SYNC_READ | SYNC_END)
    def conv(self):
        self.ob.sync(SYNC_READ | SYNC_START)
        self.d0.recv(self.ob); self.d1.send(self.mb); self.d1.wait(0x04); self.d0.wait(0x34)   # conv_ip MM2S=dma1, S2MM=dma0
        self.ob.sync(SYNC_READ | SYNC_END)
