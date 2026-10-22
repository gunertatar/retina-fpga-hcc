import mmap, os, struct, sys
class Mem:
    def __init__(self, base, size=0x10000):
        self.f = os.open('/dev/mem', os.O_RDWR | os.O_SYNC); pg = base & ~0xFFF; self.off = base - pg
        self.m = mmap.mmap(self.f, size + self.off, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=pg)
    def r(self, o): return struct.unpack('<I', self.m[self.off+o:self.off+o+4])[0]
    def w(self, o, v): self.m[self.off+o:self.off+o+4] = struct.pack('<I', v & 0xFFFFFFFF)
if __name__ == '__main__':
    a = int(sys.argv[1], 16); m = Mem(a & ~0xFFFF); o = a & 0xFFFF
    if len(sys.argv) > 2: m.w(o, int(sys.argv[2], 16))
    print(hex(m.r(o)))
