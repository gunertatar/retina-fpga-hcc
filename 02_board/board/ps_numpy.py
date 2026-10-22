"""Reference PS-side implementation of LightUNet layers enc2 ... out_conv (float32 NumPy).
Matches the PyTorch model exactly up to float32 rounding (BN folded, ConvTranspose2d k=2 s=2)."""
import numpy as np, ctypes, os
_lib = None
def use_c_gemm(path):
    """Replace NumPy matmul by the OpenMP C SGEMM (gemm.c) for all PS convolutions."""
    global _lib; _lib = ctypes.CDLL(path)
    _lib.sgemm.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
    _lib.set_threads(int(os.environ.get("PS_THREADS", "4")))
def mm(A, B):
    if _lib is None: return A @ B
    A = np.ascontiguousarray(A, np.float32); B = np.ascontiguousarray(B, np.float32)
    C = np.empty((A.shape[0], B.shape[1]), np.float32)
    _lib.sgemm(A.shape[0], B.shape[1], A.shape[1], A.ctypes.data, B.ctypes.data, C.ctypes.data); return C
def fold(W, g, b, mu, var, eps=1e-5):
    s = g / np.sqrt(var + eps); return (W * s[:, None, None, None]).astype(np.float32), (b - mu * s).astype(np.float32)
class PS:
    def __init__(self, npz):
        r = np.load(npz); self.L = {}
        def dc(p):
            w1, b1 = fold(r[p+'.block.0.weight'], r[p+'.block.1.weight'], r[p+'.block.1.bias'], r[p+'.block.1.running_mean'], r[p+'.block.1.running_var'])
            w2, b2 = fold(r[p+'.block.3.weight'], r[p+'.block.4.weight'], r[p+'.block.4.bias'], r[p+'.block.4.running_mean'], r[p+'.block.4.running_var'])
            return (w1.reshape(w1.shape[0], -1), b1, w2.reshape(w2.shape[0], -1), b2)
        for n in ('enc1.conv', 'enc2.conv', 'enc3.conv', 'bridge', 'dec3.conv', 'dec2.conv', 'dec1.conv'): self.L[n] = dc(n)
        for n in ('dec3', 'dec2', 'dec1'):
            W = r[n+'.up.weight'].astype(np.float32)  # (Cin, Cout, 2, 2)
            self.L[n+'.up'] = (np.ascontiguousarray(W.transpose(1, 2, 3, 0).reshape(-1, W.shape[0])), r[n+'.up.bias'].astype(np.float32), W.shape[1])
        self.L['out'] = (r['out_conv.weight'].reshape(1, -1).astype(np.float32), r['out_conv.bias'].astype(np.float32))
    @staticmethod
    def conv3(x, w, b):
        C, H, W = x.shape; xp = np.zeros((C, H + 2, W + 2), np.float32); xp[:, 1:-1, 1:-1] = x
        cols = np.empty((C, 9, H, W), np.float32); k = 0
        for kr in range(3):
            for kc in range(3): cols[:, k] = xp[:, kr:kr+H, kc:kc+W]; k += 1
        y = mm(w, cols.reshape(C * 9, H * W)) + b[:, None]
        return np.maximum(y, 0, out=y).reshape(-1, H, W)
    def dconv(self, x, n):
        w1, b1, w2, b2 = self.L[n]; return self.conv3(self.conv3(x, w1, b1), w2, b2)
    @staticmethod
    def pool(x):
        C, H, W = x.shape; return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))
    def up(self, x, n):
        Wm, b, Co = self.L[n+'.up']; C, H, W = x.shape
        y = mm(Wm, x.reshape(C, H * W)).reshape(Co, 2, 2, H, W)
        return y.transpose(0, 3, 1, 4, 2).reshape(Co, 2 * H, 2 * W) + b[:, None, None]
    def forward(self, s1, timer=None):
        T = timer or (lambda name, f: f())
        x = T('pool1', lambda: self.pool(s1))
        s2 = T('enc2', lambda: self.dconv(x, 'enc2.conv')); x = T('pool2', lambda: self.pool(s2))
        s3 = T('enc3', lambda: self.dconv(x, 'enc3.conv')); x = T('pool3', lambda: self.pool(s3))
        x = T('bridge', lambda: self.dconv(x, 'bridge'))
        for n, s in (('dec3', s3), ('dec2', s2), ('dec1', s1)):
            u = T(n + '.up', lambda: self.up(x, n)); c = T(n + '.concat', lambda: np.concatenate([s, u], 0))
            x = T(n + '.conv', lambda: self.dconv(c, n + '.conv'))
        w, b = self.L['out']
        o = T('out', lambda: (w @ x.reshape(x.shape[0], -1) + b[:, None]).reshape(x.shape[1:]))
        return T('sigmoid', lambda: 1.0 / (1.0 + np.exp(-o)))
