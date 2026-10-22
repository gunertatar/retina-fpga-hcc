"""Bit-exact NumPy model of the two HLS cores (preprocess_ip, conv_ip).

Arithmetic mirrors hls/preprocess.cpp and conv_ip (int16-output version):
  input  x_q = floor(px*2^F/255)                      (fixed_t = ap_int<16>)
  conv1  acc = (b1<<F) + sum x_q*w1  (mod 2^32)       (acc_t = ap_int<32>)
         mid = clip(acc>>F, 0, 32767)                 (ReLU + saturation)
  conv2  acc = (b2<<F) + sum mid*w2  (mod 2^32); out = clip(acc>>F,0,32767)
Integer addition modulo 2^32 is associative, so the result is independent of
ICH_TILE and of the reduction-tree shape.
"""
import re, numpy as np

def wrap(a, bits):
    m = 1 << bits
    a = np.asarray(a, dtype=np.int64) & (m - 1)
    return np.where(a >= m >> 1, a - m, a)

def read_header(path):
    txt = open(path).read()
    out = {}
    for name in ("CONV1_W", "CONV1_B", "CONV2_W", "CONV2_B"):
        m = re.search(name + r"\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt)
        out[name] = np.array([int(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()], dtype=np.int64)
    return out

def fold_enc1(sd, frac=10, wbits=16, rounding="round"):
    """BN-fold enc1 and quantise (as gen_weights.py) -> dict of int64 arrays."""
    def g(k): return sd[k].detach().cpu().numpy().astype(np.float64)
    def fold(p, b):
        s = g(f"enc1.conv.block.{b}.weight") / np.sqrt(g(f"enc1.conv.block.{b}.running_var") + 1e-5)
        return g(f"enc1.conv.block.{p}.weight") * s[:, None, None, None], g(f"enc1.conv.block.{b}.bias") - g(f"enc1.conv.block.{b}.running_mean") * s
    w1, b1 = fold(0, 1); w2, b2 = fold(3, 4)
    lim = (1 << (wbits - 1))
    q = lambda x: np.clip(np.round(x * (1 << frac)) if rounding == "round" else np.floor(x * (1 << frac)), -lim, lim - 1).astype(np.int64)
    return {"CONV1_W": q(w1).ravel(), "CONV1_B": q(b1), "CONV2_W": q(w2).ravel(), "CONV2_B": q(b2)}, (w1, b1, w2, b2)

def _conv3x3_int(x, w, b, frac, acc_bits):
    # x: (Cin,H,W) int64 ; w: (Cout,Cin,3,3) ; b: (Cout,)
    Cin, H, W = x.shape
    xp = np.zeros((Cin, H + 2, W + 2), np.int64); xp[:, 1:-1, 1:-1] = x
    cols = np.stack([xp[:, kr:kr + H, kc:kc + W] for kr in range(3) for kc in range(3)], 1)  # Cin,9,H,W
    acc = np.tensordot(w.reshape(w.shape[0], Cin, 9), cols, axes=([1, 2], [0, 1]))       # exact int64
    acc = acc + (b[:, None, None] << frac)
    return wrap(acc, acc_bits)

def conv_ip(px_u8, q, frac=10, acc_bits=32, act_bits=16, shift_round=False, in_div=255):
    """px_u8: (64,64) uint8 -> (32,64,64) int64 activations at scale 2^frac."""
    x = (px_u8.astype(np.int64) * (1 << frac)) // in_div
    w1 = q["CONV1_W"].reshape(32, 1, 3, 3); w2 = q["CONV2_W"].reshape(32, 32, 3, 3)
    amax = (1 << (act_bits - 1)) - 1
    def post(acc):
        if shift_round: acc = acc + (1 << (frac - 1))
        return np.clip(acc >> frac, 0, amax)
    mid = post(_conv3x3_int(x[None], w1, q["CONV1_B"], frac, acc_bits))
    return post(_conv3x3_int(mid, w2, q["CONV2_B"], frac, acc_bits))

G = np.array([[16, 32, 16], [32, 64, 32], [16, 32, 16]], np.int64)
def preprocess_ip(px_u8, prev_last_row=None):
    """Exact model of preprocess.cpp. The column loop shifts the line buffer one
    column at a time, so when output (r,c) is formed the right-hand window column
    (c+1) still holds rows r-3..r-1 while columns c-1, c hold rows r-2..r.
    The result is written to row r; rows 0-1 and columns 0,63 pass through."""
    I = px_u8.astype(np.int64); H, W = I.shape
    Z = np.zeros(W, np.int64) if prev_last_row is None else prev_last_row.astype(np.int64)
    row = lambda k: I[k] if k >= 0 else Z
    T = I.copy()
    for r in range(2, H):
        new = [row(r - 2), row(r - 1), row(r)]; old = [row(r - 3), row(r - 2), row(r - 1)]
        c = np.arange(1, W - 1)
        s = sum(G[kr, 0] * new[kr][c - 1] + G[kr, 1] * new[kr][c] + G[kr, 2] * old[kr][c + 1] for kr in range(3))
        T[r, 1:W - 1] = (s >> 8) & 0xFF
    mn, mx = T.min(), T.max(); rng = max(mx - mn, 1); sf = (255 * 256) // rng
    return np.clip(((T - mn) * sf) >> 8, 0, 255).astype(np.uint8)
