"""R1-2: partition design-space exploration on the two targets using measured per-stage PS times,
the conv_ip cycle model calibrated on the measured enc1 latency, and on-chip memory feasibility."""
import json, re, math
def parse(f):
    d = {}
    for l in open(f):
        m = re.match(r'\s+(\S+)\s+([\d.]+) ms', l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
kv = parse('results/board_raw/kv260/latency.log'); pz = parse('results/board_raw/pynqz1/lat_mar26.log')
def cyc_dc(cin, cout, s, T): return cout*s*s*9*math.ceil(cin/T) + cout*s*s*9*math.ceil(cout/T)
# calibration: measured enc1 (PL, incl. DMA) / model cycles at the deployed clock
for name, d, clk, T, bram_used, bram_av, uram_av in (('KV260', kv, 250e6, 32, 201, 288, 64), ('PYNQ-Z1', pz, 100e6, 8, 202, 280, 0)):
    model_enc1 = cyc_dc(1, 32, 64, T) / clk * 1e3
    k = d['pl_enc1'] / model_enc1
    sw_enc1 = d['ps_only_enc1_numpy']
    base = d['total']
    sw_only = base - d['pl_preprocess'] - d['pl_enc1'] - d['ps_readback'] + sw_enc1   # all-software (preprocessing on PS neglected)
    rb_per_kb = d['ps_readback'] / 256.0
    # extra on-chip storage (int16) for adding enc2 / enc3: input + intermediate + weights, in BRAM18 (2.25 KB)
    enc2_kb = (32*32*32 + 64*32*32)*2/1024 + (18432+36864+128)*2/1024
    enc3_kb = (64*16*16 + 128*16*16)*2/1024 + (73728+147456+256)*2/1024
    rows = []
    rows.append(('SW only (all layers on PS)', sw_only, 'n/a', 0))
    rows.append(('pre + enc1 (deployed)', base, 'feasible (measured)', bram_used))
    pl2 = k * cyc_dc(32, 64, 32, T) / clk * 1e3
    p2 = base - d['pool1'] - d['enc2'] + pl2 + rb_per_kb * 128   # enc2 output (64x32x32 int16 = 128 KB) returned as skip2
    rows.append(('pre + enc1 + enc2', p2, None, bram_used + enc2_kb / 2.25))
    pl3 = k * cyc_dc(64, 128, 16, T) / clk * 1e3
    p3 = p2 - d['pool2'] - d['enc3'] + pl3 + rb_per_kb * 64
    rows.append(('pre + enc1 + enc2 + enc3', p3, None, bram_used + (enc2_kb + enc3_kb) / 2.25))
    print(f'== {name}: calibration factor k={k:.3f} (measured enc1 {d["pl_enc1"]:.2f} ms vs model {model_enc1:.2f} ms); SW enc1 {sw_enc1:.1f} ms')
    for r in rows:
        feas = r[2] or (('BRAM18 %.0f/%d' % (r[3], bram_av)) + (' -> needs URAM (%d avail)' % uram_av if r[3] > bram_av and uram_av else (' INFEASIBLE' if r[3] > bram_av else '')))
        print(f'   {r[0]:30s} e2e {r[1]:9.1f} ms  ({100*(r[1]/base-1):+.1f}% vs deployed; {sw_only/r[1]:.3f}x vs SW-only)  {feas}')
    # full network, cycle model (T), calibrated
    full = sum([cyc_dc(1,32,64,T), cyc_dc(32,64,32,T), cyc_dc(64,128,16,T), cyc_dc(128,256,8,T), cyc_dc(256,128,16,T), cyc_dc(128,64,32,T), cyc_dc(64,32,64,T)]) \
         + 128*16*16*math.ceil(256/T) + 64*32*32*math.ceil(128/T) + 32*64*64*math.ceil(64/T) + 64*64*math.ceil(32/T)
    print(f'   full network (cycle model, calibrated): {k*full/clk*1e3:.1f} ms/patch  ({base/(k*full/clk*1e3):.1f}x vs deployed hybrid)')
