"""R1-3 / R1-2: per-stage MACs, parameters and on-chip storage of LightUNet as a function of patch size S and base width B;
cycle model of the conv_ip datapath (validated against HLS C-synthesis and on-board enc1 latency)."""
import json, math
def stages(S=64, B=32):
    L = []
    def dc(name, cin, cout, s): L.append((name, 'dconv', cin, cout, s))
    dc('enc1', 1, B, S); dc('enc2', B, 2*B, S//2); dc('enc3', 2*B, 4*B, S//4); dc('bridge', 4*B, 8*B, S//8)
    for n, cin, cout, s in (('dec3', 8*B, 4*B, S//4), ('dec2', 4*B, 2*B, S//2), ('dec1', 2*B, B, S)):
        L.append((n + '.up', 'convT', cin, cout, s)); dc(n, 2*cout, cout, s)
    L.append(('out', 'conv1', B, 1, S)); return L
def macs(st):
    n, t, cin, cout, s = st
    if t == 'dconv': return 9*cin*cout*s*s + 9*cout*cout*s*s
    if t == 'convT': return 4*cin*cout*s*s//4 * 1  # each output pixel receives cin products; outputs = cout*s*s
    return cin*cout*s*s
def params(st):
    n, t, cin, cout, s = st
    if t == 'dconv': return 9*cin*cout + 9*cout*cout + 4*cout
    if t == 'convT': return 4*cin*cout + cout
    return cin*cout + cout
def cycles(st, T):   # conv_ip-style datapath: 9 taps x ceil(Cin/T) cycles per output pixel per output channel
    n, t, cin, cout, s = st
    if t == 'dconv': return cout*s*s*9*math.ceil(cin/T) + cout*s*s*9*math.ceil(cout/T)
    if t == 'convT': return cout*s*s*math.ceil(cin/T)
    return cout*s*s*math.ceil(cin/T)
R = {}
for S in (64, 128, 256):
    for B in (16, 32, 64):
        L = stages(S, B); M = sum(macs(x) for x in L)
        row = dict(S=S, B=B, macs=M, params=sum(params(x) for x in L),
                   enc1_macs=macs(L[0]), enc1_share=macs(L[0])/M,
                   enc1_buf_kb=B*S*S*2/1024,                          # int16 intermediate buffer M[B][S][S]
                   enc1_w_kb=(9*B + 9*B*B + 2*B)*2/1024,
                   skip_kb=sum(x[3]*x[4]*x[4]*2 for x in L if x[1] == 'dconv' and x[0].startswith('enc'))/1024,
                   full_cycles_T32=sum(cycles(x, 32) for x in L), enc1_cycles_T32=cycles(L[0], 32), enc1_cycles_T8=cycles(L[0], 8))
        R[f'S{S}_B{B}'] = row
for k, r in R.items():
    print(f"{k:9s} MACs={r['macs']/1e6:8.1f}M params={r['params']/1e6:6.3f}M enc1 share={100*r['enc1_share']:5.1f}%  enc1 buf={r['enc1_buf_kb']:7.0f}KB  enc1 w={r['enc1_w_kb']:5.1f}KB  skips={r['skip_kb']:7.0f}KB  enc1@250MHz T32={r['enc1_cycles_T32']/250e3:7.2f}ms  full T32={r['full_cycles_T32']/250e3:8.1f}ms")
L = stages()
print('\nper-stage (S=64,B=32):')
for x in L: print(f"  {x[0]:9s} MACs={macs(x)/1e6:7.2f}M ({100*macs(x)/sum(macs(y) for y in L):5.1f}%)  params={params(x):8d}  cyc(T=32)={cycles(x,32)/1e6:6.2f}M")
json.dump(dict(scaling=R, stages=[dict(name=x[0], macs=macs(x), params=params(x), cycles_T32=cycles(x, 32), cycles_T8=cycles(x, 8)) for x in L]), open('results/complexity.json', 'w'), indent=1)
