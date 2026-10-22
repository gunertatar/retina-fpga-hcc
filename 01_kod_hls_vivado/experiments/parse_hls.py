import re, glob, os, json
out = []
for P, lbl, clk in (('xc7z020clg400-1', 'Zynq-7020', 10.0), ('xck26-sfvc784-2LV-c', 'ZU5EV (KV260)', 3.8)):
    for T in (1, 2, 4, 8, 16, 32):
        f = f'hls/sweep/prj_{P}_T{T}/sol/syn/report/conv_ip_csynth.rpt'
        if not os.path.exists(f): continue
        t = open(f).read()
        est = re.search(r'\|\s*ap_clk\s*\|\s*([\d.]+) ns\s*\|\s*([\d.]+) ns', t)
        lat = re.search(r'\+ Latency:.*?\n\s*\|\s*(\d+)\|\s*(\d+)\|', t, re.S)
        tot = re.search(r'\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', t)
        avail = re.search(r'\|Available\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', t)
        # II of conv2 inner loop
        ii = re.findall(r'\|-\s*(CONV2\S*)\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', t)
        r = dict(device=lbl, T=T, clk_target=clk, clk_est=float(est.group(2)) if est else None,
                 lat_min=int(lat.group(1)) if lat else None, lat_max=int(lat.group(2)) if lat else None,
                 bram18=int(tot.group(1)), dsp=int(tot.group(2)), ff=int(tot.group(3)), lut=int(tot.group(4)), uram=int(tot.group(5)),
                 avail=[int(avail.group(i)) for i in range(1, 6)] if avail else None, loops=ii)
        r['ms'] = r['lat_max'] * clk / 1e6 if r['lat_max'] else None
        out.append(r)
        print(f"{lbl:14s} T={T:2d} est.clk={r['clk_est']} ns  lat={r['lat_max']} cyc = {r['ms']:.2f} ms  BRAM18={r['bram18']} DSP={r['dsp']} FF={r['ff']} LUT={r['lut']} URAM={r['uram']}  loops={ii[:3]}")
json.dump(out, open('hls/sweep.json', 'w'), indent=1)
