import sys, numpy as np, torch, json
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model'); import hwemu
board = sys.argv[1]
import glob, os
def load_validate(d):   # one file written by board_run.py, or the two parts stored in this repository
    f = os.path.join(d, 'validate_out.npz')
    parts = [f] if os.path.exists(f) else sorted(glob.glob(os.path.join(d, 'validate_out_part*.npz')))
    return {k: np.concatenate([np.load(x)[k] for x in parts]) for k in ('pre', 'conv')}
V = load_validate(f'results/board_raw/{board}'); P = np.load('board/board_inputs.npz')['val_patches']
sd = torch.load('weights/best_model.pth', map_location='cpu', weights_only=False)['model_state']; q, _ = hwemu.fold_enc1(sd)
pre_ex = conv_ex = conv_given_pre = 0; rows = []; tot = 0
N = int(sys.argv[2]) if len(sys.argv) > 2 else len(P)
for i, p in enumerate(P[:N]):
    prev = P[i - 1][-1] if i > 0 else None
    e_pre = hwemu.preprocess_ip(p, prev_last_row=prev)
    bp = V['pre'][i]; bc = V['conv'][i].astype(np.int64)
    pre_ex += np.array_equal(bp, e_pre)
    if not np.array_equal(bp, e_pre): rows += sorted(set(np.nonzero(bp != e_pre)[0].tolist()))
    c = hwemu.conv_ip(bp, q)   # emulate conv_ip on the board's own preprocess output
    conv_given_pre += np.array_equal(bc, c); tot += c.size
    conv_ex += np.array_equal(bc, hwemu.conv_ip(e_pre, q))
r = dict(n=N, preprocess_bit_exact=int(pre_ex), conv_ip_bit_exact_given_board_pre=int(conv_given_pre), end_to_end_bit_exact=int(conv_ex), mismatch_rows=sorted(set(rows))[:20], values_compared=int(tot))
print(json.dumps(r, indent=1)); json.dump(r, open(f'results/board_raw/{board}/check.json', 'w'), indent=1)
