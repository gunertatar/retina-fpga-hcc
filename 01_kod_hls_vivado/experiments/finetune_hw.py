"""Hardware-aware fine-tuning of the PS-resident layers (enc2 ... out_conv).
The PL front-end (preprocess_ip + conv_ip) is emulated bit-exactly and frozen,
so the deployed bitstream (enc1 weights in ROM) is left unchanged.
Training data: DRIVE training split only (20 images); fixed schedule, no test-set selection."""
import sys, random, numpy as np, torch, time
sys.path.insert(0, 'emulator'); sys.path.insert(0, 'model'); sys.path.insert(0, 'model')
import common, hwemu
from model import BCEDiceLoss
ds = sys.argv[1] if len(sys.argv) > 1 else 'DRIVE'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out = sys.argv[3] if len(sys.argv) > 3 else f'weights/hwaware_ps_{ds}.pth'
base = sys.argv[4] if len(sys.argv) > 4 else 'weights/best_model.pth'
MODE = sys.argv[5] if len(sys.argv) > 5 else 'hw'   # 'hw' (PS layers only, PL front-end frozen) | 'fp32' (all layers, float)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
m, sd = common.load_model(base)
if MODE == 'hw':
    # PL front-end: enc1 ROM of the DEPLOYED bitstream (DRIVE best_model) - never changed
    _, sd0 = common.load_model('weights/best_model.pth'); q, _ = hwemu.fold_enc1(sd0); hw = common.HWFront(q, preprocess=True)
    for p in m.enc1.parameters(): p.requires_grad = False
    params = [p for n, p in m.named_parameters() if not n.startswith('enc1.')]
else:
    params = list(m.parameters())
opt = torch.optim.Adam(params, lr=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
crit = BCEDiceLoss(alpha=0.5, beta=0.5)
tr = common.load_split(ds, 'train')
imgs = [(np.floor(common.train_preprocess(d['rgb']) * 255) if MODE == 'hw' else common.train_preprocess(d['rgb']), d['gt'].astype(np.float32), d['fov']) for d in tr]
def sample(n=80, ps=64):
    X, Y = [], []
    for u8, gt, fov in imgs:
        H, W = u8.shape; k = 0; tries = 0
        while k < n and tries < n * 20:
            tries += 1; r = random.randint(0, H - ps); c = random.randint(0, W - ps)
            if fov[r:r+ps, c:c+ps].mean() < 0.9: continue
            x, y = u8[r:r+ps, c:c+ps], gt[r:r+ps, c:c+ps]
            if random.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < .5: x, y = x[::-1], y[::-1]
            kk = random.randint(0, 3); x, y = np.rot90(x, kk), np.rot90(y, kk)
            X.append(x.copy()); Y.append(y.copy()); k += 1
    return np.stack(X)[:, None], np.stack(Y)[:, None]
t0 = time.time()
for ep in range(EPOCHS):
    X, Y = sample(); idx = np.random.permutation(len(X)); m.train(); L = []
    if MODE == 'hw': m.enc1.eval()
    for i in range(0, len(X), 16):
        b = idx[i:i+16]
        y = torch.tensor(Y[b], device=common.DEV)
        if MODE == 'hw':
            with torch.no_grad(): s1 = (hw(torch.tensor(X[b], dtype=torch.float64)) / 1024).float()
            loss = crit(common.rest_forward(m, s1), y)
        else:
            loss = crit(m(torch.tensor(X[b], dtype=torch.float32, device=common.DEV)), y)
        opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
    sch.step()
    if ep % 5 == 0 or ep == EPOCHS - 1: print(f'ep {ep} loss {np.mean(L):.4f} ({time.time()-t0:.0f}s)', flush=True)
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({'model_state': m.state_dict(), 'note': 'PS layers fine-tuned with bit-exact PL front-end; enc1 identical to base'}, out)
sd2 = m.state_dict(); print('enc1 equals deployed ROM model:', all(torch.equal(sd2[k].cpu(), common.load_model('weights/best_model.pth')[1][k].cpu()) for k in sd2 if k.startswith('enc1.')))
print('saved', out)
