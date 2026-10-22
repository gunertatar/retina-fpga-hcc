import os, glob, numpy as np, torch, torch.nn.functional as F, sys
from PIL import Image
sys.path.insert(0, 'model')
from model import LightUNet
DS = 'datasets'
DRV = 'datasets/DRIVE_official'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path='weights/best_model.pth'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state'] if 'model_state' in ck else ck
    m = LightUNet(1, 32); m.load_state_dict(sd); return m.eval().to(DEV), sd

def fov_from_rgb(rgb):
    r = rgb[..., 0].astype(np.float32)
    m = r > 0.1 * r.max() if r.max() > 0 else np.ones(r.shape, bool)
    from scipy.ndimage import binary_erosion, binary_fill_holes
    m = binary_fill_holes(m); return binary_erosion(m, iterations=3)

def load_split(name, split):
    """returns list of dict(id, rgb uint8 HxWx3, gt bool, fov bool)"""
    out = []
    if name == 'DRIVE':
        if split == 'test':
            for i in range(1, 21):
                rgb = np.array(Image.open(f'{DS}/DRIVE-digital-retinal-images-for-vessel-extraction/val/input/{i:02d}.tif').convert('RGB'))
                gt = np.array(Image.open(f'{DS}/DRIVE-digital-retinal-images-for-vessel-extraction/val/label/{i:02d}_manual1.png').convert('L')) > 128
                fov = np.array(Image.open(f'{DRV}/test/mask/{i:02d}_test_mask.gif').convert('L')) > 128
                out.append(dict(id=f'{i:02d}', rgb=rgb, gt=gt, fov=fov))
        else:
            for i in range(21, 41):
                rgb = np.array(Image.open(f'{DRV}/training/images/{i}_training.tif').convert('RGB'))
                gt = np.array(Image.open(f'{DRV}/training/1st_manual/{i}_manual1.gif').convert('L')) > 128
                fov = np.array(Image.open(f'{DRV}/training/mask/{i}_training_mask.gif').convert('L')) > 128
                out.append(dict(id=str(i), rgb=rgb, gt=gt, fov=fov))
        return out
    root = {'STARE': 'STARE-structured-analysis-of-the-retina', 'CHASE': 'CHASE_DB1-retinal-dataset', 'HRF': 'HRF-high-resolution-fundus'}[name]
    sub = 'train' if split == 'train' else 'val'
    imgs = sorted(glob.glob(f'{DS}/{root}/{sub}/input/*'))
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = {'STARE': f'{stem}.ah.ppm', 'CHASE': f'{stem}_1stHO.png', 'HRF': f'{stem}.tif'}[name]
        rgb = np.array(Image.open(p).convert('RGB'))
        gt = np.array(Image.open(f'{DS}/{root}/{sub}/label/{lab}').convert('L')) > 128
        if name == 'HRF':  # 3504x2336 -> half resolution (common practice)
            H, W = rgb.shape[:2]
            rgb = np.array(Image.fromarray(rgb).resize((W // 2, H // 2), Image.BILINEAR))
            gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((W // 2, H // 2), Image.NEAREST)) > 128
        out.append(dict(id=stem, rgb=rgb, gt=gt, fov=fov_from_rgb(rgb)))
    return out

def train_preprocess(rgb):
    g = rgb[:, :, 1].astype(np.float32)
    p2, p98 = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - p2) / (p98 - p2 + 1e-6), 0, 1).astype(np.float32)

def metrics(pred, gt, fov, prob=None):
    p = pred[fov]; g = gt[fov]
    TP = np.sum(p & g); TN = np.sum(~p & ~g); FP = np.sum(p & ~g); FN = np.sum(~p & g); s = 1e-6
    d = dict(Dice=(2*TP+s)/(2*TP+FP+FN+s), IoU=(TP+s)/(TP+FP+FN+s), Sens=(TP+s)/(TP+FN+s),
             Spec=(TN+s)/(TN+FP+s), Acc=(TP+TN)/(TP+TN+FP+FN))
    if prob is not None:
        from sklearn.metrics import roc_auc_score
        d['AUC'] = roc_auc_score(g, prob[fov])
    return d

# ---------------- bit-exact hardware front-end (torch, float64 exact integer arithmetic) ----
def _wrap(a, bits):
    m = float(1 << bits); return a - m * torch.floor((a + m / 2) / m)

class HWFront:
    def __init__(self, q, frac=10, acc_bits=32, act_bits=16, shift_round=False, preprocess=True, wscale_frac=None):
        self.w1 = torch.tensor(q['CONV1_W'].reshape(32, 1, 3, 3), dtype=torch.float64, device=DEV)
        self.b1 = torch.tensor(q['CONV1_B'], dtype=torch.float64, device=DEV)
        self.w2 = torch.tensor(q['CONV2_W'].reshape(32, 32, 3, 3), dtype=torch.float64, device=DEV)
        self.b2 = torch.tensor(q['CONV2_B'], dtype=torch.float64, device=DEV)
        self.frac, self.acc_bits, self.amax = frac, acc_bits, float((1 << (act_bits - 1)) - 1)
        self.shift_round, self.preprocess = shift_round, preprocess
        G = torch.tensor([[16, 32, 16], [32, 64, 32], [16, 32, 16]], dtype=torch.float64, device=DEV)
        self.G = G.view(1, 1, 3, 3)
    def preprocess_ip(self, u8):  # u8: (N,1,64,64) float64 integer valued; exact model of preprocess.cpp
        N, _, H, W = u8.shape
        z = torch.zeros((N, 1, 1, W), dtype=u8.dtype, device=u8.device)
        # the RTL line buffer is not cleared between patches: row "-1" is the last row of the
        # previously streamed patch (only affects output row 2). Patches are streamed in order.
        prev = torch.cat([self.carry.to(u8.device) if getattr(self, 'carry', None) is not None else z[:1], u8[:-1, :, -1:]], 0)
        self.carry = u8[-1:, :, -1:].clone()
        up1 = torch.cat([prev, u8[:, :, :-1]], 2)       # row r-1 (row -1 = previous patch's last row)
        up2 = torch.cat([z, up1[:, :, :-1]], 2)         # row r-2
        up3 = torch.cat([z, up2[:, :, :-1]], 2)         # row r-3
        G = [[16, 32, 16], [32, 64, 32], [16, 32, 16]]
        newr = [up2, up1, u8]; oldr = [up3, up2, up1]
        s = 0
        for kr in range(3):
            s = s + G[kr][0] * newr[kr][..., 0:W-2] + G[kr][1] * newr[kr][..., 1:W-1] + G[kr][2] * oldr[kr][..., 2:W]
        T = u8.clone()
        T[:, :, 2:, 1:-1] = torch.floor(s[:, :, 2:] / 256)
        mn = T.amin(dim=(2, 3), keepdim=True); mx = T.amax(dim=(2, 3), keepdim=True)
        rng = torch.clamp(mx - mn, min=1); sf = torch.floor(255 * 256 / rng)
        return torch.clamp(torch.floor((T - mn) * sf / 256), 0, 255)
    def post(self, acc):
        if self.shift_round: acc = acc + (1 << (self.frac - 1))
        return torch.clamp(torch.floor(acc / (1 << self.frac)), 0, self.amax)
    def __call__(self, u8):
        x = u8.to(DEV, torch.float64)
        if self.preprocess: x = self.preprocess_ip(x)
        x = torch.floor(x * (1 << self.frac) / 255)
        a = F.conv2d(x, self.w1, padding=1) + self.b1.view(1, -1, 1, 1) * (1 << self.frac)
        mid = self.post(_wrap(a, self.acc_bits))
        a = F.conv2d(mid, self.w2, padding=1) + self.b2.view(1, -1, 1, 1) * (1 << self.frac)
        return self.post(_wrap(a, self.acc_bits))  # integer activations at scale 2^frac

def rest_forward(m, s1):
    x = F.max_pool2d(s1, 2)
    s2, x = m.enc2(x); s3, x = m.enc3(x); x = m.bridge(x)
    x = m.dec3(x, s3); x = m.dec2(x, s2); x = m.dec1(x, s1)
    return m.out_conv(x)

def patch_grid(H, W, ps=64, st=32):
    rs = list(range(0, max(H - ps, 0) + 1, st)); cs = list(range(0, max(W - ps, 0) + 1, st))
    if rs[-1] != H - ps: rs.append(H - ps)
    if cs[-1] != W - ps: cs.append(W - ps)
    return [(r, c) for r in rs for c in cs]

@torch.no_grad()
def infer_image(m, rgb, mode='fp32', hw=None, bs=256, ps=64, st=32):
    """mode: 'fp32' (float network) | 'hw' (hw front-end bit-exact + float PS layers)
             | 'fp32_pre' (preprocess_ip emulated, float network)"""
    x = train_preprocess(rgb); H, W = x.shape
    grid = patch_grid(H, W, ps, st)
    prob = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
    u8 = np.floor(x * 255).astype(np.float64)
    for i in range(0, len(grid), bs):
        g = grid[i:i + bs]
        if mode == 'fp32':
            t = torch.tensor(np.stack([x[r:r+ps, c:c+ps] for r, c in g])[:, None], device=DEV)
            o = m(t)
        else:
            tu = torch.tensor(np.stack([u8[r:r+ps, c:c+ps] for r, c in g])[:, None], device=DEV)
            if mode == 'fp32_pre':
                pre = hw.preprocess_ip(tu.double()) if hw.preprocess else tu.double()
                o = m((pre / 255).float())
            else:
                s1 = (hw(tu) / (1 << hw.frac)).float()
                o = rest_forward(m, s1)
        p = torch.sigmoid(o)[:, 0].cpu().numpy()
        for (r, c), pp in zip(g, p):
            prob[r:r+ps, c:c+ps] += pp; cnt[r:r+ps, c:c+ps] += 1
    return prob / np.maximum(cnt, 1)
