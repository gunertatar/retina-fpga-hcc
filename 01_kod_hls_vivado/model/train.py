import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import DRIVEDataset
from model import LightUNet, BCEDiceLoss

# ------------------------------------------------------------------
# Konfigürasyon — buradan ayarla
# ------------------------------------------------------------------
CFG = {
    "data_root"  : "data/DRIVE",
    "patch_size" : 64,
    "n_patches"  : 80,        # Her görüntüden 40 yama → 20*40 = 800 yama/epoch
    "batch_size" : 16,
    "epochs"     : 150,
    "lr"         : 1e-3,
    "base_ch"    : 32,        # U-Net kanal sayısı
    "save_dir"   : "models",
    "device"     : "cuda" if torch.cuda.is_available() else "cpu",
}


# ------------------------------------------------------------------
# Metrik: Dice Skoru
# ------------------------------------------------------------------
def dice_score(pred_logits, targets, threshold=0.5):
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(2, 3))
    union        = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ------------------------------------------------------------------
# Tek Epoch Eğitim
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, lbls in tqdm(loader, desc="Eğitim", leave=False):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(outputs.detach(), lbls)

    n = len(loader)
    return total_loss / n, total_dice / n


# ------------------------------------------------------------------
# Doğrulama
# ------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, lbls in tqdm(loader, desc="Doğrulama", leave=False):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        outputs    = model(imgs)
        loss       = criterion(outputs, lbls)
        total_loss += loss.item()
        total_dice += dice_score(outputs, lbls)

    n = len(loader)
    return total_loss / n, total_dice / n


# ------------------------------------------------------------------
# Ana Eğitim Döngüsü
# ------------------------------------------------------------------
def main():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device(CFG["device"])
    print(f"Kullanılan cihaz: {device}")

    # Dataset
    train_ds = DRIVEDataset(
        root_dir   = CFG["data_root"],
        split      = "training",
        patch_size = CFG["patch_size"],
        n_patches  = CFG["n_patches"],
        augment    = True,
    )
    # Test setini doğrulama olarak kullanıyoruz
    val_ds = DRIVEDataset(
        root_dir   = CFG["data_root"],
        split      = "training",
        patch_size = CFG["patch_size"],
        n_patches  = 10,
        augment    = False,
    )

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Eğitim yama sayısı   : {len(train_ds)}")
    print(f"Doğrulama yama sayısı: {len(val_ds)}")

    # Model
    model     = LightUNet(in_channels=1, base_ch=CFG["base_ch"]).to(device)
    criterion = BCEDiceLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

    # Öğrenme oranı azaltma — 10 epoch iyileşme yoksa lr/2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    best_dice  = 0.0
    history    = {"train_loss": [], "train_dice": [],
                  "val_loss":   [], "val_dice":   []}

    print(f"\n{'='*55}")
    print(f"  Eğitim Başlıyor — {CFG['epochs']} Epoch")
    print(f"{'='*55}\n")

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_dice = validate(model, val_loader, criterion, device)

        scheduler.step(vl_dice)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(vl_loss)
        history["val_dice"].append(vl_dice)

        print(f"Epoch [{epoch:03d}/{CFG['epochs']}]  "
              f"Eğitim Kayıp: {tr_loss:.4f}  Dice: {tr_dice:.4f}  |  "
              f"Doğrulama Kayıp: {vl_loss:.4f}  Dice: {vl_dice:.4f}")

        # En iyi modeli kaydet
        if vl_dice > best_dice:
            best_dice = vl_dice
            save_path = os.path.join(CFG["save_dir"], "best_model.pth")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_dice"   : vl_dice,
                "cfg"        : CFG,
            }, save_path)
            print(f"  ✓ En iyi model kaydedildi → Dice: {best_dice:.4f}")

    # Eğitim geçmişini kaydet
    np.save("outputs/history.npy", history)
    print(f"\nEğitim tamamlandı! En iyi Doğrulama Dice: {best_dice:.4f}")
    print(f"Model kaydedildi: {os.path.join(CFG['save_dir'], 'best_model.pth')}")


if __name__ == "__main__":
    main()
