import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class DRIVEDataset(Dataset):
    """
    DRIVE Retina Damar Segmentasyon Dataset'i

    Klasör yapısı:
        data/DRIVE/training/images/       -> .tif görüntüler
        data/DRIVE/training/1st_manual/   -> ground truth maskeler
        data/DRIVE/training/mask/         -> FOV maskeleri

    Args:
        root_dir  : DRIVE klasörünün yolu  (örn. "data/DRIVE")
        split     : "training" veya "test"
        patch_size: kırpılacak yama boyutu (FPGA uyumu için 64 veya 128 önerilir)
        n_patches : eğitim için her görüntüden kaç yama alınsın
        augment   : veri artırma uygulansın mı?
    """

    def __init__(self, root_dir, split="training",
                 patch_size=64, n_patches=20, augment=True):
        self.split      = split
        self.patch_size = patch_size
        self.n_patches  = n_patches
        self.augment    = augment and (split == "training")

        img_dir    = os.path.join(root_dir, split, "images")
        label_dir  = os.path.join(root_dir, split, "1st_manual")
        mask_dir   = os.path.join(root_dir, split, "mask")

        # Dosya isimlerini sırala (tekrarlanabilirlik için)
        img_files   = sorted([f for f in os.listdir(img_dir)
                               if f.endswith(".tif")])
        label_files = sorted([f for f in os.listdir(label_dir)
                               if f.endswith(".gif") or f.endswith(".png")])
        mask_files  = sorted([f for f in os.listdir(mask_dir)
                               if f.endswith(".gif") or f.endswith(".png")])

        self.images  = []
        self.labels  = []
        self.fov_masks = []

        for img_f, lbl_f, msk_f in zip(img_files, label_files, mask_files):
            # Görüntü: RGB → numpy
            img = np.array(Image.open(os.path.join(img_dir, img_f)).convert("RGB"),
                           dtype=np.float32)

            # Etiket (ground truth damar maskesi): ikili (0/1)
            lbl = np.array(Image.open(os.path.join(label_dir, lbl_f)).convert("L"),
                           dtype=np.float32)
            lbl = (lbl > 128).astype(np.float32)

            # FOV maskesi (gözün dışını görmezden gel)
            msk = np.array(Image.open(os.path.join(mask_dir, msk_f)).convert("L"),
                           dtype=np.float32)
            msk = (msk > 128).astype(np.float32)

            self.images.append(img)
            self.labels.append(lbl)
            self.fov_masks.append(msk)

        # Eğitimde yama tabanlı çalışıyoruz
        # Testte tam görüntü kullanılacak
        if split == "training":
            self.patches = self._extract_patches()
        else:
            self.patches = None

    # ------------------------------------------------------------------
    # Ön İşleme
    # ------------------------------------------------------------------
    def _preprocess(self, img):
        """
        1. Yeşil kanal: retina damarları yeşil kanalda en belirgin
        2. CLAHE benzeri kontrast normalizasyonu (basit histogram gerilimi)
        3. [0, 1] normalizasyon
        """
        green = img[:, :, 1]                     # Yeşil kanal al
        # Histogram gerilimi (CLAHE'ye basit alternatif)
        p2, p98 = np.percentile(green, 2), np.percentile(green, 98)
        green = np.clip((green - p2) / (p98 - p2 + 1e-6), 0, 1)
        return green.astype(np.float32)

    # ------------------------------------------------------------------
    # Yama Çıkarımı (Eğitim)
    # ------------------------------------------------------------------
    def _extract_patches(self):
        patches = []
        ps = self.patch_size

        for img, lbl, msk in zip(self.images, self.labels, self.fov_masks):
            processed = self._preprocess(img)
            H, W = processed.shape
            count = 0
            attempts = 0

            while count < self.n_patches and attempts < self.n_patches * 10:
                attempts += 1
                # Rastgele konum seç
                r = random.randint(0, H - ps - 1)
                c = random.randint(0, W - ps - 1)

                # FOV maskesi içinde olsun (gözün dışı olmasın)
                msk_patch = msk[r:r+ps, c:c+ps]
                if msk_patch.mean() < 0.9:
                    continue

                img_patch = processed[r:r+ps, c:c+ps]
                lbl_patch = lbl[r:r+ps, c:c+ps]
                patches.append((img_patch, lbl_patch))
                count += 1

        return patches

    # ------------------------------------------------------------------
    # Veri Artırma
    # ------------------------------------------------------------------
    def _augment(self, img_t, lbl_t):
        """Yatay çevirme, dikey çevirme, 90° dönme"""
        if random.random() > 0.5:
            img_t = TF.hflip(img_t)
            lbl_t = TF.hflip(lbl_t)
        if random.random() > 0.5:
            img_t = TF.vflip(img_t)
            lbl_t = TF.vflip(lbl_t)
        k = random.randint(0, 3)
        img_t = torch.rot90(img_t, k, dims=[-2, -1])
        lbl_t = torch.rot90(lbl_t, k, dims=[-2, -1])
        return img_t, lbl_t

    # ------------------------------------------------------------------
    # PyTorch Dataset Arayüzü
    # ------------------------------------------------------------------
    def __len__(self):
        if self.patches:
            return len(self.patches)
        return len(self.images)

    def __getitem__(self, idx):
        if self.patches:
            img_np, lbl_np = self.patches[idx]
        else:
            img_np  = self._preprocess(self.images[idx])
            lbl_np  = self.labels[idx]

        # numpy → torch tensor  [1, H, W]
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        lbl_t = torch.from_numpy(lbl_np).unsqueeze(0)

        if self.augment:
            img_t, lbl_t = self._augment(img_t, lbl_t)

        return img_t, lbl_t


# ------------------------------------------------------------------
# Hızlı Test
# ------------------------------------------------------------------
if __name__ == "__main__":
    dataset = DRIVEDataset(
        root_dir="data/DRIVE",
        split="training",
        patch_size=64,
        n_patches=20,
        augment=True
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    imgs, lbls = next(iter(loader))
    print(f"Görüntü batch boyutu : {imgs.shape}")   # [8, 1, 64, 64]
    print(f"Etiket batch boyutu  : {lbls.shape}")   # [8, 1, 64, 64]
    print(f"Görüntü min/max      : {imgs.min():.3f} / {imgs.max():.3f}")
    print(f"Etiket benzersiz     : {lbls.unique()}")
    print(f"Toplam yama sayısı   : {len(dataset)}")
    print("dataset.py başarıyla çalıştı!")
