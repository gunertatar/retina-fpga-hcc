import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Temel Yapı Taşı: Çift Konvolüsyon Bloğu
# ------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    [Conv2d → BatchNorm → ReLU] x 2

    FPGA notu:
        - kernel_size=3 sabit tutuldu (HLS'de 3x3 pencere donanımı standart)
        - padding=1 ile boyut korunuyor (HLS'de sınır dolgusu hesabı kolaylaşır)
        - BatchNorm eğitim sonrası Conv'a katlanacak (fold) → FPGA'da saf Conv kalır
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------
# Encoder Bloğu: DoubleConv + MaxPool
# ------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # skip connection için sakla
        down = self.pool(skip)
        return skip, down


# ------------------------------------------------------------------
# Decoder Bloğu: Upsample + Skip Concat + DoubleConv
# ------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    FPGA notu:
        - Bilinear upsample yerine ConvTranspose2d kullandık
        - ConvTranspose2d HLS'de parametre olarak tanımlanabilir
        - Bilinear HLS'de kayan nokta hesabı gerektirir
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # in_ch = out_ch (skip) + out_ch (up)

    def forward(self, x, skip):
        x = self.up(x)
        # Boyut uyuşmazlığı durumunda kırp
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------
# Küçük U-Net — FPGA Uyumlu
# ------------------------------------------------------------------
class LightUNet(nn.Module):
    """
    Orijinal U-Net'in sadeleştirilmiş versiyonu.

    Kanal sayıları kasıtlı olarak küçük tutuldu:
        - 16 → 32 → 64 → 128  (orijinal: 64→128→256→512)
        - Parametre sayısı: ~490K  (orijinal: ~31M)
        - 4GB VRAM ile batch=32 rahat çalışır
        - FPGA'da fixed-point'e dönüştürme için ideal boyut

    Giriş : [B, 1, 64, 64]   (tek kanal — yeşil kanal)
    Çıkış  : [B, 1, 64, 64]   (sigmoid — piksel başına damar olasılığı)
    """

    def __init__(self, in_channels=1, base_ch=16):
        super().__init__()

        # --- Encoder (Aşağı yol) ---
        self.enc1 = EncoderBlock(in_channels, base_ch)       # 64→32, ch:16
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)   # 32→16, ch:32
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)   # 16→8,  ch:64

        # --- Köprü (Bottleneck) ---
        self.bridge = DoubleConv(base_ch * 4, base_ch * 8)   # 8x8,   ch:128

        # --- Decoder (Yukarı yol) ---
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4)   # 8→16,  ch:64
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2)   # 16→32, ch:32
        self.dec1 = DecoderBlock(base_ch * 2, base_ch)       # 32→64, ch:16

        # --- Çıkış Katmanı ---
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)  # 1x1 Conv
        # Sigmoid eğitim sırasında loss içinde (BCEWithLogitsLoss)
        # Çıkarım sırasında manuel uygulayacağız

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck
        x = self.bridge(x)

        # Decoder
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Çıkış (ham logit — sigmoid dışarıda uygulanacak)
        return self.out_conv(x)


# ------------------------------------------------------------------
# Kayıp Fonksiyonu: BCE + Dice Kombinasyonu
# ------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    """
    Retina segmentasyonunda kullanılan standart kayıp fonksiyonu.

    Neden Dice?
        - Damarlar görüntünün küçük bir kısmını kaplar (~10%)
        - Sadece BCE kullansak model "her şeyi arka plan yap" diyebilir
        - Dice kayıp bu dengesizliği düzeltir

    alpha: BCE ağırlığı
    beta : Dice ağırlığı
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * bce + self.beta * dice


# ------------------------------------------------------------------
# Model Özeti ve Parametre Sayısı
# ------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LightUNet(in_channels=1, base_ch=16)
    criterion = BCEDiceLoss()

    # Test geçişi
    x   = torch.randn(4, 1, 64, 64)   # 4 görüntülük batch
    lbl = torch.randint(0, 2, (4, 1, 64, 64)).float()

    out  = model(x)
    loss = criterion(out, lbl)
    model = LightUNet(in_channels=1, base_ch=16)

    print("Parametre sayısı:", count_parameters(model))
    print(f"Giriş boyutu  : {x.shape}")
    print(f"Çıkış boyutu  : {out.shape}")
    print(f"Kayıp değeri  : {loss.item():.4f}")
    print(f"Parametre sayısı: {count_parameters(model):,}")
    print("model.py başarıyla çalıştı!")
