"""
gen_weights.py — BN fold + Q1.10 fixed-point donusum
conv_ip.h icindeki CONV1_W, CONV1_B, CONV2_W, CONV2_B satirlarini gunceller.

Kullanim:
    python3 gen_weights.py --model ../models/best_model.pth --frac 10 --check
    python3 gen_weights.py --model ../models/best_model.pth --frac 10 --header conv_ip.h
"""

import argparse
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def load_and_fold(model_path, frac_bits):
    """
    Modeli yukle, BN fold et, Q1.10 fixed-point'e cevir.
    Donusler: (w1_fixed, b1_fixed, w2_fixed, b2_fixed)
    """
    import torch
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state']

    def get(key):
        if key in state:
            return state[key].numpy().astype(np.float32)
        raise KeyError(f"Key not found in state_dict: {key}")

    def fold_bn(w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        """BN parametrelerini conv agirliklarına fold et."""
        std   = np.sqrt(bn_var + eps)
        scale = bn_w / std                        # her kanal icin scale
        w_fold = w * scale[:, None, None, None]   # (out_ch, in_ch, kH, kW)
        b_fold = bn_b - bn_mean * scale
        return w_fold, b_fold

    # enc1.conv.block:
    #   0 = Conv2d(1->32, 3x3)
    #   1 = BatchNorm2d(32)
    #   2 = ReLU
    #   3 = Conv2d(32->32, 3x3)
    #   4 = BatchNorm2d(32)
    #   5 = ReLU
    w1     = get('enc1.conv.block.0.weight')
    bn1_w  = get('enc1.conv.block.1.weight')
    bn1_b  = get('enc1.conv.block.1.bias')
    bn1_mu = get('enc1.conv.block.1.running_mean')
    bn1_v  = get('enc1.conv.block.1.running_var')

    w2     = get('enc1.conv.block.3.weight')
    bn2_w  = get('enc1.conv.block.4.weight')
    bn2_b  = get('enc1.conv.block.4.bias')
    bn2_mu = get('enc1.conv.block.4.running_mean')
    bn2_v  = get('enc1.conv.block.4.running_var')

    w1f, b1f = fold_bn(w1, bn1_w, bn1_b, bn1_mu, bn1_v)
    w2f, b2f = fold_bn(w2, bn2_w, bn2_b, bn2_mu, bn2_v)

    # Float -> Q1.10 fixed-point (int16)
    scale_q = float(1 << frac_bits)

    def to_fixed(x, name):
        x_scaled = x * scale_q
        # Klipler oncesi istatistik
        n_clip = np.sum(np.abs(x_scaled) > 32767)
        if n_clip > 0:
            print(f"  UYARI: {name} icin {n_clip} deger kliplendi "
                  f"(min={x_scaled.min():.1f}, max={x_scaled.max():.1f})")
        return np.clip(np.round(x_scaled), -32768, 32767).astype(np.int16)

    return (to_fixed(w1f.flatten(), 'CONV1_W'),
            to_fixed(b1f.flatten(), 'CONV1_B'),
            to_fixed(w2f.flatten(), 'CONV2_W'),
            to_fixed(b2f.flatten(), 'CONV2_B'))


def array_to_c(arr, per_row=9):
    """NumPy int16 dizisini C sabit dizisi icin satir basan stringe cevir."""
    lines = []
    for i in range(0, len(arr), per_row):
        chunk = arr[i:i+per_row]
        lines.append("    " + ",".join(str(int(v)) for v in chunk))
    return ",\n".join(lines)


def update_header(header_path, w1, b1, w2, b2):
    """conv_ip.h icindeki 4 sabit diziyi yerinde gunceller."""
    with open(header_path, 'r') as f:
        content = f.read()

    def replace_block(content, array_name, new_data_str):
        # "static const fixed_t NAME[...] = { ... };" bloğunu yakala
        pattern = (r'(static const fixed_t '
                   + re.escape(array_name)
                   + r'\[.*?\]\s*=\s*\{)[^}]*(};)')
        replacement = r'\g<1>\n' + new_data_str + r'\n\g<2>'
        new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
        if n == 0:
            raise ValueError(f"'{array_name}' blogu '{header_path}' icinde bulunamadi! "
                             "Dizi tanimini kontrol edin.")
        return new_content

    content = replace_block(content, 'CONV1_W', array_to_c(w1, per_row=9))
    content = replace_block(content, 'CONV1_B', array_to_c(b1, per_row=16))
    content = replace_block(content, 'CONV2_W', array_to_c(w2, per_row=9))
    content = replace_block(content, 'CONV2_B', array_to_c(b2, per_row=16))

    with open(header_path, 'w') as f:
        f.write(content)
    print(f"[OK] {header_path} guncellendi.")


def print_stats(name, arr):
    print(f"  {name:10s}: {len(arr):5d} eleman | "
          f"min={arr.min():7d}  max={arr.max():7d}  "
          f"ort={arr.mean():.1f}  sifir_olmayan={np.count_nonzero(arr)}")


def main():
    parser = argparse.ArgumentParser(
        description='BN fold + Q1.10 fixed-point agirlik uretici')
    parser.add_argument('--model',  required=True,
                        help='PyTorch checkpoint yolu (best_model.pth)')
    parser.add_argument('--header', default='conv_ip.h',
                        help='Guncellenecek header dosyasi (varsayilan: conv_ip.h)')
    parser.add_argument('--frac',   type=int, default=10,
                        help='Kesirli bit sayisi (varsayilan: 10 → Q1.10)')
    parser.add_argument('--check',  action='store_true',
                        help='Ilk birkac degeri yazdir (debug)')
    args = parser.parse_args()

    print(f"Model yukleniyor : {args.model}")
    print(f"Frac bits        : {args.frac}  (scale = {1 << args.frac})")
    print()

    w1, b1, w2, b2 = load_and_fold(args.model, args.frac)

    print("Agirlik istatistikleri:")
    print_stats('CONV1_W', w1)
    print_stats('CONV1_B', b1)
    print_stats('CONV2_W', w2)
    print_stats('CONV2_B', b2)
    print()

    update_header(args.header, w1, b1, w2, b2)

    if args.check:
        print("\n--- Ilk 9 CONV1_W ---")
        print(w1[:9])
        print("--- Ilk 16 CONV1_B ---")
        print(b1[:16])
        print("--- Ilk 9 CONV2_W ---")
        print(w2[:9])
        print("--- Ilk 16 CONV2_B ---")
        print(b2[:16])


if __name__ == '__main__':
    main()
