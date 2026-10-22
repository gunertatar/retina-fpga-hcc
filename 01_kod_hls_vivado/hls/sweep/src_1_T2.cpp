/**
 * conv_ip_kv260.cpp — KV260 versiyonu
 * ICH_TILE=32, 300MHz, BRAM-safe
 *
 * BRAM SORUNU ÇÖZÜMÜ:
 *   w2_l cyclic partition KALDIRILDI
 *   Bunun yerine w2_l ROM olarak URAM'a map edildi
 *   mid_buf cyclic factor=8 korundu (okuma için yeterli)
 *
 * KV260 kaynakları:
 *   DSP:  1248  → ICH_TILE=32: 32×9 = 288 DSP ✓
 *   BRAM: 288   → mid_buf(32×64×64×2B) = 256KB → ~114 BRAM18 ✓
 *   URAM: 64    → w2_l(32×32×3×3×2B)  = 18KB  → 1 URAM ✓
 */

#include "conv_ip.h"

#define ICH_TILE 2

void conv_ip(
    hls::stream<axis_pixel_t>  &in_img,
    hls::stream<axis_fixed_t>  &out_img
)
{
#pragma HLS INTERFACE axis         port=in_img
#pragma HLS INTERFACE axis         port=out_img
#pragma HLS INTERFACE ap_ctrl_none port=return

    // ── Giriş tamponu ─────────────────────────────────────────
    fixed_t img_buf[ROWS][COLS];
#pragma HLS ARRAY_PARTITION variable=img_buf cyclic factor=3 dim=2

    READ: for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II=1
            axis_pixel_t px = in_img.read();
            img_buf[r][c] = (fixed_t)((int)(px.data) * SCALE / 255);
        }
    }

    // ── Ağırlık tamponları ────────────────────────────────────
    fixed_t w1_l[MID_CH][KERNEL][KERNEL];
    fixed_t w2_l[OUT_CH][MID_CH][KERNEL][KERNEL];
    fixed_t b1_l[MID_CH];
    fixed_t b2_l[OUT_CH];

    // w1: kernel boyutlarında complete partition (9 paralel MAC)
#pragma HLS ARRAY_PARTITION variable=w1_l complete dim=2
#pragma HLS ARRAY_PARTITION variable=w1_l complete dim=3
#pragma HLS ARRAY_PARTITION variable=b1_l complete
#pragma HLS ARRAY_PARTITION variable=b2_l complete

    // w2_l: ICH_TILE=32 için dim=2 cyclic factor=32
    // URAM'a map et — BRAM yerine URAM kullanır (64 URAM var KV260'ta)
#pragma HLS ARRAY_PARTITION variable=w2_l cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE variable=w2_l type=RAM_2P impl=URAM

    // ── Ağırlık kopyalama ─────────────────────────────────────
    COPY_W1: for (int och = 0; och < MID_CH; och++)
        for (int kr = 0; kr < KERNEL; kr++)
            for (int kc = 0; kc < KERNEL; kc++) {
#pragma HLS PIPELINE II=1
                w1_l[och][kr][kc] = CONV1_W[och*KERNEL*KERNEL + kr*KERNEL + kc];
            }

    COPY_B1: for (int i = 0; i < MID_CH; i++) {
#pragma HLS PIPELINE II=1
        b1_l[i] = CONV1_B[i];
    }

    COPY_W2: for (int och = 0; och < OUT_CH; och++)
        for (int ich = 0; ich < MID_CH; ich++)
            for (int kr = 0; kr < KERNEL; kr++)
                for (int kc = 0; kc < KERNEL; kc++) {
#pragma HLS PIPELINE II=1
                    w2_l[och][ich][kr][kc] =
                        CONV2_W[och*MID_CH*KERNEL*KERNEL
                               + ich*KERNEL*KERNEL
                               + kr*KERNEL + kc];
                }

    COPY_B2: for (int i = 0; i < OUT_CH; i++) {
#pragma HLS PIPELINE II=1
        b2_l[i] = CONV2_B[i];
    }

    // ── Ara tampon ────────────────────────────────────────────
    // ICH_TILE=32 → 32 paralel okuma → cyclic factor=32
    fixed_t mid_buf[MID_CH][ROWS][COLS];
#pragma HLS ARRAY_PARTITION variable=mid_buf cyclic factor=2 dim=1
#pragma HLS BIND_STORAGE variable=mid_buf type=RAM_2P impl=BRAM

    // ── PASS 1 — CONV1 ────────────────────────────────────────
    CONV1_OCH: for (int och = 0; och < MID_CH; och++) {
        CONV1_ROW: for (int r = 0; r < ROWS; r++) {
            CONV1_COL: for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II=1
                acc_t acc = (acc_t)b1_l[och] << FRAC_BITS;

                for (int kr = 0; kr < KERNEL; kr++) {
#pragma HLS UNROLL
                    for (int kc = 0; kc < KERNEL; kc++) {
#pragma HLS UNROLL
                        int pr = r - PAD + kr;
                        int pc = c - PAD + kc;
                        fixed_t pval = 0;
                        if (pr >= 0 && pr < ROWS && pc >= 0 && pc < COLS)
                            pval = img_buf[pr][pc];
                        acc += (acc_t)pval * (acc_t)w1_l[och][kr][kc];
                    }
                }

                acc_t res = acc >> FRAC_BITS;
                fixed_t v;
                if      (res <= 0)    v = 0;
                else if (res > 32767) v = 32767;
                else                  v = (fixed_t)res;
                mid_buf[och][r][c] = v;
            }
        }
    }

    // ── PASS 2 — CONV2 (ICH_TILE=32, tree reduction) ─────────
    int total_out = OUT_CH * ROWS * COLS;
    int out_count = 0;

    CONV2_OCH: for (int och = 0; och < OUT_CH; och++) {
        CONV2_ROW: for (int r = 0; r < ROWS; r++) {
            CONV2_COL: for (int c = 0; c < COLS; c++) {
#pragma HLS LOOP_FLATTEN
                acc_t acc = (acc_t)b2_l[och] << FRAC_BITS;

                CONV2_KR: for (int kr = 0; kr < KERNEL; kr++) {
                    CONV2_KC: for (int kc = 0; kc < KERNEL; kc++) {
                        int pr = r - PAD + kr;
                        int pc = c - PAD + kc;
                        bool valid = (pr >= 0 && pr < ROWS &&
                                      pc >= 0 && pc < COLS);

                        // ICH_TILE=32: tek grup, tam paralel
                        CONV2_ICH: for (int ich = 0; ich < MID_CH; ich += ICH_TILE) {
#pragma HLS PIPELINE II=1

                            // 32 bağımsız kısmi akümülatör
                            acc_t acc_part[ICH_TILE];
#pragma HLS ARRAY_PARTITION variable=acc_part complete

                            // 32 paralel çarpım
                            for (int t = 0; t < ICH_TILE; t++) {
#pragma HLS UNROLL
                                fixed_t mval = valid ?
                                    mid_buf[ich+t][pr][pc] : (fixed_t)0;
                                acc_part[t] = (acc_t)mval *
                                              (acc_t)w2_l[och][ich+t][kr][kc];
                            }

                            // Tree reduction (same structure as the deployed source)
acc_t lv1[1];
#pragma HLS ARRAY_PARTITION variable=lv1 complete
for (int t = 0; t < 1; t++) {
#pragma HLS UNROLL
    lv1[t] = acc_part[t*2] + acc_part[t*2+1];
}
acc += lv1[0];
                        }
                    }
                }

                // ReLU + stream
                acc_t res = acc >> FRAC_BITS;
                fixed_t out_val;
                if      (res <= 0)    out_val = 0;
                else if (res > 32767) out_val = 32767;
                else                  out_val = (fixed_t)res;

                axis_fixed_t px_out;
                px_out.data = out_val;
                out_count++;
                px_out.last = (out_count == total_out) ? 1 : 0;
                px_out.keep = 0xFF;
                px_out.strb = 0xFF;
                out_img.write(px_out);
            }
        }
    }
}
