/**
 * preprocess.cpp — Gaussian Blur + Kontrast Normalizasyon
 * AXI Stream arayuzu (TLAST dahil)
 *
 * Duzeltmeler:
 * - min/max hesabina tum pikseller dahil edildi (kenar pikseller de)
 * - Gaussian filtresi uygulanamayan kenar pikseller ham deger olarak temp'e yaziliyor
 *   ama min/max hesabina yine de giriyor (daha tutarli normalizasyon)
 * - NORM adimi ayni kaliyor
 */

#include "preprocess.h"

static const int GAUSS[3][3] = {
    {16, 32, 16},
    {32, 64, 32},
    {16, 32, 16}
};
// Toplam kernel agirlik: 256 → >> 8 ile normalize

void preprocess(
    hls::stream<axis_pixel_t> &in_img,
    hls::stream<axis_pixel_t> &out_img
)
{
#pragma HLS INTERFACE axis port=in_img
#pragma HLS INTERFACE axis port=out_img
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Line buffer: 3 satir x COLS piksel
    ap_uint<8> line_buf[3][COLS];
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    // Tam cikis tamponu
    ap_uint<8> temp[ROWS][COLS];
#pragma HLS ARRAY_PARTITION variable=temp cyclic factor=4 dim=2

    int min_val = 255;
    int max_val = 0;

    // ----------------------------------------------------------------
    // AŞAMA 1: Gaussian Filtre
    // Kenar pikseller (r<2 veya c sinirlari) ham deger olarak yazilir
    // min/max her piksel icin guncellenir
    // ----------------------------------------------------------------
    GAUSS_ROW: for (int r = 0; r < ROWS; r++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        GAUSS_COL: for (int c = 0; c < COLS; c++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE II=1

            // Line buffer'i kaydır
            line_buf[0][c] = line_buf[1][c];
            line_buf[1][c] = line_buf[2][c];

            // Yeni piksel oku
            axis_pixel_t px_in = in_img.read();
            line_buf[2][c] = px_in.data;

            ap_uint<8> result;

            if (r < 2 || c < 1 || c >= COLS - 1) {
                // Kenar piksel: Gaussian uygulanamaz, ham deger kullan
                result = line_buf[2][c];
            } else {
                // 3x3 Gaussian uygula
                int sum = 0;
                for (int kr = 0; kr < 3; kr++) {
#pragma HLS UNROLL
                    for (int kc = 0; kc < 3; kc++) {
#pragma HLS UNROLL
                        sum += (int)line_buf[kr][c - 1 + kc] * GAUSS[kr][kc];
                    }
                }
                result = (ap_uint<8>)(sum >> 8);
            }

            temp[r][c] = result;

            // Min/max guncelle (tum pikseller dahil)
            int val = (int)result;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    // ----------------------------------------------------------------
    // AŞAMA 2: Kontrast Normalizasyon [0, 255]
    // ----------------------------------------------------------------
    int range = max_val - min_val;
    if (range == 0) range = 1;  // Tek renk goruntu koruması
    // scale_factor = 255 * 256 / range  (>> 8 ile kullanilacak)
    int scale_factor = (255 * 256) / range;

    int out_count = 0;
    int total_out = ROWS * COLS;

    NORM_ROW: for (int r = 0; r < ROWS; r++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        NORM_COL: for (int c = 0; c < COLS; c++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE II=1

            int val = ((int)temp[r][c] - min_val) * scale_factor;
            val = val >> 8;
            if (val < 0)   val = 0;
            if (val > 255) val = 255;

            axis_pixel_t px_out;
            px_out.data = (ap_uint<8>)val;
            out_count++;
            px_out.last = (out_count == total_out) ? 1 : 0;
            px_out.keep = 0xFF;
            px_out.strb = 0xFF;
            out_img.write(px_out);
        }
    }
}
