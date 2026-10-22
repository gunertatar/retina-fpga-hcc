/**
 * preprocess.h — Güncellenmiş AXI Stream arayüzü
 */

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define ROWS 64
#define COLS 64

typedef ap_axiu<8, 1, 1, 1> axis_pixel_t;

void preprocess(
    hls::stream<axis_pixel_t> &in_img,
    hls::stream<axis_pixel_t> &out_img
);

#endif
