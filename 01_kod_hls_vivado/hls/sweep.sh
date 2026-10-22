source /tools/Xilinx/Vitis_HLS/2022.1/settings64.sh
cd hls/sweep
for T in 32 8 1 2 4 16; do
  PART=xc7z020clg400-1 CLK=10 URAM=0 T=$T vitis_hls -f run.tcl > log_z7_T$T.txt 2>&1
  PART=xck26-sfvc784-2LV-c CLK=3.8 URAM=1 T=$T vitis_hls -f run.tcl > log_k26_T$T.txt 2>&1
done
echo SWEEP_DONE
