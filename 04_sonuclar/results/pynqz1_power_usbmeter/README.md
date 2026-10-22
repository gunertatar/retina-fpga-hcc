# PYNQ-Z1 board power (Table 11)

The PYNQ-Z1 has no power sensor. The board was supplied over USB from a 5 V source through a
Keweisi USB power meter (display resolution 0.01 V / 0.01 A, charge counter 1 mAh). Each state was
run in a continuous loop with `board/pz_phase.py <idle|pl|ps|full> <seconds>`.

| State | Reading | Power |
|---|---|---|
| idle (bitstream loaded) | 4.96 V, 0.39 A (display) | 1.93 W |
| PL stage only (preprocess + enc1) | 4.96 V, 0.43 A (display) | 2.13 W |
| PS layers only | 4.96 V, 0.43 A (display) | 2.13 W |
| complete pipeline | charge counter 0.214 -> 0.288 Ah in 584 s, 0.456 A at 4.96 V | 2.26 W |

Per-patch times for the energy figures (PL stage 102.4 ms, complete pipeline 5267.6 ms) are from the
power run of `board/board_run.py pynqz1 power`; enc1 in software (382.0 ms) is from Table 10.
