xmutil unloadapp >/dev/null 2>&1; xmutil loadapp retina >/dev/null 2>&1
python3 /home/root/retina/devmem.py 0xFF5E00C0 0x01010600 >/dev/null   # PL0 = IOPLL(1500)/6 = 250 MHz
