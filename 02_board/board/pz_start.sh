#!/bin/sh
# usage: pz_start.sh "<env assignments>" <mode> <logname>   (runs board_run.py with root rights; sudo asks for the board password)
cd "$(dirname "$0")"
sudo -E env $1 XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 board_run.py pynqz1 $2 > $3.log 2>&1
