#!/bin/sh
cd /home/root/retina
for p in $(pidof -x bokeh) ; do kill $p; done; pkill -f kria-dashboard 2>/dev/null
sh load.sh
NLAT=50 python3 board_run.py kv260 latency > latency.log 2>&1
sleep 20
python3 board_run.py kv260 power > power.log 2>&1
python3 board_run.py kv260 image > image.log 2>&1
PS_THREADS=1 NLAT=10 python3 board_run.py kv260 latency > latency_1thread.log 2>&1
echo CHAIN_DONE > chain.done
