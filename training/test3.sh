#!/bin/bash

UNIT=(256 512 1024)
DROP=(0.2 0.5 0.8)

iter=0
for u in ${UNIT[@]}
do
for d in ${DROP[@]}
do
iter=$(($iter + 1))
python mc_LSTM.py -f $iter -u $u -d_p $d -m NYSE -e 40
iter=$(($iter + 1))
python mc_LSTM.py -f $iter -u $u -d_p $d -m NASDAQ -e 40
done
done