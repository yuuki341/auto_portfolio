#!/bin/bash

UNIT=(256 512 1024 2048)
SEQ=(30 40 50)
DROP=(0.2 0.5 0.8)
iter=0
for u in ${UNIT[@]}
do
for d in ${DROP[@]}
do
for s in ${SEQ[@]}
do
iter=$(($iter + 1))
python mc_LSTM.py -l $s -f $iter -u $u -d_p $d -m NYSE
python mc_LSTM.py -l $s -f $iter -u $u -d_p $d -m NASDAQ
done
done
done