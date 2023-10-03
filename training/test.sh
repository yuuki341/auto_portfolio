#!/bin/bash

UNIT=(64 2048)
SEQ=(1 5 10 20 50)
BATCH=(10 100 500 1000)
iter=0
for u in ${UNIT[@]}
do
for b in ${BATCH[@]}
do
for s in ${SEQ[@]}
do
iter=$(($iter + 1))
python mc_LSTM.py -b $b -l $s -f $iter -u $u 
done
done
done