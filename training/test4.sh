#!/bin/bash

nas=(5 5 5 10 10)
ny=(6 14 12 8 10)

for e in `seq 0 4`
do
python mc_LSTM_test.py --seed $e -e 30 -f NYSE_${e} -m NYSE -l 10
python mc_LSTM_test.py --seed $e -e 30 -f NASDAQ_${e} -l 10
done
