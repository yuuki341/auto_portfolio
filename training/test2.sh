#!/bin/bash

nas=(5 5 5 10 10)
ny=(6 14 12 8 10)

for e in `seq 0 4`
do
python mc_LSTM.py --seed $e -e ${ny[e]} -f NYSE_${e} -m NYSE
python mc_LSTM.py --seed $e -e ${nas[e]} -f NASDAQ_${e}
done
