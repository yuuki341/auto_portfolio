import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
from time import time
import csv
from multiprocessing import Pool
import functools

from load_data import load_EOD_data
from evaluator import evaluate, get_top_n_stock, model_const, cal_std_cov, evaluate_portfolio,greedy

stock_sum = 4
topnum = 3
mcdrop_num = 5
a =  np.random.randint(0, 5, (stock_sum, mcdrop_num))
a = np.array([[2,0,1,4,3],
 [0,0,3,3,4],
 [4,0,2,1,1],
 [1,2,2,3,0]])
t = 0

def tet(at,a):
    print(at,np.sum(a))

with Pool() as pool:
    pool.map(functools.partial(tet,at = t),a)
#greedy(topnum, a, mcdrop_num)