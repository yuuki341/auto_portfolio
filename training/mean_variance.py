import argparse
import copy
import numpy as np
import os
import random
import time
import csv
from multiprocessing import Pool
import matplotlib.pyplot as plt
from load_data import load_EOD_data
from evaluator import evaluate, get_top_n_stock, model_const, cal_std_cov, evaluate_portfolio,greedy, cardinary_constrained


if __name__ == "__main__":
    valid_index = 756
    test_index = 1008
    mcdrop_num = 1000
    topnum=5
    gt_data = np.loadtxt("NASDAQ_gt_data.csv", delimiter=",")
    train, test = np.split(gt_data,[test_index],axis=1)
    #train,val,test = np.split(gt_data,[valid_index,test_index],axis=1)
    #index_list, model_weight = greedy(topnum, train, valid_index)
    #index_list, model_weight = greedy(topnum, train, test_index)
    index_list, model_weight = cardinary_constrained(topnum, train, test_index)
    test = test.transpose()
    day_weight = np.zeros([test.shape[0], topnum],dtype=float)
    top_n_list_index_test_days = np.zeros([test.shape[0], topnum], dtype=int)
    for day in range(test.shape[0]):
        if topnum != 1:
            top_n_list_index_test_days[day,:], day_weight[day,:] = index_list, model_weight
        else:
            var_list = np.var(mc_drop_list[day],axis=1)
            top_n_list_index_test_days[day,:] = np.argmin(var_list)
            day_weight[day,:] = 1
    evaluate_portfolio(test, test, top_n_list_index_test_days, day_weight, fname="MV")
    np.savetxt(f'MV_index.csv', top_n_list_index_test_days, delimiter=',')
    np.savetxt(f'MV_weight.csv', day_weight, delimiter=',')

