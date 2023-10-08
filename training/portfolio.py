#import pandas as pd
import random
import math
from gurobipy import Model, quicksum, GRB, multidict
from scipy.stats import norm
import numpy as np
import argparse
import csv
import inspect
from evaluator import evaluate, get_top_n_stock, model_const, cal_std_cov, evaluate_portfolio,_markowitz
from load_data import load_EOD_data


"""
csvファイルを読み込む
行が株式の数、列がmcdropoutしたサンプル数
"""
def read_stock_price(args):
    stock_std_r = []
    with open(args.path) as f:
        reader = csv.reader(f)
        stock_sample = [list(map(float,row)) for row in reader]
        stock_sample = np.array(stock_sample)
    return stock_sample

def main(args):
    stock_sample = read_stock_price(args)
    top_n_list, top_n_list_index = get_top_n_stock(args.num, stock_sample)
    std_list, sigma = cal_std_cov(args.num, stock_sample, top_n_list, top_n_list_index)
    model_weight = model_const(args.alpha, args.num, top_n_list, std_list, sigma)
    #eod_data, mask_data, gt_data, price_data = \
    #        load_EOD_data(data_path, market_name, self.tickers, steps)
    #evaluate_portfolio(cur_test_pred, cur_test_gt, cur_test_topn_index, model_weight, fname=args.file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',help='path of sample data',
                        default='../sample_1000/mcdropout_day0.csv')
    parser.add_argument('-a', '--alpha',help='acceptance threshold',
                        default=1.001,type= float )
    parser.add_argument('-n', '--num',help='Top n stock',
                        default=5,type = int)
    parser.add_argument('-f', help='decide final csv file name',
                        default="data")

    args = parser.parse_args()
    main(args)
