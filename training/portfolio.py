#import pandas as pd
import random
import math
from gurobipy import Model, quicksum, GRB, multidict
from scipy.stats import norm
import numpy as np
import argparse
import csv
import inspect

def markowitz(alpha,I, std,sigma,r):
    """markowitz -- simple markowitz model for portfolio optimization.
    Parameters:
        - I: set of items
        - std[i]: standard deviation of item i
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
    Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar(len(I))
    
    portfolio_risk = x @ sigma @ x
    model.setObjective(portfolio_risk, GRB.MINIMIZE)
    model.addConstr(x.sum() == 1, 'budget')
    model.addConstr(r @ x   >= alpha)

    model.__data = x
    return model

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

"""
平均収益率が高い順にソートして、上位n個出力
"""
def get_top_n_stock(args,stock_sample):
    mean_revenue_list = np.mean(stock_sample,axis=1)
    mean_revenue_sort_index = np.argsort(mean_revenue_list)
    max_n_list = []
    max_n_list_index = [] 
    for i in range(args.num):
        max_n = mean_revenue_sort_index[-1 * (i + 1)]
        max_n_list_index.append(max_n)
        max_n_list.append(mean_revenue_list[max_n])
    max_n_list = np.array(max_n_list)
    return max_n_list, max_n_list_index

"""
標準偏差と共分散を計算
"""
def cal_std_cov(args,stock_sample,max_n_list,max_n_list_index):
    rev_stock_sample = []
    for i in range(args.num):
        rev_i = max_n_list_index[i]
        rev_stock_sample.append(stock_sample[rev_i])
    rev_stock_sample = np.array(rev_stock_sample)

    std_list = np.std(rev_stock_sample, axis=1)  
    sigma = np.cov(rev_stock_sample)
    return std_list, sigma

def main(args):
    stock_sample = read_stock_price(args)
    max_n_list, max_n_list_index = get_top_n_stock(args,stock_sample)
    std_list, sigma = cal_std_cov(args,stock_sample,max_n_list,max_n_list_index)

    #収益率を1に正規化
    I = list(range(len(max_n_list)))
    def plus(e):
        return e+1
    pvec = np.vectorize(plus)
    r_list = pvec(max_n_list)
    #エラーにならないように
    if r_list[0] < args.alpha:
        alpha = r_list[0]
    else:
        alpha = args.alpha

    model = markowitz(alpha, I, std_list, sigma, r_list)
    model.optimize()

    x = model.__data
    EPS = 1.0e-6
    print("%5s\t%8s" % ("i", "x[i]"))    
    for i in I:
        print("%5s\t%8g" % (i, x[i].X))
    print("sum:", sum(x[i].X for i in I))
    print("Obj:", model.ObjVal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',help='path of sample data',
                        default='../sample_08/mcdropout_day0.csv')
    parser.add_argument('-a', '--alpha',help='acceptance threshold',
                        default=1.001,type= float )
    parser.add_argument('-n', '--num',help='Top n stock',
                        default=10,type = int)

    args = parser.parse_args()
    main(args)
