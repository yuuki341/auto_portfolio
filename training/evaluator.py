import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from gurobipy import Model, GRB
from scipy.stats import norm
import csv
from gurobipy import *

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10


    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    # performance['btl5'] = bt_long5
    # performance['btl10'] = bt_long10
    return performance

def _markowitz(alpha,I, std,sigma,r):
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
    model.addConstr( x.sum() == 1, 'budget')
    model.addConstr( r @ x >= alpha)
    model.__data = x
    return model

"""
平均収益率が高い順にソートして、上位n個出力
"""
def get_top_n_stock(num,stock_sample):
    mean_revenue_list = np.mean(stock_sample,axis=1)
    mean_revenue_sort_index = np.argsort(mean_revenue_list)
    top_n_list = np.zeros(num,dtype=float)
    top_n_list_index = np.zeros(num,dtype=int)
    for i in range(num):
        max_n = mean_revenue_sort_index[-1 * (i + 1)]
        top_n_list_index[i] = max_n
        top_n_list[i] = mean_revenue_list[max_n]
    return top_n_list, top_n_list_index

def get_top_n_stock_nomc(num,stock_sample):
    mean_revenue_sort_index = np.argsort(stock_sample)
    top_n_list = np.zeros(num,dtype=float)
    top_n_list_index = np.zeros(num,dtype=int)
    for i in range(num):
        max_n = mean_revenue_sort_index[-1 * (i + 1)]
        top_n_list_index[i] = max_n
        top_n_list[i] = stock_sample[max_n]
    return top_n_list, top_n_list_index

"""
標準偏差と共分散を計算
"""
def cal_std_cov(top_n,stock_sample,max_n_list,max_n_list_index):
    rev_stock_sample = []
    for i in range(top_n):
        rev_i = max_n_list_index[i]
        rev_stock_sample.append(stock_sample[rev_i])
    rev_stock_sample = np.array(rev_stock_sample)

    std_list = np.std(rev_stock_sample, axis=1)  
    sigma = np.cov(rev_stock_sample)
    return std_list, sigma

def model_const(num, max_n_list, std_list, sigma):
    model_weight = np.zeros(num, dtype=float)

    I = list(range(num))
    #エラーにならないように
    ave_max_n_list = np.average(max_n_list[-1])

    model = _markowitz(ave_max_n_list, I, std_list, sigma, max_n_list)
    model.optimize()

    x = model.__data
    EPS = 1.0e-6
    for i in I:
        model_weight[i] = x[i].X
    return model_weight

def evaluate_portfolio(prediction, ground_truth, topn, topn_weight,fname, report=False):
    bt_longn = 0
    #テスト日分の評価
    with open(f'./test12/{fname}_data.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(prediction.shape[0]):
            # back testing on top n
            real_ret_rat_topn = 0
            weight_index = 0
            #bt_longn = 0
            for pre in topn[i,:]:
                bt_longn += ground_truth[i,pre] * topn_weight[i, weight_index]
                weight_index += 1
            writer.writerow([float(bt_longn)])
    return bt_longn

def greedy_optimization(alpha, weight_num,sigma,r):
    """
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
        Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar(weight_num)
    portfolio_risk = x @ sigma @ x
    model.setObjective(portfolio_risk, GRB.MINIMIZE)
    model.addConstr( x.sum() == 1, 'budget')
    model.addConstr( r @ x >= alpha)
    model.__data = x
    return model

def greedy_optimization_rev(alpha, weight_num,sigma,r):
    """
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
        Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar(weight_num)
    portfolio_risk = x @ r
    model.setObjective(portfolio_risk, GRB.MAXIMIZE)
    model.addConstr( x.sum() == 1, 'budget')
    model.addConstr( x @ sigma @ x <= alpha)
    model.__data = x
    return model


def cardinality_constrained_optimization(alpha, weight_num,sigma,r):
    """
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
        Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar((r.shape[0],))
    z = {}
    for i in range(r.shape[0]):
        z[i] = model.addVar(vtype = GRB.BINARY)
    portfolio_risk = x @ sigma @ x
    model.setObjective(portfolio_risk, GRB.MINIMIZE)
    model.addConstr( r @ x >= alpha)
    model.addConstr(quicksum(z[i] for i in range(r.shape[0])) == weight_num)
    for i in range(r.shape[0]):
        model.addConstr( x[i] <= z[i])
        model.addConstr( 0 <= x[i])
    model.addConstr( x.sum() == 1, 'budget')
    model.__data = x
    return model

def cardinality_constrained_optimization_rev(alpha, weight_num,sigma,r):
    """
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
        Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar((r.shape[0],))
    z = {}
    for i in range(r.shape[0]):
        z[i] = model.addVar(vtype = GRB.BINARY)
    portfolio_risk = x @ r
    model.setObjective(portfolio_risk, GRB.MAXIMIZE)
    model.addConstr( x @ sigma @ x <= alpha)
    model.addConstr(quicksum(z[i] for i in range(r.shape[0])) == weight_num)
    for i in range(r.shape[0]):
        model.addConstr( x[i] <= z[i])
        model.addConstr( 0 <= x[i])
    model.addConstr( x.sum() == 1, 'budget')
    model.__data = x
    return model


def cardinality_constrained_optimization_rev_multiobject(alpha, weight_num,sigma,r):
    """
        - sigma[i,j]: covariance of item i and j
        - r[i]: revenue of item i
        Returns a model, ready to be solved.
    """
    model = Model("portfolio")
    x = model.addMVar((r.shape[0],))
    z = {}
    for i in range(r.shape[0]):
        z[i] = model.addVar(vtype = GRB.BINARY)
    portfolio_risk = x @ r - alpha* (x @ sigma @ x)
    model.setObjective(portfolio_risk, GRB.MAXIMIZE)
    model.addConstr(quicksum(z[i] for i in range(r.shape[0])) == weight_num)
    for i in range(r.shape[0]):
        model.addConstr( x[i] <= z[i])
        model.addConstr( 0 <= x[i])
    model.addConstr( x.sum() == 1, 'budget')
    model.__data = x
    return model

def greedy(weight_max, mc_drop_list_day, mc_drop_num):
    index_list = []
    optimal_min = float('inf')
    
    #平均予測収益のベクトル
    mean_list = np.mean(mc_drop_list_day,axis=1)
    #予測収益の分散のベクトル
    var_list = np.var(mc_drop_list_day,axis=1)
    #二次計画問題の基準
    standard = mean_list[mean_list>0].mean()
    #print(standard.shape)
    #分散が最小なものを追加
    stock_list_index = np.where(mean_list > standard)[0]
    
    var_min = float('inf')
    for i in stock_list_index:
        if var_min > var_list[i]:
            var_min = var_list[i]
            var_min_index = i
    index_list.append(var_min_index)
    
    for weight_num in range(2, weight_max + 1):
        model_weight = np.zeros(weight_num, dtype=float)
        index_min = 0
        #for i in stock_list_index:
        for i in range(mean_list.shape[0]):
            if i in index_list:
                continue
            stock_sample_list_day = np.zeros((weight_num, mc_drop_num)) 
            r = np.zeros(weight_num)
            for j in range(weight_num - 1):
                stock_sample_list_day[j,:] = mc_drop_list_day[index_list[j],:]
                r[j] = mean_list[index_list[j]]
            stock_sample_list_day[-1,:] = mc_drop_list_day[i,:]
            sigma = np.cov(stock_sample_list_day)
            r[-1] = mean_list[i] 
           
            model = greedy_optimization(standard, weight_num, sigma, r)
            model.optimize()
            x = model.__data
            if optimal_min > model.ObjVal:
                for j in range(weight_num):
                    model_weight[j] = x[j].X
                index_min = i
                optimal_min = model.ObjVal
        index_list.append(index_min)
    return index_list, model_weight


def greedy_rev(weight_max, mc_drop_list_day, mc_drop_num):
    index_list = []
    optimal_min = float('inf')
    
    #平均予測収益のベクトル
    mean_list = np.mean(mc_drop_list_day,axis=1)
    #予測収益の分散のベクトル
    var_list = np.var(mc_drop_list_day,axis=1)
    #二次計画問題の基準
    standard = np.mean(var_list)
    #分散が最小なものを追加
    stock_list_index = np.where(var_list < standard)[0]
    
    return_max = -float('inf')
    for i in stock_list_index:
        if return_max > mean_list[i]:
            return_max = mean_list[i]
            return_max_index = i
    index_list.append(return_max_index)
    
    for weight_num in range(2, weight_max + 1):
        model_weight = np.zeros(weight_num, dtype=float)
        index_min = 0
        for i in range(mean_list.shape[0]):
            if i in index_list:
                continue
            stock_sample_list_day = np.zeros((weight_num, mc_drop_num)) 
            r = np.zeros(weight_num)
            for j in range(weight_num - 1):
                stock_sample_list_day[j,:] = mc_drop_list_day[index_list[j],:]
                r[j] = mean_list[index_list[j]]
            stock_sample_list_day[-1,:] = mc_drop_list_day[i,:]
            sigma = np.cov(stock_sample_list_day)
            r[-1] = mean_list[i] 
           
            model = greedy_optimization_rev(standard, weight_num, sigma, r)
            model.optimize()
            x = model.__data
            if optimal_min > model.ObjVal:
                for j in range(weight_num):
                    model_weight[j] = x[j].X
                index_min = i
                optimal_min = model.ObjVal
        index_list.append(index_min)
    return index_list, model_weight

def cardinary_constrained(weight_max, mc_drop_list_day, mc_drop_num):
    index_list = []
    
    #平均予測収益のベクトル
    mean_list = np.mean(mc_drop_list_day,axis=1)
    #予測収益の分散のベクトル
    #var_list = np.var(mc_drop_list_day,axis=1)
    #二次計画問題の基準
    standard = mean_list[mean_list>0].mean()
    sigma = np.cov(mc_drop_list_day)
    print(mean_list.shape,sigma.shape,standard.shape)

    model = cardinality_constrained_optimization(standard, weight_max,sigma,mean_list)
    model.optimize()
    x = model.__data
    index_list = np.where(x.X > 0)
    print(index_list[0], x.X[x.X>0])
    
    return index_list[0], x.X[x.X>0]

def cardinary_constrained_rev(weight_max, mc_drop_list_day, mc_drop_num):
    index_list = []
    
    #平均予測収益のベクトル
    mean_list = np.mean(mc_drop_list_day,axis=1)
    #予測収益の分散のベクトル
    var_list = np.var(mc_drop_list_day,axis=1)
    #二次計画問題の基準
    standard = np.mean(var_list) / weight_max
    sigma = np.cov(mc_drop_list_day)
    print(mean_list.shape,sigma.shape,standard.shape)

    model = cardinality_constrained_optimization_rev(standard, weight_max,sigma,mean_list)
    model.optimize()
    x = model.__data
    index_list = np.where(x.X > 0)
    print(index_list[0], x.X[x.X>0])
    
    return index_list[0], x.X[x.X>0]

def cardinary_constrained_rev_multiobj(weight_max, mc_drop_list_day, mc_drop_num):
    index_list = []
    
    #平均予測収益のベクトル
    mean_list = np.mean(mc_drop_list_day,axis=1)
    #予測収益の分散のベクトル
    var_list = np.var(mc_drop_list_day,axis=1)
    #二次計画問題の基準
    standard = 0.5
    sigma = np.cov(mc_drop_list_day)

    model = cardinality_constrained_optimization_rev_multiobject(standard, weight_max,sigma,mean_list)
    model.optimize()
    x = model.__data
    index_list = np.where(x.X > 0)
    print(index_list[0], x.X[x.X>0])
    
    return index_list[0], x.X[x.X>0]
    