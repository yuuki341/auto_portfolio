import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
import time
import csv
from multiprocessing import Pool

import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from load_data import load_EOD_data
from evaluator import evaluate, get_top_n_stock, model_const, cal_std_cov, evaluate_portfolio,greedy, cardinary_constrained
from sklearn.metrics import mean_squared_error

def mc_prediction(model,test_x,test_y,test_y_pred,MSE_list,mc_drop_list):
    test_y_pred = model.predict(test_x)
    MSE = mean_squared_error(test_y_pred,test_y)
    MSE_list[num] = MSE
    print(MSE)
    mc_drop_list[:,:,num] = test_y_pred
    print(num,self.mcdrop_num)

class mcLSTM:
    def __init__(self, data_path, market_name, tickers_fname, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        ### DEBUG
        #self.tickers = self.tickers[0: 1]
        print('#tickers selected:', len(self.tickers))
        #self.eod_data(1026,1245,5):(tickers,日数,特徴量) 全データ
        #self.mask_data(1026,1245):(tickers,日数) 株価欠けているデータ
        #self.gt_data(1026,1245):(tickers,日数) 収益率
        #self.price_data(1026,1245):(tickers,日数) ベースの株価
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5
        self.mcdrop_num = args.drop_num
        self.mcdrop_p = args.drop_ratio
        self.top_num = args.top_num

    # LSTMモデルを作成する関数
    def train(self):
        # 乱数初期化
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        #データの整形
        train,val,test = np.split(self.gt_data,[self.valid_index,self.test_index],axis=1)
        #train,val,test = np.split(self.eod_data[:,:,-1],[self.valid_index,self.test_index],axis=1)
        train_x = np.zeros(
            [len(train[0]) - self.parameters['seq'], len(self.tickers) , self.parameters['seq']],
            dtype=float)
        train_y = np.zeros(
            [len(train[0]) - self.parameters['seq'], len(self.tickers)])
        val_x = np.zeros(
            [len(val[0]) - self.parameters['seq'], len(self.tickers) , self.parameters['seq']],
            dtype=float)
        val_y = np.zeros(
            [len(val[0]) - self.parameters['seq'], len(self.tickers)])
        test_x = np.zeros(
            [len(test[0]) - self.parameters['seq'], len(self.tickers) , self.parameters['seq']],
            dtype=float)
        test_y = np.zeros(
            [len(test[0]) - self.parameters['seq'], len(self.tickers)])
        for i in range(len(train[0]) - self.parameters['seq']):
            train_x[i] += train[:,i : i + self.parameters['seq']]
            train_y[i] += train[:,i + self.parameters['seq']]
            #train_y[i] += self.gt_data[:,i + self.parameters['seq']]
        for i in range(len(val[0]) - self.parameters['seq']):
            val_x[i] += val[:,i : i + self.parameters['seq']]
            val_y[i] += val[:,i + self.parameters['seq']]
            #val_y[i] += self.gt_data[:,self.valid_index + i + self.parameters['seq']]
        for i in range(len(test[0]) - self.parameters['seq']):
            test_x[i] += test[:,i : i + self.parameters['seq']]
            test_y[i] += test[:,i + self.parameters['seq']]
            #test_y[i] += self.gt_data[:,self.test_index + i + self.parameters['seq']]
        train_x = train_x.transpose(0,2,1)
        val_x = val_x.transpose(0,2,1)
        test_x = test_x.transpose(0,2,1)
        p = np.random.permutation(len(train_x))
        train_x = train_x[p]
        train_y = train_y[p]

        #モデルの作成
        inputs = Input(shape=(self.parameters['seq'], len(self.tickers)))
        lstm = LSTM(self.parameters['unit'], return_sequences=True)(inputs)
        lstm = Dropout(self.mcdrop_p)(lstm,training=True)
        """
        lstm = LSTM(self.parameters['unit'], return_sequences=True)(lstm)
        lstm = Dropout(self.mcdrop_p)(lstm,training=True)
        """
        lstm = LSTM(self.parameters['unit'], return_sequences=True)(lstm)
        lstm = Dropout(self.mcdrop_p)(lstm,training=True)
        lstm = LSTM(self.parameters['unit'], return_sequences=False)(lstm)
        drop = Dropout(self.mcdrop_p)(lstm,training=True)
        dense = Dense(len(self.tickers))(drop)
        model =  Model(inputs, dense)
        model.compile(loss='mse', optimizer=Adam(lr=self.parameters['lr']), metrics = ['mse'])
        model.summary()
        #tf.keras.utils.plot_model(model, show_shapes=True)

        #モデルの訓練・検証
        hist = model.fit(train_x, train_y, epochs=self.epochs, validation_data=(val_x, val_y), batch_size=self.batch_size,verbose=0)

        # 損失値(Loss)の遷移
        plt.plot(hist.history['loss'], label="train set")
        plt.plot(hist.history['val_loss'], label="val set")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.ylim( 0.0005, 0.0006)
        #plt.ylim( 0, 0.05)
        plt.legend()
        #fig_name = f"result/{args.m}_dropr{args.drop_ratio}_unit{args.u}_{args.f}.png"
        #fig_name = f"search/fig{args.m}_layer4_e{args.e}.png"
        fig_name = f"test9/{args.e}_{args.m}_dropr{args.drop_ratio}_unit{args.u}_{args.f}_fig.png"
        plt.savefig(fig_name)
        val_graph = [np.min(hist.history['val_loss']),np.argmin(hist.history['val_loss'])]
        txt_name = f"test9/{args.e}_{args.m}_dropr{args.drop_ratio}_unit{args.u}_{args.f}.txt"
        np.savetxt(txt_name, val_graph, delimiter=',')

        mc_drop_list = np.zeros([test_y.shape[0], len(self.tickers), self.mcdrop_num],dtype=float)
        MSE_list = np.zeros([self.mcdrop_num])
        #テストデータでの予測
        for num in range(self.mcdrop_num):
            test_y_pred = model.predict(test_x)
            MSE = mean_squared_error(test_y_pred,test_y)
            MSE_list[num] = MSE
            print(MSE)
            mc_drop_list[:,:,num] = test_y_pred
            print(num,self.mcdrop_num)
        day_weight = np.zeros([test_y.shape[0], self.top_num],dtype=float)
        top_n_list_index_test_days = np.zeros([test_y.shape[0], self.top_num], dtype=int)
        for day in range(test_y.shape[0]):
            if self.top_num != 1:
                index_temp,weight_temp = cardinary_constrained(self.top_num, mc_drop_list[day], self.mcdrop_num)
                if index_temp.shape[0] <= self.top_num - 1:
                    while index_temp.shape[0] <=  self.top_num - 1:
                        index_temp = np.concatenate([index_temp, [0]])
                        weight_temp = np.concatenate([weight_temp, [0]])
                top_n_list_index_test_days[day,:] = index_temp
                day_weight[day,:] = weight_temp
            else:
                var_list = np.var(mc_drop_list[day],axis=1)
                top_n_list_index_test_days[day,:] = np.argmin(var_list)
                day_weight[day,:] = 1
        evaluate_portfolio(test_y_pred, test_y, top_n_list_index_test_days, day_weight, fname=args.f)
        np.savetxt(f'./test9/{args.f}_seed{args.seed}_epoch{args.e}_{args.m}_index.csv', top_n_list_index_test_days, delimiter=',')
        np.savetxt(f'./test9/{args.f}_seed{args.seed}_epoch{args.e}_{args.f}_weight.csv', day_weight, delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=50,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=256,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.0001,
                        help='learning rate')
    parser.add_argument('-n', '--top_num',help='Top n stock',
                        default=5,type = int)
    parser.add_argument('-d', '--drop_num',help='number of dropout',
                        default=1000,type = int)
    parser.add_argument('-d_p', '--drop_ratio',help='ratio of dropout',
                        default=0.2,type = float)
    parser.add_argument('-f', help='decide final csv file name',
                        default="data")
    parser.add_argument('-b',help='number of batchsize',
                        default=50,type=int)
    parser.add_argument('-e',help='number of epochs',
                        default=5,type=int)
    parser.add_argument('--seed',help='seed',
                        default=0,type = int)

    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r)}
    print('arguments:', args)
    print('parameters:', parameters)

    start = time.time()
    mc_LSTM = mcLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=args.e, batch_size=args.b
    )
    pred_all = mc_LSTM.train()
    end = time.time()
    time_diff = end - start
    print(f"process time:{time_diff}")