import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
from time import time
import csv

import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops

    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)

from load_data import load_EOD_data
from evaluator import evaluate, get_top_n_stock, model_const, cal_std_cov, evaluate_portfolio

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
        #self.tickers = self.tickers[0: 10]
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
        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, -1], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    # LSTMモデルを作成する関数
    def train(self):
        # 乱数初期化
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        #データの整形
        train,val,test = np.split(self.gt_data,[self.valid_index,self.test_index],axis=1)
        train_x = np.zeros(
            [len(train[0]) - (self.parameters['seq'] + 1), len(self.tickers) , self.parameters['seq']],
            dtype=float)
        train_y = np.zeros(
            [len(train[0]) - (self.parameters['seq'] + 1), len(self.tickers)])
        val_x = np.zeros(
            [len(val[0]) - (self.parameters['seq'] + 1), len(self.tickers) , self.parameters['seq']],
            dtype=float)
        val_y = np.zeros(
            [len(val[0]) - (self.parameters['seq'] + 1), len(self.tickers)])
        test_x = np.zeros(
            [len(test[0]) - (self.parameters['seq'] + 1), len(self.tickers) , self.parameters['seq']],
            dtype=float)
        test_y = np.zeros(
            [len(test[0]) - (self.parameters['seq'] + 1), len(self.tickers)])
        for i in range(len(train[0]) - (self.parameters['seq'] + 1)):
            train_x[i] += train[:,i : i + self.parameters['seq']]
            train_y[i] += train[:,i + 1 + self.parameters['seq']]
        for i in range(len(val[0]) - (self.parameters['seq'] + 1)):
            val_x[i] += val[:,i : i + self.parameters['seq']]
            val_y[i] += val[:,i + 1 + self.parameters['seq']]
        for i in range(len(test[0]) - (self.parameters['seq'] + 1)):
            test_x[i] += test[:,i : i + self.parameters['seq']]
            test_y[i] += test[:,i + 1 + self.parameters['seq']]
        train_x = train_x.transpose(0,2,1)
        val_x = val_x.transpose(0,2,1)
        test_x = test_x.transpose(0,2,1)

        #モデルの作成
        model = Sequential()
        model.add(LSTM(self.parameters['unit'], batch_input_shape=(None, self.parameters['seq'], len(self.tickers))))
        model.add(Dense(len(self.tickers))) 
        model.compile(loss='mse', optimizer=Adam() , metrics = ['mse'])
        model.summary()

        #モデルの訓練・検証
        hist = model.fit(train_x, train_y, epochs=self.epochs, validation_data=(val_x, val_y), batch_size=self.batch_size)

        # 損失値(Loss)の遷移
        plt.plot(hist.history['loss'], label="train set")
        plt.plot(hist.history['val_loss'], label="test set")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("sin.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=10,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-n', '--num',help='Top n stock',
                        default=5,type = int)
    parser.add_argument('-ta', '--threshold_alpha',help='acceptance threshold',
                        default=1.001,type= float )
    parser.add_argument('-d', '--drop_num',help='number of dropout',
                        default=1000,type = int)
    parser.add_argument('-d_p', '--drop_ratio',help='ratio of dropout',
                        default=0.8,type = float)
    parser.add_argument('-f', '--file_name',help='decide final csv file name',
                        default="data")

    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    mc_LSTM = mcLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=100, batch_size=None, gpu=args.gpu
    )
    pred_all = mc_LSTM.train()
