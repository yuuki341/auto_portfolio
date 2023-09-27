import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
from time import time
import csv

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

class RankLSTM:
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
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )


    def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()

            seed = 0
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.placeholder(tf.float32,
                [self.batch_size, self.parameters['seq'], self.fea_dim])
            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.parameters['unit']
            )

            initial_state = lstm_cell.zero_state(self.batch_size,
                                                 dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell, feature, dtype=tf.float32,
                initial_state=initial_state
            )
            outputs = tf.nn.dropout(outputs,self.mcdrop_p)
            seq_emb = outputs[:, -1, :]
            # One hidden layer
            prediction = tf.layers.dense(
                seq_emb, units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.parameters['lr']
            ).minimize(reg_loss)

            avg_loss = tf.summary.scalar("loss",loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./tensorboard_sample', sess.graph)

        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_valid_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_test_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.parameters['seq'] -
                           self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                [train_summary,(cur_loss, cur_reg_loss, cur_rank_loss, batch_out )]= \
                    sess.run([avg_loss,(loss, reg_loss, rank_loss, optimizer)],
                             feed_dict)
                writer.add_summary(train_summary,j)
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))

            # test on validation set
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_semb, cur_rr, = \
                    sess.run((loss, reg_loss, rank_loss, seq_emb,
                              return_ratio), feed_dict)

                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                                              self.parameters['seq'] -
                                              self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            print('Valid MSE:',
                  val_loss / (self.test_index - self.valid_index),
                  val_reg_loss / (self.test_index - self.valid_index),
                  val_rank_loss / (self.test_index - self.valid_index))
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                      cur_valid_mask)
            print('\t Valid preformance:', cur_valid_perf)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_topn_index = np.zeros(
                [args.num, self.trade_dates - self.test_index],
                dtype=int
            )
            cur_test_topn_weight = np.zeros(
                [args.num, self.trade_dates - self.test_index],
                dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(
                self.test_index - self.parameters['seq'] - self.steps + 1,
                self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                if i == self.epochs - 1:
                    #最終エポックdropoutを実行して、それぞれのtickerのdropout分のデータを集める
                    for num in range(self.mcdrop_num):
                        [test_summary,(cur_loss, cur_reg_loss, cur_rank_loss, cur_semb, cur_rr)] = \
                            sess.run([avg_loss,(loss, reg_loss, rank_loss, seq_emb,
                                    return_ratio)], feed_dict)
                        if num == 0:
                            concat_dropnum_rr = copy.copy(cur_rr)
                        else:
                            concat_dropnum_rr = np.concatenate([concat_dropnum_rr,cur_rr],1)
                    days = cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)
                    #topnのリストを求める
                    top_n_list, top_n_list_index = get_top_n_stock(args.num, concat_dropnum_rr)
                    #標準偏差、共分散を求める
                    std_list, sigma = cal_std_cov(args.num , concat_dropnum_rr, top_n_list, top_n_list_index)
                    #それぞれの株式の割合を求める
                    model_weight = model_const(args.threshold_alpha, args.num, top_n_list, std_list, sigma)
                    #np.savetxt(f'../sample/mcdropout_day{days}.csv', concat_dropnum_rr, delimiter=',')
                else:
                    [test_summary,(cur_loss, cur_reg_loss, cur_rank_loss, cur_semb, cur_rr)] = \
                            sess.run([avg_loss,(loss, reg_loss, rank_loss, seq_emb,
                                    return_ratio)], feed_dict)
                    concat_dropnum_rr = copy.copy(cur_rr)
                writer.add_summary(test_summary)

                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

                cur_test_pred[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(np.mean(concat_dropnum_rr,axis=1))
                    #copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
                if i == self.epochs - 1:
                    cur_test_topn_index[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                        copy.copy(top_n_list_index[:])
                    cur_test_topn_weight[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                        copy.copy(model_weight[:])
                    #print(model_weight)
            # print('----------')
            print('Test MSE:',
                  test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  test_rank_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            if i == self.epochs - 1:
                cur_test_perf = evaluate_portfolio(cur_test_perf, cur_test_pred, cur_test_gt, cur_test_topn_index, cur_test_topn_weight, fname=args.file_name)
                """
                with open(f'./weight.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(cur_test_topn_weight)

                with open(f'./weight_index.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(cur_test_topn_index)
                """

            print('\t Test performance:', cur_test_perf)
            # if cur_valid_perf['mse'] < best_valid_perf['mse']:
            if val_loss / (self.test_index - self.valid_index) < \
                    best_valid_loss:
                best_valid_loss = val_loss / (self.test_index -
                                              self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)

                print('Better valid loss:', best_valid_loss)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)
        sess.close()
        tf.reset_default_graph()

        return best_valid_pred, best_valid_gt, best_valid_mask, \
               best_test_pred, best_test_gt, best_test_mask

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    desc = 'train a rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
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
    parser.add_argument('-f', '--file_name',help='decide final csv file name',
                        default="data")
    parser.add_argument('-d', '--drop_num',help='number of dropout',
                        default=100,type = int)
    parser.add_argument('-d_p', '--drop_ratio',help='ratio of dropout',
                        default=0.8,type = float)

    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    rank_LSTM = RankLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=10, batch_size=None, gpu=args.gpu
    )
    pred_all = rank_LSTM.train()
