import argparse
import copy
import numpy as np
import pandas as pd
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
from tqdm import tqdm
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

from training.load_data_nasdaq import load_EOD_data, load_relation_data
from training.evaluator import evaluate

seed = 123456789
np.random.seed(seed)
tf.set_random_seed(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.hgat_nasdaq import HGAT 
from scipy import sparse
from torch_geometric import utils
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks, 1, device=device)
    pre_pw_dif = (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) 
                  - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth, 0, 1)) -
                 torch.matmul(ground_truth, torch.transpose(all_ones, 0, 1)))
    mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1))
    rank_loss = torch.mean(F.relu((pre_pw_dif * gt_pw_dif) * mask_pw))
    loss = reg_loss + alpha * rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio

class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None, flat=False, gpu=False, in_pro=False):
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # Load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        self.batch_size = len(self.tickers) if batch_size is None else batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.device = 'cuda' if gpu else 'cpu'

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (self.eod_data[:, offset:offset + seq_len, :], 
                np.expand_dims(mask_batch, axis=1),
                np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
                np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1))

    def train(self):
        global df
        model = HGAT(self.batch_size).to(self.device)  # Ensure model is on the correct device
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        
        optimizer_hgat = optim.Adam(model.parameters(), lr=self.parameters['lr'], weight_decay=5e-4)
        inci_mat = np.load('nasdaq.npy')
        inci_sparse = sparse.coo_matrix(inci_mat)
        incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
        hyp_input = incidence_edge[0].to(self.device)
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        hyp_input = hyp_input.to(self.device)
    
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss, tra_reg_loss, tra_rank_loss = 0.0, 0.0, 0.0
            model.train()
            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                
                # Move data to device
                emb_batch = torch.FloatTensor(emb_batch).to(self.device)
                price_batch = torch.FloatTensor(price_batch).to(self.device)
                gt_batch = torch.FloatTensor(gt_batch).to(self.device)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device)
                
                optimizer_hgat.zero_grad()
                
                # Forward pass
                output = model(emb_batch, hyp_input)
                
                cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(
                    output.reshape((1026, 1)), 
                    price_batch, 
                    gt_batch, 
                    mask_batch, 
                    self.parameters['alpha'], 
                    self.batch_size
                )
                
                tra_loss += cur_loss.item()
                tra_reg_loss += cur_reg_loss.item()
                tra_rank_loss += cur_rank_loss.item()
                
                cur_loss.backward()
                optimizer_hgat.step()
    
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))
    
            with torch.no_grad():
                # Test on validation set
                cur_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                cur_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                cur_valid_mask = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                val_loss, val_reg_loss, val_rank_loss = 0.0, 0.0, 0.0
                model.eval()
                for cur_offset in range(self.valid_index - self.parameters['seq'] - self.steps + 1,
                                        self.test_index - self.parameters['seq'] - self.steps + 1):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                    
                    # Move data to device
                    emb_batch = torch.FloatTensor(emb_batch).to(self.device)
                    mask_batch = torch.FloatTensor(mask_batch).to(self.device)
                    price_batch = torch.FloatTensor(price_batch).to(self.device)
                    gt_batch = torch.FloatTensor(gt_batch).to(self.device)
                    
                    output_val = model(emb_batch, hyp_input)
                    
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(
                        output_val, 
                        price_batch, 
                        gt_batch, 
                        mask_batch, 
                        self.parameters['alpha'], 
                        self.batch_size
                    )
                    
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1026, 1))
                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Valid MSE:',
                      val_loss / (self.test_index - self.valid_index),
                      val_reg_loss / (self.test_index - self.valid_index),
                      val_rank_loss / (self.test_index - self.valid_index))
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
                print('\t Valid performance:', cur_valid_perf)
    
                # Test on testing set
                cur_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                cur_test_gt = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                cur_test_mask = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                test_loss, test_reg_loss, test_rank_loss = 0.0, 0.0, 0.0
                model.eval()
                for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                        self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                    
                    # Move data to device
                    emb_batch = torch.FloatTensor(emb_batch).to(self.device)
                    mask_batch = torch.FloatTensor(mask_batch).to(self.device)
                    price_batch = torch.FloatTensor(price_batch).to(self.device)
                    gt_batch = torch.FloatTensor(gt_batch).to(self.device)
                    
                    output_test = model(emb_batch, hyp_input)
                    
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(
                        output_test, 
                        price_batch, 
                        gt_batch, 
                        mask_batch, 
                        self.parameters['alpha'], 
                        self.batch_size
                    )
                    
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1026, 1))
                    test_loss += cur_loss.detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_test_pred[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Test MSE:',
                      test_loss / (self.trade_dates - self.test_index),
                      test_reg_loss / (self.trade_dates - self.test_index),
                      test_rank_loss / (self.trade_dates - self.test_index))
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', cur_test_perf)



    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True

if __name__ == '__main__':
    desc = 'Train a relational rank LSTM model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='Path of EOD data', default='../data/2013-01-01')
    parser.add_argument('-m', help='Market name', default='NASDAQ')
    parser.add_argument('-t', help='Filename for selected tickers')
    parser.add_argument('-l', default=4, help='Length of historical sequence for feature')
    parser.add_argument('-u', default=64, help='Number of hidden units in LSTM')
    parser.add_argument('-s', default=1, help='Steps to make prediction')
    parser.add_argument('-r', default=0.001, help='Learning rate')
    parser.add_argument('-a', default=1, help='Alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Use GPU')
    parser.add_argument('-e', '--emb_file', type=str, default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy', help='Filename for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str, default='sector_industry', help='Relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)
    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r), 'alpha': float(args.a)}

    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod
    )

    RR_LSTM.train()
