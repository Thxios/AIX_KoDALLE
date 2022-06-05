import argparse
import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial


class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.docs = pd.read_csv(file, sep='\t')
        docs_pd = pd.read_csv(file, sep='\t')
        docs_pd.dropna(inplace=True)
        self.len = docs_pd.shape[0]

        # self.caption_ids = list(map(self.tokenizer.encode, docs_pd.iloc[:, 0]))
        # self.encoding = list(map(eval, docs_pd['encoding']))

        self.caption = docs_pd.iloc[:, 0].values

        # self.encoding = docs_pd.iloc[:, 1:].values.tolist()
        self.encoding = docs_pd.iloc[:, 1:].values
        eos = np.array([self.tokenizer.eos_token_id] * self.len).reshape([-1, 1])
        self.encoding = np.concatenate([self.encoding, eos], axis=1)

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        # instance = self.docs.iloc[idx]
        # input_ids = self.tokenizer.encode(instance['caption'])
        input_ids = np.array(self.tokenizer.encode(self.caption[idx]))
        # input_ids = self.caption_ids[idx]
        input_ids = self.add_padding_data(input_ids)

        # label_ids = self.encoding[idx]
        # label_ids.append(self.tokenizer.eos_token_id)
        # dec_input_ids = [self.tokenizer.eos_token_id]
        # dec_input_ids += label_ids[:-1]

        # label_ids = np.concatenate([self.encoding[idx], [self.tokenizer.eos_token_id]])
        label_ids = self.encoding[idx]
        dec_input_ids = np.concatenate([[self.tokenizer.eos_token_id], label_ids[:-1]])

        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len


class KobartSummaryModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=257,
                 batch_size=8,
                 num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                          self.tok,
                                          self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                         self.tok,
                                         self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test
