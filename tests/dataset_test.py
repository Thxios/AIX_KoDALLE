
import os
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
from train_scripts.dataset import KoBARTSummaryDataset, KobartSummaryModule
from tqdm import tqdm
from time import time, sleep

data_dir = '../dataset/caption_encode_test.tsv'

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1', cache_dir='../.cache')
    print('loaded tokenizer')
    print(tokenizer.vocab_size)
    print('eos id:', tokenizer.eos_token_id)
    # docs_pd.drop(376414, inplace=True)
    # docs_pd.dropna(inplace=True)
    # print(docs_pd.shape)
    # print(docs_pd.loc[376414])
    # for i, caption in enumerate(tqdm(docs_pd['caption'])):
    #     try:
    #         tokenizer.encode(caption)
    #     except TypeError:
    #         print(i, caption)
    #         print(type(caption))
    # caption_ids = list(map(tokenizer.encode, docs_pd['caption']))
    # print(len(caption_ids))
    # print(caption_ids[:5])
    # caption = docs_pd['caption'].values.tolist()
    # encoding = list(map(eval, docs_pd['encoding']))
    # print(len(caption))
    # print(caption[:10])
    # print(encoding[:10])
    # print(len(encoding))

    dataset = KoBARTSummaryDataset(data_dir, tokenizer, 257)
    print(dataset[0])

    # dm = KobartSummaryModule(data_dir,
    #                          data_dir,
    #                          tokenizer,
    #                          batch_size=32,
    #                          max_len=256,
    #                          num_workers=4)
    # dm.setup()
    # print(dm)
    # for batch in dm.train_dataloader():
    #     print(batch)
    #     break
    pass

