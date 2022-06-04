
import os
import pandas as pd
import numpy as np

data_dir = '../dataset/caption_encode_test.tsv'

if __name__ == '__main__':
    docs_pd = pd.read_csv(data_dir, sep='\t')
    caption = docs_pd['caption'].values.tolist()
    encoding = list(map(eval, docs_pd['encoding']))
    print(len(caption))
    print(caption[:10])
    print(encoding[:10])
    print(len(encoding))

