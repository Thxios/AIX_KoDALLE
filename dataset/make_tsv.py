import os
import csv
import json
import tqdm
import random as rd
import pandas as pd

img_size = 128
out_path = f'../outs/outs{img_size}'
caption_path = '../MSCOCO_korean_caption'
tsv_path_train = f'caption_encode_train_{img_size}.tsv'
tsv_path_test = f'caption_encode_test_{img_size}.tsv'

out_files = os.listdir(out_path)
encoding = []


for out in out_files:
    if out.split('.')[-1] != 'tsv':
        continue
    with open(os.path.join(out_path, out), 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            encoding.append(row)
encoding.sort()
print(len(encoding))

with open(os.path.join(caption_path, 'MSCOCO_train_val_Korean.json'), 'r') as f:
    caption = json.load(f)
caption.sort(key=lambda x: x['file_path'])

data = []

for i in tqdm.tqdm(range(len(encoding))):
    for capt in caption[i]['caption_ko']:
        # data.append((capt, encoding[i][1]))
        data.append((capt, *eval(encoding[i][1])))
        # print((capt, encoding[i][1]))

n_data = len(data)
print(n_data)
rd.shuffle(data)
test_split = 0.1
train_data, test_data = data[int(test_split*n_data):], data[:int(test_split*n_data)]
print(f'train {len(train_data)}')
print(f'test {len(test_data)}')
# train_df = pd.DataFrame(train_data, columns=['caption', 'encoding'])
train_df = pd.DataFrame(train_data)
train_df.dropna(inplace=True)
print(train_df.head())
print(train_df.shape)
train_df.to_csv(tsv_path_train, sep='\t', index=False, encoding='utf-8')

# test_df = pd.DataFrame(test_data, columns=['caption', 'encoding'])
test_df = pd.DataFrame(test_data)
test_df.dropna(inplace=True)
print(test_df.head())
print(test_df.shape)
test_df.to_csv(tsv_path_test, sep='\t', index=False, encoding='utf-8')
