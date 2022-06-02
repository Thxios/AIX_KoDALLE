import os
import csv
import json
import tqdm
import pandas as pd

out_path = '../outs'
caption_path = '../MSCOCO_korean_caption'
tsv_path = 'caption_encode.tsv'

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
        data.append((capt, encoding[i][1]))
        # print((capt, encoding[i][1]))

print(len(data))
df = pd.DataFrame(data, columns=['caption', 'encoding'])
print(df.head())
print(df.shape)
df.to_csv(tsv_path, sep='\t', index=False, encoding='utf-8')
