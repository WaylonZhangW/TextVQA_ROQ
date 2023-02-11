from turtle import pd
import numpy as np
import os
import pdb
from tqdm import tqdm
path = '/root/data/imdb/ssbaseline_stvqa/qi_obj_frcn_imdb_subtrain.npy'
imdb = np.load(path, allow_pickle=True)
print(imdb[0],imdb[1].keys())

print(type(imdb))


my_path = '/root/data/imdb/our_stvqa/stvqa_imdb_subtrain.npy'
my_imdb = np.load(my_path, allow_pickle=True)
print(my_imdb[0])
pdb.set_trace()
for idx,item in tqdm(enumerate(my_imdb[1:])):
    assert item['question_id'] == imdb[idx+1]['question_id']
    my_imdb[idx+1]['obj_text'] = imdb[idx+1]['obj_label']

print(my_imdb[100].keys())

np.save(my_path,my_imdb)


