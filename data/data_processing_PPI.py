#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 
import random
import os

# In[2]:


# takes around 2 min, but need some RAM, if does not work for you, please checkout data_processing_GDI for a more efficient way

ppi = pd.read_csv('./PPI/hi-iii.txt', sep = '\t', header = None)

pd.DataFrame(list(set(ppi[0].tolist() + ppi[1].tolist()))).rename(columns={0: "Protein1_ID"}).to_csv('./PPI/protein_list.csv')

protein_list = pd.DataFrame(list(set(ppi[0].tolist() + ppi[1].tolist()))).rename(columns={0: "Protein1_ID"})
comb = combinations(list(protein_list.values.T[0]), 2)
comb = list(comb)

pos = [(i[0], i[1]) for i in (ppi.values)]
neg = list(set(comb) - set(pos))
comb_flipped = [(i[1], i[0]) for i in comb]
neg_2 = list(set(comb_flipped) - set(pos))
neg_2 = [(i[1], i[0]) for i in neg_2]
neg_final = list(set(neg) & set(neg_2))

random.seed(a = 1)
neg_sample = random.sample(neg_final, len(ppi))

df = pd.DataFrame(pos+neg_sample)
df['label'] = np.array([1]*len(pos) + [0]*len(neg_sample))
df = df.rename({0:'Protein1_ID', 1:'Protein2_ID'}, axis = 1)


# In[3]:


def create_fold(df, x):
    test = df.sample(frac = 0.2, replace = False, random_state = x)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = 0.125, replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    path = './PPI/fold'+str(x)
    train.reset_index(drop = True).to_csv(path + '/train.csv')
    val.reset_index(drop = True).to_csv(path + '/val.csv')
    test.reset_index(drop = True).to_csv(path + '/test.csv')
    
    return train, val, test


# In[4]:

for fold_n in range(1, 6):
    if not os.path.exists(f'./PPI/fold{fold_n}'):
        os.makedirs(f'./PPI/fold{fold_n}')
        train, val, test = create_fold(df, fold_n)


# In[ ]:

network_type = 'PPI'
def create_missing_edges(df, train_percent, seed):
    train = df.sample(frac=train_percent, replace=False, random_state=seed)
    test_val = df[~df.index.isin(train.index)]
    val = test_val.sample(frac=0.1, replace=False, random_state=1)
    test = test_val[~test_val.index.isin(val.index)]

    path = f'./{network_type}/{int(train_percent*100)}/fold{fold_n}'
    train.reset_index(drop=True).to_csv(path + '/train.csv')
    val.reset_index(drop=True).to_csv(path + '/val.csv')
    test.reset_index(drop=True).to_csv(path + '/test.csv')

    return train, val, test


for train_percent in [0.1, 0.3, 0.5, 0.7]:
    for fold_n in range(1, 6):
        print(f'./{network_type}/{int(train_percent*100)}/fold{fold_n}')
        if not os.path.exists(f'./{network_type}/{int(train_percent*100)}/fold{fold_n}'):
            os.makedirs(f'./{network_type}/{int(train_percent*100)}/fold{fold_n}')
            train, val, test = create_missing_edges(df, train_percent, fold_n)
