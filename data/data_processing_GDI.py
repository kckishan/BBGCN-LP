#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product 
import os

# In[2]:


gdi = pd.read_csv('./GDI/curated_gene_disease_associations.tsv', sep = '\t')
gdi = gdi[['geneId','diseaseId']]
gene = list(set(gdi['geneId'].tolist()))
disease = list(set(gdi['diseaseId'].tolist()))

pd.DataFrame(list(set(gdi['geneId'].tolist() + gdi['diseaseId'].tolist()))).rename(columns={0: "Entity_ID"}).to_csv('./GDI/entity_list.csv')
entity_list = pd.DataFrame(list(set(gdi['geneId'].tolist() + gdi['diseaseId'].tolist()))).rename(columns={0: "Entity_ID"})


# In[3]:


# It is too large a number to do combination from., 
# we use the alternative method than DDI, DTI, PPI below. 
# Basically, sample a fixed number for each node, instead of do combination


# In[4]:


comb = []
for i in gene:
    comb = comb + (list(zip([i] * 20, random.choices(disease, k = 20))))
    
for j in disease:
    comb = comb + (list(zip(random.choices(gene, k = 20), [i] * 20)))


# In[5]:


pos = [(i[0], i[1]) for i in (gdi.values)]
neg = list(set(comb) - set(pos))
comb_flipped = [(i[1], i[0]) for i in comb]
neg_2 = list(set(comb_flipped) - set(pos))
neg_2 = [(i[1], i[0]) for i in neg_2]
neg_final = list(set(neg) & set(neg_2))

random.seed(a = 1)
neg_sample = random.sample(neg_final, len(gdi))

df = pd.DataFrame(pos+neg_sample)
df['label'] = np.array([1]*len(pos) + [0]*len(neg_sample))

df = df.rename({0:'Gene_ID', 1:'Disease_ID'}, axis = 1)


# In[6]:


def create_fold(df, x):
    test = df.sample(frac = 0.2, replace = False, random_state = x)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = 0.125, replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    path = './GDI/fold'+str(x)
    train.reset_index(drop = True).to_csv(path + '/train.csv')
    val.reset_index(drop = True).to_csv(path + '/val.csv')
    test.reset_index(drop = True).to_csv(path + '/test.csv')
    
    return train, val, test


# In[8]:


for fold_n in range(1, 6):
    if not os.path.exists(f'./GDI/fold{fold_n}'):
        os.makedirs(f'./GDI/fold{fold_n}')
        train, val, test = create_fold(df, fold_n)


network_type = 'GDI'
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

