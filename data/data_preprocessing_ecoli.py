
import pandas as pd
import numpy as np
from itertools import combinations
import random
import os

# %%

# should take less than 1 min
ecoli = pd.read_csv('./ecoli/ecoli.edgelist', sep=' ', header=None)
gene_list = pd.DataFrame(list(set(ecoli[0].tolist() + ecoli[1].tolist()))).rename(columns={0: "Gene_ID"})
print(gene_list)
comb = combinations(list(gene_list.values.T[0]), 2)
comb = list(comb)
pos = [(i[0], i[1]) for i in (ecoli.values)]
neg = list(set(comb) - set(pos))
print(len(neg))
comb_flipped = [(i[1], i[0]) for i in comb]
neg_2 = list(set(comb_flipped) - set(pos))
neg_2 = [(i[1], i[0]) for i in neg_2]
neg_final = list(set(neg) & set(neg_2))

random.seed(a=1)
# adjust negative ratio here.
neg_sample = random.sample(neg_final, len(ecoli))
#
df = pd.DataFrame(pos + neg_sample)
df['label'] = np.array([1] * len(pos) + [0] * len(neg_sample))
df = df.rename({0: 'Gene1_ID', 1: 'Gene2_ID'}, axis=1)


def create_fold(df, x):
    test = df.sample(frac=0.2, replace=False, random_state=x)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=0.125, replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]

    path = './ecoli/fold' + str(x)
    print(path)
    train.reset_index(drop=True).to_csv(path + '/train.csv')
    val.reset_index(drop=True).to_csv(path + '/val.csv')
    test.reset_index(drop=True).to_csv(path + '/test.csv')

    return train, val, test


for fold_n in range(1, 6):
    if not os.path.exists(f'./ecoli/fold{fold_n}'):
        os.makedirs(f'./ecoli/fold{fold_n}')
    train, val, test = create_fold(df, fold_n)