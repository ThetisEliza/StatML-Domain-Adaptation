import math
import os
import random

import pandas as pd
import numpy as np


PATH='RawData'

FEMALE = 'FEMALE.csv'
MALE = 'MALE.csv'
MIXED = 'MIXED.csv'


DOMAIN_TR_NUM = 100
DOMAIN_DEV_NUM = 100
test_rate = 0.1

def one_hot_for_c(column, upper):
    assert upper>1
    pos = upper
    A = np.zeros(shape=(len(column), pos))

    for i, c in enumerate(column):
        A[i][-c] = 1
    return A

def one_hot_for_arr(arr):
    assert arr.shape[1] == 7
    year_c = one_hot_for_c(arr[:,0], 3)
    FSMnVR = arr[:,1:3] / np.array([100, 50])
    VR_band = one_hot_for_c(arr[:,3], 3)
    Ethnic = one_hot_for_c(arr[:,4], 11)
    Schoold = one_hot_for_c(arr[:,5], 3)
    Score = np.expand_dims(arr[:,6], axis=1)

    return np.hstack([year_c, FSMnVR, VR_band, Ethnic, Schoold, Score])







def make_source_target_folds(*dfs, shuffle=False, random_seed=1):
    dfs_list = list(dfs)
    folds_list = []

    if shuffle:
        dfs_list = []
        for df in dfs:
            df = df.sample(frac=1, random_state=random_seed)
            dfs_list.append(df)

    dfs_map = {}
    for i, df in enumerate(dfs_list):
        train = one_hot_for_arr(df.values[:DOMAIN_TR_NUM])
        dev = one_hot_for_arr(df.values[DOMAIN_TR_NUM:DOMAIN_TR_NUM + DOMAIN_DEV_NUM])
        test = one_hot_for_arr(df.values[:int(test_rate*len(df.values))])

        # train = df.values[:DOMAIN_TR_NUM]
        # dev = df.values[DOMAIN_TR_NUM:DOMAIN_TR_NUM + DOMAIN_DEV_NUM]
        # test = df.values[DOMAIN_TR_NUM + DOMAIN_DEV_NUM:]

        dfs_map[i] = train, dev, test

    for i, df in enumerate(dfs_list):
        target_train, dev, test = dfs_map[i]
        source_trains = [dfs_map[j][2] for j in [j for j in range(len(dfs_list)) if j != i]]
        folds_list.append((source_trains, target_train,  dev, test))
    return folds_list



def get_folds(shuffle, seed=None):
    female_df = pd.read_csv(os.path.join(PATH, FEMALE))
    male_df = pd.read_csv(os.path.join(PATH, MALE))
    mixed_df = pd.read_csv(os.path.join(PATH, MIXED))
    if seed is None:
        seed = random.randrange(100000000)
    print('Seed:', seed)
    return make_source_target_folds(*[female_df, male_df, mixed_df], shuffle=shuffle, random_seed=seed)

