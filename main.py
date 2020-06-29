import copy
from collections import defaultdict
from math import inf

from dataExtractor import get_folds
from model import fitnevalue, MethodEnum
import itertools
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class ModelEnum:
    LR = 'LinearRegression', {}
    LASSO = 'Lasso', {'alpha': [0.1]}
    RIDGE = 'Ridge', {'alpha': [10**i for i in range(-3, 3, 1)]}
    NN = 'NeuralNetwork', {}

    Models = [LR, LASSO, RIDGE]

models = [ModelEnum.RIDGE, ModelEnum.NN]
methods = [MethodEnum.ALL, MethodEnum.FEATURE_AUGMENTATION]




def test_avg(*seeds):
    ds = [{}, {}, {}]
    for seed in seeds:
        folds = get_folds(True, seed)
        for model in models:
            for method in methods:
                if len(model[1]) == 0:
                    results = fitnevalue(method, model[0], *folds, **{})

                else:
                    min1, min2, min3 = inf, inf, inf

                    para_items = list(model[1].items())
                    # print(para_items)
                    ks, paras_list = zip(*para_items)
                    # print(ks)
                    # print(paras_list)
                    paras = list(itertools.product(*paras_list))
                    # print(paras)
                    paras_choices = []
                    for para in paras:
                        d = {}
                        for k, p in zip(ks, para):
                            d[k] = p
                        paras_choices.append(d)
                    for para in paras_choices:
                        r1, r2, r3 = fitnevalue(method, model[0], *folds, **para)
                        if r1<min1 and r2<min2 and r3<min3:
                            min1 = r1
                            min2 = r2
                            min3 = r3
                    results = min1, min2, min3

                test_para = 'Method:{:20} Model:{:20}'.format(method, model[0])
                for d, mse in list(zip(ds, results)):
                    if d.get(test_para):
                        d[test_para] += mse
                    else:
                        d[test_para] = mse
    domains = ['Female', 'Male', 'Mixed']
    for d, domain in list(zip(ds, domains)):
        print(domain)
        for test_para, mse in d.items():
            d[test_para] =  mse / (len(seeds))
            print('{:50} mse:{:8}'.format(
                test_para,
                round(mse / len(seeds), 2) if mse / len(seeds) < 200 else inf
            )
            )
        print()

def test(seed=None):
    test_avg(seed)

if __name__ == '__main__':
    test(3)
    # A = np.array([[1,2],[3,4],[1,5]])
    # B = np.array([2,3])
    # print(A)
    # Me = np.mean(A, axis=0)
    # Ma = np.max(A, axis=0)
    # Mi = np.min(A, axis=0)
    # R = Ma-Mi
    # print((A-Me)/R)
