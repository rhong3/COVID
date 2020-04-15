"""
Prepare sample split

Created on 04/10/2020

@author: RH

"""
import os
import pandas as pd
import numpy as np


def set_sep(path, cut=0.3):
    trlist = []
    telist = []
    valist = []
    pos = pd.read_csv('../COVID-CT-MetaInfo.csv', header=0, usecols=['image', 'patient'])
    neg = pd.read_csv('../NonCOVID-CT-MetaInfo.csv', header=0, usecols=['image', 'patient'])
    pos['label'] = 1
    neg['label'] = 0
    pos['path'] = '../images/CT_COVID/' + pos['image']
    neg['path'] = '../images/CT_nonCOVID/' + neg['image']
    pos = pos.drop(['image'], axis=1)
    neg = neg.drop(['image'], axis=1)
    unqp = list(pos.patient.unique())
    unqn = list(neg.patient.unique())
    np.random.shuffle(unqp)
    np.random.shuffle(unqn)

    validation = unqp[:int(len(unqp) * cut / 2)]
    valist.append(pos[pos['patient'].isin(validation)])
    test = unqp[int(len(unqp) * cut / 2):int(len(unqp) * cut)]
    telist.append(pos[pos['patient'].isin(test)])
    train = unqp[int(len(unqp) * cut):]
    trlist.append(pos[pos['patient'].isin(train)])

    validation = unqn[:int(len(unqn) * cut / 2)]
    valist.append(neg[neg['patient'].isin(validation)])
    test = unqn[int(len(unqn) * cut / 2):int(len(unqn) * cut)]
    telist.append(neg[neg['patient'].isin(test)])
    train = unqn[int(len(unqn) * cut):]
    trlist.append(neg[neg['patient'].isin(train)])

    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)

    tepd = pd.DataFrame(test.sample(frac=1), columns=['patient', 'label', 'path'])
    vapd = pd.DataFrame(validation.sample(frac=1), columns=['patient', 'label', 'path'])
    trpd = pd.DataFrame(train.sample(frac=1), columns=['patient', 'label', 'path'])

    tepd.to_csv(path + '/te_sample.csv', header=True, index=False)
    trpd.to_csv(path + '/tr_sample.csv', header=True, index=False)
    vapd.to_csv(path + '/va_sample.csv', header=True, index=False)



