"""
Prepare sample split

Created on 04/10/2020

@author: RH

"""
import os
import pandas as pd
import numpy as np


def set_sep(path, cut=0.3):
    imlist = []
    for m in os.listdir('../images/CT_COVID'):
        imlist.append(['../images/CT_COVID/{}'.format(m), 1])
    for m in os.listdir('../images/CT_nonCOVID'):
        imlist.append(['../images/CT_nonCOVID/{}'.format(m), 0])
    np.random.shuffle(imlist)
    telist = imlist[:int(len(imlist)*cut/2)]
    valist = imlist[int(len(imlist)*cut/2):int(len(imlist)*cut)]
    trlist = imlist[int(len(imlist)*cut):]

    tepd = pd.DataFrame(telist, columns=['path', 'label'])
    vapd = pd.DataFrame(valist, columns=['path', 'label'])
    trpd = pd.DataFrame(trlist, columns=['path', 'label'])

    tepd.to_csv(path + '/te_sample.csv', header=True, index=False)
    trpd.to_csv(path + '/tr_sample.csv', header=True, index=False)
    vapd.to_csv(path + '/va_sample.csv', header=True, index=False)

    return trpd, vapd, tepd





