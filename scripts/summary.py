"""
summarize results

Created on 04/19/2020

@author: RH
"""
import pandas as pd
import os
from itertools import islice


def dissect(path):
    with open(path, 'r') as myfile:
        infolist = list(islice(myfile, -21))
        for line in myfile:
            if 'Best model @ epoch:' in line:
                infolist.append(line)
            if 'Test set: Average loss:' in line:
                infolist.append(line)
    return infolist


if __name__ == '__main__':
    for m in os.listdir('../Results/'):
        for n in os.listdir('../Results/{}'.format(m)):
            if 'txt' in n:
                info = dissect('../Results/{}/{}'.format(m, n))
                print(info)

