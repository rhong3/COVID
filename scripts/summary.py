"""
summarize results

Created on 04/19/2020

@author: RH
"""
import pandas as pd
import os, sys, re
import numpy as np


def dissect(path):
    start = 100000000000000000000000
    count = 0
    tempa = []
    info = []
    with open(path, 'r') as myfile:
       for line in myfile:
           tempa.append(line.strip())
           if 'Best model @ epoch:' in line.strip():
               start = count
               info.append(line.strip().split('epoch:  ')[1])
           if 'Test set:' in line.strip():
               info.append(line.strip().split('loss: ')[1].split(',')[0])

           if 'Per image metrics' in line.strip() and count > start:
               recorda = count
           if 'Per patient metrics' in line.strip() and count > start:
               recordb = count
           count += 1

    info.extend([re.search('TP= (.+?) ', tempa[recorda+1]).group(1),
                 re.search('TN= (.+?) ', tempa[recorda + 1]).group(1),
                 re.search('FN= (.+?) ', tempa[recorda + 1]).group(1),
                 tempa[recorda + 1].split('FP= ')[1]])
    if int(tempa[recorda+2].split('TP+FP ')[1]) == 0:
        info.extend([np.nan, np.nan, np.nan, tempa[recorda+3].split('acc ')[1],
                     tempa[recorda+4].split('AUC ')[1], tempa[recorda+5].split('Average precision-recall score: ')[1]])
    else:
        info.extend([tempa[recorda+3].split('precision ')[1], tempa[recorda+4].split('recall ')[1],
                     tempa[recorda+5].split('F1 ')[1], tempa[recorda+6].split('acc ')[1],
                     tempa[recorda+7].split('AUC ')[1], tempa[recorda+8].split('Average precision-recall score: ')[1]])

    info.extend([re.search('TP= (.+?) ', tempa[recordb + 1]).group(1),
                 re.search('TN= (.+?) ', tempa[recordb + 1]).group(1),
                 re.search('FN= (.+?) ', tempa[recordb + 1]).group(1),
                 tempa[recordb + 1].split('FP= ')[1]])
    if int(tempa[recordb + 2].split('TP+FP= ')[1]) == 0:
        info.extend([np.nan, np.nan, np.nan, tempa[recordb + 3].split('acc= ')[1],
                     tempa[recordb + 4].split('AUC= ')[1],
                     tempa[recordb + 5].split('Average precision-recall score: ')[1]])
    else:
        info.extend([tempa[recordb + 3].split('precision= ')[1], tempa[recordb + 4].split('recall= ')[1],
                     tempa[recordb + 5].split('F1= ')[1], tempa[recordb + 6].split('acc= ')[1],
                     tempa[recordb + 7].split('AUC= ')[1],
                     tempa[recordb + 8].split('Average precision-recall score: ')[1]])
    return info


if __name__ == '__main__':
    composite = []
    for m in os.listdir('../Results/'):
        if m == '.DS_Store':
            continue
        for n in os.listdir('../Results/{}'.format(m)):
            if 'txt' in n:
                information = dissect('../Results/{}/{}'.format(m, n))
                if '_pt' in m:
                    information.append('pretrained')
                    information.append(m.split('_ptCOVID')[0])
                else:
                    information.append('scratch')
                    information.append(m.split('COVID')[0])
                composite.append(information)
    summ = pd.DataFrame(composite, columns=['best epoch', 'test loss', 'TP_image', 'TN_image', 'FN_image', 'FP_image',
                                            'precision_image', 'recall_image', 'F1_image', 'accuracy_image',
                                            'AUROC_image', 'AUPRC_image', 'TP_patient', 'TN_patient', 'FN_patient',
                                            'FP_patient', 'precision_patient', 'recall_patient', 'F1_patient',
                                            'accuracy_patient', 'AUROC_patient', 'AUPRC_patient', 'state', 'model'])

    summ = summ[['model', 'state', 'best epoch', 'test loss', 'TP_image', 'TN_image', 'FN_image', 'FP_image',
                                            'precision_image', 'recall_image', 'F1_image', 'accuracy_image',
                                            'AUROC_image', 'AUPRC_image', 'TP_patient', 'TN_patient', 'FN_patient',
                                            'FP_patient', 'precision_patient', 'recall_patient', 'F1_patient',
                                            'accuracy_patient', 'AUROC_patient', 'AUPRC_patient']]

    summ = summ.round(5)

    summ.to_csv('../Summary.csv', index=False)


