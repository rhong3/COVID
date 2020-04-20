"""
pytorch main to use pretrained models

Created on 04/13/2020

@author: RH
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = pd.read_csv('../Summary.csv', header=0)
results['name'] = results['model']+'_'+results['state'].str.get(0)

# # All in one
# a = results[['name', 'AUROC_image']]
# a['metrics'] = 'AUROC_image'
# a = a.rename(columns={'AUROC_image': 'value'})
# b = results[['name', 'AUPRC_image']]
# b['metrics'] = 'AUPRC_image'
# b = b.rename(columns={'AUPRC_image': 'value'})
# c = results[['name', 'accuracy_image']]
# c['metrics'] = 'accuracy_image'
# c = c.rename(columns={'accuracy_image': 'value'})
# d = results[['name', 'AUROC_patient']]
# d['metrics'] = 'AUROC_patient'
# d = d.rename(columns={'AUROC_patient': 'value'})
# e = results[['name', 'AUPRC_patient']]
# e['metrics'] = 'AUPRC_patient'
# e = e.rename(columns={'AUPRC_patient': 'value'})
# f = results[['name', 'accuracy_patient']]
# f['metrics'] = 'accuracy_patient'
# f = f.rename(columns={'accuracy_patient': 'value'})
#
# recombined = pd.concat([a, d, b, e, c, f])
#
# grid = sns.catplot(x="name", y="value", hue="metrics", kind="bar", data=recombined, height=5, aspect=4)
# grid.set_xticklabels(rotation=45, horizontalalignment='right', fontsize='medium', fontweight='light')
# plt.show()
# plt.savefig('../summary.png', pad_inches=1)