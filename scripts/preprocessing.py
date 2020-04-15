"""
Preprocess images

Created on 04/10/2020

@author: RH

"""

import numpy as np
import os
from PIL import Image
import pandas as pd

for w in ['CT_COVID', 'CT_nonCOVID']:
    for m in os.listdir('../images/raw/{}'.format(w)):
        if '.png' or '.jpg' in m:
            im = Image.open('../images/raw/{}/{}'.format(w, m))
            oldsize = im.size
            new_size = (np.max(oldsize), np.max(oldsize))
            new_im = Image.new("RGB", new_size)
            new_im.paste(im, (int((new_size[0] - oldsize[0]) / 2), int((new_size[1] - oldsize[1]) / 2)))
            new_im = new_im.resize((299, 299))
            new_im.save('../images/{}/{}.png'.format(w, m[:-4]))

pos = pd.read_csv('../COVID-CT-MetaInfo.csv', header=0)
posa = pos.loc[pos['image'].str.contains('.png')]
posb = pos.loc[~pos['image'].str.contains('.png')]
posb['image'] = posb['image']+'.png'
posfinal = pd.concat([posa, posb], axis=0)
posfinal['patient'] = posfinal['patient'].str.replace(' ', '_')

posfinal.to_csv('../COVID-CT-MetaInfo.csv', index=False)
neg = pd.read_csv('../NonCOVID-CT-MetaInfo.csv', header=0)
neg['image'] = neg['image'].str.replace('jpg', 'png')
neg.to_csv('../NonCOVID-CT-MetaInfo.csv', index=False)


