"""
Preprocess images

Created on 04/10/2020

@author: RH

"""

import numpy as np
import os
from PIL import Image

for w in ['CT_COVID', 'CT_nonCOVID']:
    for m in os.listdir('../images/raw/{}'.format(w)):
        if '.png' in m:
            im = Image.open('../images/raw/{}/{}'.format(w, m))
            oldsize = im.size
            new_size = (np.max(oldsize), np.max(oldsize))
            new_im = Image.new("RGB", new_size)
            new_im.paste(im, (int((new_size[0] - oldsize[0]) / 2), int((new_size[1] - oldsize[1]) / 2)))
            new_im = new_im.resize((299, 299))
            new_im.save('../images/{}/{}'.format(w, m))

