import os
import numpy as np
import pandas as pd

from math import sqrt

# generate ajacency matrix
location = '/staging/biology/u1307362/CPTAC-COAD_tile_location'
features = '/staging/biology/u1307362/CPTAC-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'
adjacency = '/staging/biology/u1307362/adjacency/CPTAC-COAD'

for csv in os.listdir(features):
    df = pd.read_csv(os.path.join(features, csv))
    patch_list = df['Unnamed: 0'].values.tolist()
    adj = np.zeros((len(df), len(df)))
    coord = pd.read_csv(os.path.join(location, csv[:-4] + '.tsv'), sep = '\t')
    for index_r, patch_r in enumerate(patch_list):
        filt_r = (coord['Tile'] == patch_r)
        loc_r = coord.loc[filt_r, ['Row', 'Column']].values
        loc_r = np.array(loc_r).flatten().tolist()
        x_i, y_i = loc_r[0], loc_r[1]
        for index_c, patch_c in enumerate(patch_list):
            filt_c = (coord['Tile'] == patch_c)
            loc_c = coord.loc[filt_c, ['Row', 'Column']].values
            loc_c = np.array(loc_c).flatten().tolist()
            x_j, y_j = loc_c[0], loc_c[1]
            d = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if d == 1 or d == (2 ** 0.5):
                adj[index_r][index_c] = 1
    np.save(os.path.join(adjacency, csv[:-4] + '.npy'), adj)