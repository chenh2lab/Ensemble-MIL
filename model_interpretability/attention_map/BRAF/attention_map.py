import os
import slideio
import math
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

# log file
log_fw = open("info.txt", 'w')
def log(text):
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# import data
weight_root = '/staging/biology/u1307362/attention_weight/BRAF/simclr'
location_root = '/staging/biology/u1307362/pyhist/TCGA-COAD'
csv_root = '/staging/biology/u1307362/TCGA-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'
wsi_root = '/staging/biology/u1307362/TCGA-COAD_WSI/raw_diagnosis_slides'
target_root = '/staging/biology/u1307362/attention_map/BRAF/simclr'

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# attention map
for fold in range(5):
    log(f'FOLD {fold}')
    log('---------------------------------------------')
    for svs in os.listdir(os.path.join(weight_root, f'fold_{fold}')):
        try:
            log(f'processing : {svs}')
            # import attention weight
            attn_weight = np.load(os.path.join(weight_root, f'fold_{fold}', svs))
            # import feature csv
            tumor = pd.read_csv(os.path.join(csv_root, svs[:-4] + '.csv'))
            weight_dict = dict(zip(tumor['Unnamed: 0'].tolist(), attn_weight))

            # import tile location 
            location = pd.read_csv(os.path.join(location_root, svs[:-4], 'tile_selection.tsv'), sep = '\t')
            location_x = location[location['Row'] == 0]
            width = location_x['Width'].sum()
            location_y = location[location['Column'] == 0]
            height = location_y['Height'].sum()

            # import wsi
            slide = slideio.open_slide(os.path.join(wsi_root, svs[:-4] + '.svs'), 'SVS')
            scene = slide.get_scene(0)
            img = scene.read_block(size = (int(scene.rect[2] * 0.5), int(scene.rect[3] * 0.5)))
            img = Image.fromarray(img)

            # initialize heatmap
            heatmap = np.full((height, width), math.log(min(list(weight_dict.values())), 10))

            for i in weight_dict:
                x = int(location[location['Tile'] == i]['Row'].values)
                y = int(location[location['Tile'] == i]['Column'].values)
                heatmap[x * 512:(x + 1) * 512, y * 512:(y + 1) * 512] = math.log(weight_dict[i], 10)

            fig, ax = plt.subplots()
            im = ax.imshow(img)
            im = ax.imshow(heatmap, alpha = 0.8, cmap = 'jet')
            ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.1)
            crb = plt.colorbar(im, cax = cax)
            crb.set_ticks([math.log(min(list(weight_dict.values())), 10), math.log(max(list(weight_dict.values())), 10)])
            crb.set_ticklabels(['Low', 'High'])
            crb.ax.set_ylabel('Attention weight', rotation = 270, labelpad = 3)
            crb.ax.tick_params(labelsize = 8)
            crb.ax.yaxis.label.set_size(8)
            plt.savefig(os.path.join(target_root, f'fold_{fold}', svs[:-4] + '.png'), bbox_inches = 'tight')
            plt.clf()
            plt.close(fig)

            # claen memory
            del attn_weight, tumor, weight_dict, location, slide, scene, img, heatmap
            gc.collect()
        except Exception as e:
            log(f'Error processing {svs}: {e}')