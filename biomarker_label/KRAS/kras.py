import os
import math
import numpy as np
import pandas as pd

# log file
log_fw = open("info.txt", 'w')
def log(text):
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# import cBioPortal mutation data
coadread_tcga = pd.read_csv('/home/u1307362/cBioPortal/mutation/coadread_tcga/KRAS.tsv', sep = '\t')
pan_can_atlas_2018 = pd.read_csv('/home/u1307362/cBioPortal/mutation/coadread_tcga_pan_can_atlas_2018/KRAS.tsv', sep = '\t')
pub = pd.read_csv('/home/u1307362/cBioPortal/mutation/coadread_tcga_pub/KRAS.tsv', sep = '\t')
cptac = pd.read_csv('/home/u1307362/cBioPortal/mutation/coad_cptac_2019/KRAS.tsv', sep = '\t')

coadread_tcga = coadread_tcga['Sample ID'].values
coadread_tcga = [i[:12] for i in coadread_tcga]

pan_can_atlas_2018 = pan_can_atlas_2018['Sample ID'].values
pan_can_atlas_2018 = [i[:12] for i in pan_can_atlas_2018]

pub = pub['Sample ID'].values
pub = [i[:12] for i in pub]

cptac = cptac['Sample ID'].values

# import CPTAC-COAD label barcode
cptac_barcode = pd.read_csv('/home/u1307362/cBioPortal/clinical_data/coad_cptac_2019_clinical_data.tsv', sep = '\t')
cptac_barcode = cptac_barcode['Patient ID'].values

# TCGA-COAD (435 slides, 428 patients, 428 labels, 244 wildtype, 184 mutation)
ids = np.load('/home/u1307362/wsi/barcode/TCGA-COAD.npy')

log('TCGA-COAD')
label = {}
for i in ids:
    if i in coadread_tcga or i in pan_can_atlas_2018 or i in pub:
        label[i] = 1
    else:
        label[i] = 0

np.save('TCGA-COAD.npy', label)

# CPTAC-COAD (372 slides, 178 patients, 110 labels, 75 wildtype, 35 mutation)
ids = np.load('/home/u1307362/wsi/barcode/CPTAC-COAD.npy')

log('CPTAC-COAD')
label = {}
for i in ids:
    if i in cptac and i in cptac_barcode:
        label[i] = 1
    elif i not in cptac and i in cptac_barcode:
        label[i] = 0

np.save('CPTAC-COAD.npy', label)