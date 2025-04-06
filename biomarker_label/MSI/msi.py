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

# import cBioPortal clinical data, 255 patients are duplicated
pan_can_atlas_2018 = pd.read_csv('/home/u1307362/cBioPortal/clinical_data/coadread_tcga_pan_can_atlas_2018_clinical_data.tsv', sep = '\t')
pub = pd.read_csv('/home/u1307362/cBioPortal/clinical_data/coadread_tcga_pub_clinical_data.tsv', sep = '\t')
cptac = pd.read_csv('/home/u1307362/cBioPortal/clinical_data/coad_cptac_2019_clinical_data.tsv', sep = '\t')

pan_can_atlas_2018 = pan_can_atlas_2018[['Patient ID', 'MSIsensor Score']]
pan_can_atlas_2018 = dict([(id, s) for id, s in zip(pan_can_atlas_2018['Patient ID'], pan_can_atlas_2018['MSIsensor Score'])])
for id in pan_can_atlas_2018:
    if pan_can_atlas_2018[id] >= 10:
        pan_can_atlas_2018[id] = 1
    elif pan_can_atlas_2018[id] < 10:
        pan_can_atlas_2018[id] = 0

pub = pub[['Patient ID', 'MSI Status']]
pub = dict([(id, s) for id, s in zip(pub['Patient ID'], pub['MSI Status'])])
for id in pub:
    if pub[id] == 'MSI-H':
        pub[id] = 1
    elif pub[id] == 'MSI-L' or 'MSS':
        pub[id] = 0

cptac = cptac[['Patient ID', 'MSI Status']]
cptac = dict([(id, s) for id, s in zip(cptac['Patient ID'], cptac['MSI Status'])])
for id in cptac:
    if cptac[id] == 'MSI-H':
        cptac[id] = 1
    elif cptac[id] == 'MSI-L' or 'MSS':
        cptac[id] = 0

# TCGA-COAD (435 slides, 428 patients, 421 labels, 350 MSS, 71 MSI-H)
ids = np.load('/home/u1307362/wsi/barcode/TCGA-COAD.npy')

log('TCGA-COAD')
label = {}
for i in ids:
    if i in pan_can_atlas_2018 and i in pub:
        if pan_can_atlas_2018[i] == pub[i]:
            label[i] = pub[i]
        elif pan_can_atlas_2018[i] != pub[i]:
            if math.isnan(pan_can_atlas_2018[i]):
                label[i] = pub[i]
            else:
                label[i] =  pan_can_atlas_2018[i] # if not consistent, pan_can_atlas_2018 is dominant
    elif i in pan_can_atlas_2018 and i not in pub:
        if math.isnan(pan_can_atlas_2018[i]):
            log(f'{i} without label')
        else:
            label[i] =  pan_can_atlas_2018[i]
    elif i in pub and i not in pan_can_atlas_2018:
        label[i] = pub[i]
    elif i not in pub and i not in pan_can_atlas_2018:
        log(f'{i} without label')

np.save('TCGA-COAD.npy', label)

# CPTAC-COAD (372 slides, 178 patients, 110 labels, 86 MSS, 24 MSI-H)
ids = np.load('/home/u1307362/wsi/barcode/CPTAC-COAD.npy')

log('CPTAC-COAD')
label = {}
for i in ids:
    if i in cptac:
        label[i] = cptac[i]
    elif i not in cptac:
        log(f'{i} without label')

np.save('CPTAC-COAD.npy', label)