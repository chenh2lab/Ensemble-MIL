# Prediction of generalizable biomarker from colorectal adenocarcinoma whole slide images via integrated deep learning pipeline
Construct the integrated deep learning pipeline to predict BRAF, KRAS and MSI directly from pathological images in colorectal cancer.
## Pipeline
The pipeline commence with image preprocessing (Part A). Tumor detection model based on VGG19 architecture are trained to classify tumor patches from raw patches (Part B). Tumor patches form training dataset for the SimCLR self- supervised learning model. Features of the tumor patches are extracted by the use of the self-supervised learning encoder, ResNet-50 (Part D). We adopt three different biomarker prediction models, namely: attention-based model, transformer-based model and gnn-based model. Eventually, majority voting aggregate each model prediction to enhance model performance.
![Pipeline](.pipeline.png)
## Prerequisites
* Operating system: CentOS 7.8
* Programmimg language: Python, sh
* Hardware: NVIDIA Tesla V100-PCIE-32GB
## Data download
### WSI & biomarker label
* Download `TCGA-COAD` whole slide images from [GDC portal](https://portal.gdc.cancer.gov) for model training and internal testing.
* Download `CPTAC-COAD` whole slide images from [CIP Cancer Imaging Program](https://www.cancerimagingarchive.net/collection/cptac-coad/) for model external testing.
* Download corresponding biomarker label data for TCGA-COAD and CPTAC-COAD from [cBioPortal](https://www.cbioportal.org).
### Patch-level tissue category dataset
* Download `TCGA-HE-89K` dataset from [here](https://zenodo.org/records/4024676) for tumor detection model training.
## Biomarker label
There are **BRAF**, **KRAS** and **MSI** biomarker annotation in `biomarker_label` directory.
## Image preprocessing
Clone [PyHIST](https://pyhist.readthedocs.io/en/latest/) repository to setup environment.
```
git clone https://github.com/manuel-munoz-aguirre/PyHIST.git
```
Cut single WSI into 512 * 512 pixels patches.
```
sh ./image_preprocessing/patch_generation/pyhist.sh
```
## Tumor detection model
Use `TCGA-HE-89K` dataset to train a VGG19-based tumor detection model to classify tumor patches from all of the patches in WSI.
```
# training & internal testing
python train_internal_test_vgg19.py

# external testing
python external_test_vgg19.py
```
```
# tumor patch prediction on cohorts
python ./tumor_detection_model/inference/TCGA-COAD/prediction.py
python ./tumor_detection_model/inference/CPTAC-COAD/prediction.py
```
## Tumor detection model interpretability
Grad-CAM and tissue category map enhance the accuracy of the tumor detection model's classification results.
```
# Grad-CAM
python ./model_interpretability/grad_cam/grad_cam.py

# tissue category map
python ./model_interpretability/tissue_category_map/TCGA-COAD/tissue_category_map.py
python ./model_interpretability/tissue_category_map/CPTAC-COAD/tissue_category_map.py
```
## Feature extraction
**SimCLR** is a framework to train feature extractor based on Resnet50.
```
# training feature extractor
python ./feature_extractor/train_simclr.py
```
```
# feature extraction on cohorts
python ./feature_extractor/inference/TCGA-COAD/extraction.py
python ./feature_extractor/inference/CPTAC-COAD/extraction.py
```
## Biomarker prediction model
feature matrix of a WSI is utilized for biomarker prediction model establishment.  
For **GnnMIL**, it should construct graph adjacency matrix first. The related scripts are in `./graph`.
```
# training & internal testing
python ./biomarker_prediction/BRAF/attMIL/train_internal_test_attention.py
python ./biomarker_prediction/BRAF/gnnMIL/train_internal_test_gnn.py
python ./biomarker_prediction/BRAF/transMIL/train_internal_test_transformer.py

# external testing
python ./biomarker_prediction/BRAF/attMIL/external_test_attention.py
python ./biomarker_prediction/BRAF/gnnMIL/external_test_gnn.py
python ./biomarker_prediction/BRAF/transMIL/external_test_transformer.py
```
The execution method for the other biomarkers is the same as described above.

## Biomarker prediction model (ensembe)
Use majority voting to merge the results of **attMIL**, **gnnMIL** and **transMIL**.
```
# internal testing
python ./biomarker_prediction_ensemble/BRAF/internal_test_ensemble.py

# external testing
python ./biomarker_prediction_ensemble/BRAF/external_test_ensemble.py
``` 
The execution method for the other biomarkers is the same as described above.
