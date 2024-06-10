# Prediction of generalizable biomarker from colorectal adenocarcinoma whole slide images via integrated deep learning pipeline
Construct the integrated deep learning pipeline to predict BRAF, KRAS and MSI directly from pathological images in colorectal cancer
## Pipeline
* **Stage 1: feature extractor training**
![Pipeline](./imgs/pipeline_1.jpg)
* **Stage 2: biomarker prediction model training and internal testing**
![Pipeline](./imgs/pipeline_2.jpg)
* **Stage 3: biomarker prediction model external testing**
![Pipeline](./imgs/pipeline_3.jpg)
## Prerequisites
* Operating system: CentOS 7.8
* Programmimg language: Python 3, shell script
* Hardware: NVIDIA Tesla V100-PCIE-32GB
## Installation
* clone this repository:
```bash
git clone https://github.com/chenh2lab/TGY_2023
cd TGY_2023

* setup environment
```bash
cd envs
conda env create -f PyTorch.yml
conda env create -f PyHIST.yml
