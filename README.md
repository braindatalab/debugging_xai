# Feature salience - not task-informativeness - drives machine learning model explanations

This repository contains the code and data pipelines for our project, evaluating the ability of various Explainable AI (XAI) methods to detect and distinguish types of variables (confounding effects, suppressors, and outlying salient features), through watermarks and image lightness manipulations. The paper is currently under review, but a pre-print is available on [arXiv](https://arxiv.org/abs/2602.09238)

## 1. System Requirements

### Hardware requirements
The scripts require a standard computer with enough RAM to support the data processing operations. For optimal performance, we recommend a machine with the following specs:
* RAM: 16+ GB
* CPU: 4+ cores, 3.3+ GHz/core
* GPU: An NVIDIA GPU with CUDA support is highly recommended for model training, but not strictly required for the demo.

### Software requirements
#### OS Requirements
This software is supported for macOS and Linux. The software has been tested on the following systems:
* macOS: Sequoia (15.7.3)
* Linux: Ubuntu 24.04

#### Python Dependencies
The software requires **Python 3.11.8**. All required packages and their versions are detailed in the provided `environment.yml` file. Main dependencies include:
* `torch`, `torchvision`, `torchmetrics`
* `captum`, `zennit`
* `scikit-learn`, `scikit-image`, `pot`
* `matplotlib`, `seaborn`, `pandas`, `opencv-python`

## 2. Installation Guide

It is recommended to install the dependencies using `conda` (or `mamba`).

1. Clone the repository:
```bash
git clone https://github.com/braindatalab/debugging_xai.git
cd debugging_xai
```

2. Create and activate the conda environment:
```bash
conda env create --name envname --file=environment.yml
conda activate envname
```

**Typical install time:** 5-10 minutes on a normal desktop computer.

## 3. Demo

To test the software functionality, we provide a demo using a small, simulated sample dataset located in `demo/`

To run the complete pipeline on the demo data, we have provided an automated script that copies a small sample of images (if you have the full dataset locally) or expects 20 sample images to be placed in `demo/images/cat` and `demo/images/dog`. It will then run data generation, model training, and evaluation automatically.

Run the demo script from the root of the repository:

```bash
./demo/setup_and_run.sh
```

Alternatively, you can run the steps manually as they are laid out inside `demo/setup_and_run.sh`.

**Expected output:** 
The pipeline will output:
* Generated artifacts saved to `demo/artifacts/`
* Trained model weights saved to `demo/models/`
* The final explanation metrics (energy scores, etc.) saved to `demo/results/`

**Expected run time for demo:** ~5-10 minutes on a normal desktop computer.

## 4. Instructions for Use

To run the experiments on the full datasets, follow the instructions below. 

### Data Preparation
The complete datasets are required for the full experiments. 
* **Cats and Dogs:** Available internally on MS Teams, or from [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog) (Combine training and testing images into `./images/dog/` or `./images/cat/`).
* **COCO 2017:** Requires the [COCO 2017 train and val data as well as annotations](https://cocodataset.org/#download) placed in the directory structure: `project/coco/annotations/` and `project/coco/train2017/`.

### Watermark Experiments

**1. Generation**
Generate the data for experiments using:
```bash
python -m watermarks.generate_watermarks --split-index {split} --scale {zero_one|neg_one_one} 
```
* `{split}`: Integer [0..9] defining the random seed data shuffle.
* `--scale`: `'neg_one_one'` to scale data to `[-1,1]`, or `'zero_one'` to use standard `[0,1]` min-max scaling.

*For variable-position watermarks, use:* `--position variable` flag in the same script.

**2. Training**
```bash
python -m watermarks.train_watermarks_server --split-index {split} --base {all|confounder|suppressor|no_watermark} 
```
The `--base` argument refers to whether to train `all` three model variants, or just one specific type. 

**3. Evaluation**
All XAI method execution and resulting energy metric calculations:
```bash
python -m watermarks.calculate_energy --split-index {split} --seed-index {seed_ind} --position {fixed|variable}  
```
* `{seed_ind}`: Integer [0..9] defining which trained model seed to use.
* `--position`: Use variable-position watermark data if set.

### COCO Lightness Experiments

**1. Generation**
```bash
python -m coco.generate_coco_splits {split} {rescaled} 
```
If `{rescaled}` is specified to `rescaled`, it applies `[-1,1]` scaling to the HLS lightness channel only.

**2. Training**
```bash
python -m coco.coco_train_server_splits {split} {conf|sup|norm} 
```
Where `conf` trains the confounder model, `sup` the suppressor, and `norm` the normalised brightness model.

**3. Evaluation**
```bash
python -m coco.calculate_coco_energy {split} {model_ind}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
