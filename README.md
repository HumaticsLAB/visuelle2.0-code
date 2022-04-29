# The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark

The official pytorch implementation of models discussed in [The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark](https://arxiv.org/abs/2204.06972v1)
paper.
## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv mmrnn_venv
source mmrnn_venv/bin/activate
# mmrnn_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/AttentionBasedMultiModalRNN.git
cd AttentionBasedMultiModalRNN
mkdir dataset

unset INSTALL_DIR
```
## Dataset

**VISUELLE2** dataset is publicly available to download [here](https://forms.gle/8Sk431AsEgCot9Kv5). Please download and extract it inside the dataset folder. A more accurate description of the dataset inside its [official page](https://humaticslab.github.io/forecasting/visuelle).  

## Run Naive and Simple Exponential Smoothing baselines

```bash
python forecast_stat.py 
```
## Run SO-fore2−10

```bash
python train_dl.py --task_mode 0
python forecast_dl.py --task_mode 0 --ckpt_path <ckpt_path>
```
## Run SO-fore2−1
```bash
python train_dl.py --task_mode 1
python forecast_dl.py --task_mode 1 --ckpt_path <ckpt_path>
```
## Run Demand SO-fore
```bash
python train_dl.py --new_product 1
python forecast_dl.py --new_product 1 --ckpt_path <ckpt_path>
```
