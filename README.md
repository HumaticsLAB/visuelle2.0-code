# The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark

The official pytorch implementation of models discussed in [The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark](https://arxiv.org/abs/2204.06972v1)
paper.

Accepted as oral at the 5th Workshop on Computer Vision for Fashion, Art, and Design @ CVPR22

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
git clone https://github.com/HumaticsLAB/visuelle2.0-code.git
cd visuelle2.0-code

unset INSTALL_DIR
```
## Dataset

**VISUELLE2** dataset is publicly available to download [here](https://forms.gle/8Sk431AsEgCot9Kv5). Please download and extract it inside the root folder. A more accurate description of the dataset inside its [official page](https://humaticslab.github.io/forecasting/visuelle).  

## Run Naive and Simple Exponential Smoothing baselines

```bash
# Naive Method
python forecast_stat.py --method naive --use_teacher_forcing 1  #SO-fore2−1
python forecast_stat.py --method naive --use_teacher_forcing 0  #SO-fore2−10

# Simple Exponential Smoothing Method
python forecast_stat.py --method ses --use_teacher_forcing 1  #SO-fore2−1
python forecast_stat.py --method ses --use_teacher_forcing 0  #SO-fore2−10
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

## Citation
If you use **VISUELLE2** dataset or this paper implementation, please cite the following papers.

```
@inproceedings{skenderi2022multi,
  title={The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark},
  author={Skenderi, Geri and Joppi, Christian and Denitto, Matteo and Scarpa, Berniero and Cristani, Marco},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2022},
}
```
