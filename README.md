# The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark

The official pytorch implementation of models discussed in [The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark](https://arxiv.org/abs/2204.06972v1) paper.

Accepted as oral at the 5th Workshop on Computer Vision for Fashion, Art, and Design @ CVPR22

## Installation

We suggest the use of VirtualEnv.

```bash
git clone https://github.com/HumaticsLAB/visuelle2.0-code.git
cd visuelle2.0-code
virtualenv mmrnn_venv # If you don't have virtualenv, you can install it by using "pip install virtualenv"
source mmrnn_venv/bin/activate # or mmrnn_venv\Scripts\activate.bat # If you're running on Windows

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install pytorch-lightning==1.6.5
pip install pandas numpy opencv-python Pillow scikit-image scikit-learn scipy tqdm fairseq wandb statsmodels

# To deactivate the virtual env simply execute "deactivate"
```
## Dataset

**VISUELLE2** dataset is publicly available [here](https://forms.gle/8Sk431AsEgCot9Kv5). Please download and extract it inside the root folder. A more detailed description of the dataset can be found in the [project webpage](https://humaticslab.github.io/forecasting/visuelle).  

## Run Naive and Simple Exponential Smoothing baselines

```bash
# Naive Method
python forecast_stat.py --method naive --use_teacher_forcing 1  #SO-fore2−1
python forecast_stat.py --method naive --use_teacher_forcing 0  #SO-fore2−10

# Simple Exponential Smoothing Method
python forecast_stat.py --method ses --use_teacher_forcing 1  #SO-fore2−1
python forecast_stat.py --method ses --use_teacher_forcing 0  #SO-fore2−10
```

## Run SO-fore2−1
```bash
python train_dl.py --task_mode 0
python forecast_dl.py --task_mode 0 --ckpt_path <ckpt_path>
```

## Run SO-fore2−10

```bash
python train_dl.py --task_mode 1
python forecast_dl.py --task_mode 1 --ckpt_path <ckpt_path>
```

## Run Demand SO-fore
```bash
python train_dl.py --new_product 1
python forecast_dl.py --new_product 1 --ckpt_path <ckpt_path>
```
## Model zoo
You can find the pretrained models [here](bit.ly/3iu2YyL).

## Citation
If you use the **VISUELLE2.0** dataset or this particular implementation, please cite the following paper:

```
@inproceedings{skenderi2022multi,
  title={The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark},
  author={Skenderi, Geri and Joppi, Christian and Denitto, Matteo and Scarpa, Berniero and Cristani, Marco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2241--2246},
  year={2022}
}
```
