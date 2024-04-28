# HKLUT: Hundred-Kilobyte Lookup Tables
This repository serves as the official code release of the IJCAI24 paper: **Hundred-Kilobyte Lookup Tables for Efficient Single-Image Super-Resolution**
![](assets/logo.png?v=1&type=image)
<div align="center">
    <a href="https://arxiv.org/abs/2312.06101"><img src="https://img.shields.io/badge/Arxiv-2312.06101-b31b1b.svg?logo=arXiv" alt=""></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt=""></a>
</div>
<br>
<div align="center">
<strong><a href="https://harr7y.github.io/"><u>Binxiao Huang</u></a><sup>1†</sup></strong>, <strong><a href="https://www.linkedin.com/in/jason-chun-lok-li-0590b3166/"><u>Jason Chun Lok Li</u></a><sup>1†</sup></strong>, Ran Jie<sup>1</sup>, Boyu Li<sup>1</sup>, Jiajun Zhou<sup>1</sup>, Dahai Yu<sup>2</sup>, Ngai Wong<sup>1</sup>
</div>
<br>
<div align="center">
<strong><sup>†</sup>Contributed Equally</strong>
</div>
<br>
<div align="center">
<sup>1</sup>The University of Hong Kong,   <sup>2</sup>TCL Corporate Research
</div>
<div align="center">
</div>

## Usage
### Dependency
```
conda create -n hklut python=3.8
conda activate hklut
pip install -r requirements.txt
```
### 💾Dataset
Please follow the instructions below to download the corresponding datasets and place them in the specified folder structure.
##### Training set
The DIV2K dataset can be download from the [offical website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
```
data/train/DIV2K/
            /HR/*.png
            /LR/{X2, X3, X4, X8}/*.png
```
##### Testing set
Please follow the instructions on [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets) to download the SR benchmark datasets.
```
data/test/{Set5, Set14, B100, Urban100, Manga109}/
            /HR/*.png
            /LR_bicubic/{X2, X3, X4, X8}/*.png
```

### 🛠Quick Start
We have prepared bash scripts for each stage for convenient usage.
##### Training
```
./scripts/train.sh
```

##### Tranfer to LUTs
```
./scripts/transfer.sh
```

##### ️Testing
```
./scripts/test.sh
```

## 📝Citation

If HKLUT has been beneficial to your research and applications, please acknowledge it by using this BibTeX citation:
```
@article{huang2023hundred,
  title={Hundred-Kilobyte Lookup Tables for Efficient Single-Image Super-Resolution},
  author={Huang, Binxiao and Li, Jason Chun Lok and Ran, Jie and Li, Boyu and Zhou, Jiajun and Yu, Dahai and Wong, Ngai},
  journal={arXiv preprint arXiv:2312.06101},
  year={2023}
}

```


## 🙏🏼Acknowledgements
Our code is built upon [MuLUT](https://github.com/ddlee-cn/MuLUT) and [SPLUT](https://github.com/zhjy2016/SPLUT). We sincerely appreciate their contributions to open-source.