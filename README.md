SIESTA: Efficient Online Continual Learning with Sleep
=====================================
This is a PyTorch implementation of the SIESTA algorithm from our TMLR-2023 paper. An [arXiv pre-print](https://arxiv.org/abs/2303.10725) of our paper is available.

![SIESTA](./siesta_overview.png)

SIESTA is a wake/sleep based online continual learning algorithm and designed to be computationally efficient for resource-constrained applications such as edge devices, mobile phones, robots, AR-VR and so on. It is capable of rapid online learning and inference while awake, but has periods of sleep where it performs offline memory consolidation.


## Pre-trained MobileNetV3-L and OPQ Models
Download pre-trained MobileNetV3-L and Optimized Product Quantization (OPQ) models form [this link](https://drive.google.com/drive/folders/1gPg_FxsvmUj-Mwis_uASy4qcsKMjzCrJ?usp=share_link).

## Acknowledgements
Thanks for the great code base from [REMIND](https://github.com/tyler-hayes/REMIND)

## Citation
If using this code, please cite our paper.
```
@article{harun2023siesta,
title={{SIESTA}: Efficient Online Continual Learning with Sleep},
author={Md Yousuf Harun and Jhair Gallardo and Tyler L. Hayes and Ronald Kemker and Christopher Kanan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=MqDVlBWRRV},
note={}
}
```
