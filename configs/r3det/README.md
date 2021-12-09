# [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612)


## Introduction

![illustration](https://upload-images.jianshu.io/upload_images/2141706-72c9d1c698102162.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Rotation detection is a challenging task due to the difficulties of locating the multi-angle objects and separating them effectively from the background. Though considerable progress has been made, for practical settings, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end refined single-stage rotation detector for fast and accurate object detection by using a progressive regression approach from coarse to fine granularity. Considering the shortcoming of feature misalignment in existing refined single stage detector, we design a feature refinement module to improve detection performance by getting more accurate features. The key idea of feature refinement module is to re-encode the position information of the current refined bounding box to the corresponding feature points through pixel-wise feature interpolation to realize feature reconstruction and alignment. For more accurate rotation estimation, an approximate SkewIoU loss is proposed to solve the problem that the calculation of SkewIoU is not derivable. Experiments on three popular remote sensing public datasets DOTA, HRSC2016, UCAS-AOD as well as one scene text dataset ICDAR2015 show the effectiveness of our approach.

## Results and models

### DOTA1.0

|    Backbone   |   mAP   | Angle | lr schd | Ms | Rotate | Batch Size | Configs | Download  |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ResNet50 (1024,1024,200) | 65.19 | v1 | 1x | - | - | 2 | rretinanet_hbb_r50_fpn_1x_dota_v1.py |  [Baidu:0518](https://pan.baidu.com/s/1P7SoV5fnNuDtA4DrcEAXFw)/[Google]()
| ResNet50 (1024,1024,200) | 70.41 | v1 | 1x | - | - | 2 | r3det_r50_fpn_1x_dota_v1.py | [Baidu:0518](https://pan.baidu.com/s/1ECNAzE3xaXXO7Pj2p_bLDw)/[Google]()
| ResNet50 (1024,1024,200) | 70.86 | v1 | 1x | - | - | 2 | r3det_tiny_r50_fpn_1x_dota_v1.py | [Baidu:0518](https://pan.baidu.com/s/1kWg-bz2KjDcI-s_IWvUE6A)/[Google]()


## Citation
```
@inproceedings{yang2021r3det,
    title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
    author={Yang, Xue and Yan, Junchi and Feng, Ziming and He, Tao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={35},
    number={4},
    pages={3163--3171},
    year={2021}
}

```
