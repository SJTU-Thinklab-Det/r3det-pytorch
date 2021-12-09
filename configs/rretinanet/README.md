# [RRetinaNet](https://arxiv.org/pdf/1708.02002.pdf)


## Introduction

![illustration](https://upload-images.jianshu.io/upload_images/2141706-b7ac3a85fdc1c207.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.


## Results and Models

### DOTA1.0

|    Backbone   |    mAP   | Angle | lr schd | Ms | Rotate | Batch Size | Configs | Download |
|:------------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ResNet50 (1024,1024,200) | 65.19 | v1 | 1x | - | - | 2 | rretinanet_hbb_r50_fpn_1x_dota_v1.py |  [Baidu:0518](https://pan.baidu.com/s/1P7SoV5fnNuDtA4DrcEAXFw)/[Google]()
| ResNet50 (1024,1024,200) | 68.19 | v3 | 1x | - | - | 2 | rretinanet_obb_r50_fpn_1x_dota_v3.py |  [Baidu:0518](https://pan.baidu.com/s/1ijkb0y_yAaicT-Z9_ljKeA)/[Google](https://drive.google.com/drive/folders/1CeD3QPTQRRSI7WKMwWE3EUWhzD2qN4e4?usp=sharing)
| ResNet50 (1024,1024,200) | 68.64 | v2 | 1x | - | - | 2 | rretinanet_obb_r50_fpn_1x_dota_v2.py |  [Baidu:0518](https://pan.baidu.com/s/14o4sNxzfWQj1oGFjBzX8Kg)/[Google]()
| ResNet50 (1024,1024,200) | 77.45 | v3 | 1x | √ | √ | 2 | rretinanet_obb_r50_fpn_1x_dota_ms_rr_v3.py |  [Baidu:0518](https://pan.baidu.com/s/1iuyrMOOLSSJUcsxlR92CtA)/[Google]()


## Citation
```
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
