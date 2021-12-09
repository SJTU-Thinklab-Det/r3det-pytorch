# R3Det


## Installation


```shell
# install mmdetection first if you haven't installed it yet. (Refer to mmdetection for details.)
pip install mmdet==2.19

# install r3det (Compiling rotated ops is a little time-consuming.)
pip install -r requirements.txt
pip install -v -e .

```

- It is best to use opencv-python greater than 4.5.1 because its angle representation has been changed in 4.5.1. The following experiments are all run with 4.5.3.


## Quick Start

Please change [path](configs/rretinanet/rretinanet_obb_r50_fpn_1x_dota_v3.py#L3) in configs to your data path.
```shell
# train
CUDA_VISIBLE_DEVICES=0 PORT=29500 \
./tools/dist_train.sh configs/rretinanet/rretinanet_obb_r50_fpn_1x_dota_v3.py 1

# submission
CUDA_VISIBLE_DEVICES=0 PORT=29500 \
./tools/dist_test.sh configs/rretinanet/rretinanet_obb_r50_fpn_1x_dota_v3.py \
        work_dirs/rretinanet_obb_r50_fpn_1x_dota_v3/epoch_12.pth 1 --format-only\
        --eval-options submission_dir=work_dirs/rretinanet_obb_r50_fpn_1x_dota_v3/Task1_results
```

For DOTA dataset, please crop the original images into 1024×1024 patches with an overlap of 200 by run
```shell
python tools/split/img_split.py --base_json \
       tools/split/split_configs/split_configs/dota1_0/ss_trainval.json

python tools/split/img_split.py --base_json \
       tools/split/split_configs/dota1_0/ss_test.json

```
Please change path in [ss_trainval.json](./tools/split/split_configs/dota1_0/ss_trainval.json#L4-L11), [ss_test.json](./tools/split/split_configs/dota1_0/ss_test.json#L5)  to your path. (Forked from [BboxToolkit](https://github.com/jbwang1997/BboxToolkit), which is faster then DOTA_Devkit.)





### Angle Representations
Three angle representations are built-in, which can freely switch in the config.

- `v1` (from R<sup>3</sup>Det): [-PI/2, 0)
- `v2` (from S<sup>2</sup>ANet): [-Pi/4, 3PI/4)
- `v3` (from OBBDetection): [-PI/2, PI/2)

The differences of the three angle representations are reflected in poly2obb, obb2poly, obb2xyxy, obb2hbb, hbb2obb, etc. [[More](./r3det/core/bbox/rtransforms.py)], And according to the above three papers, the coders of them are different.

- DeltaXYWHAOBBoxCoder
  - `v1`：None
  - `v2`：Constrained angle + Projection of dx and dy + Normalized with PI
  - `v3`：Constrained angle and length&width + Projection of dx and dy
- DeltaXYWHAHBBoxCoder
  - `v1`：None
  - `v2`：Constrained angle  + Normalized with PI
  - `v3`：Constrained angle and length&width  + Normalized with 2PI

**We believe that different coders are the key reason for the different baselines in different papers.** The good news is that all the above coders can be freely switched in R3Det. In addition, R3Det also provide 4 NMS ops and 3 IoU_Calculators for rotation detection as follows:

- `nms.type`
  - v1：`v1`
  - v2：`v2`
  - v3：`v3`
  - mmcv: `mmcv`
- `iou_calculator`
  - v1：`RBboxOverlaps2D_v1`
  - v2：`RBboxOverlaps2D_v2`
  - v3：`RBboxOverlaps2D_v3`

<!-- **Note: After switching the `angle_version` on the first line of the configuration file, please confirm whether the above mentioned `nms.type` and `iou_calculator` are consistent with the angle representation.** -->


### Performance


<summary>DOTA1.0 (Task1)</summary>


| Model | Backbone | Lr schd | MS | RR | Angle | box AP | Official | Download |
|:--------:|:--------:|:-------:|:--:|:------:|:--------:|:------:|:------:|:------:|
|RRetinaNet HBB |  R50-FPN |    1x   |  - |    -   | v1 |  65.19  |  [65.73](https://github.com/yangxue0827/RotationDetection)  | [Baidu:0518](https://pan.baidu.com/s/1ijkb0y_yAaicT-Z9_ljKeA)/[Google](https://drive.google.com/drive/folders/1CeD3QPTQRRSI7WKMwWE3EUWhzD2qN4e4?usp=sharing)
|RRetinaNet OBB|  R50-FPN |    1x   |  - |    -   | v3 |  68.20  |  [69.40](https://github.com/jbwang1997/OBBDetection/tree/master/configs/obb/retinanet_obb)  | [Baidu:0518](https://pan.baidu.com/s/1ijkb0y_yAaicT-Z9_ljKeA)/[Google](https://drive.google.com/drive/folders/1CeD3QPTQRRSI7WKMwWE3EUWhzD2qN4e4?usp=sharing) |
|RRetinaNet OBB |  R50-FPN |    1x   |  - |    -   | v2 |  68.64  |  [68.40](https://github.com/csuhan/s2anet)  | [Baidu:0518](https://pan.baidu.com/s/14o4sNxzfWQj1oGFjBzX8Kg)/[Google]()|
|R<sup>3</sup>Det|  R50-FPN |    1x   |  - |    -   | v1 |  70.41  |  [70.66](https://github.com/yangxue0827/RotationDetection)  | [Baidu:0518](https://pan.baidu.com/s/1ECNAzE3xaXXO7Pj2p_bLDw)/[Google]() |
|R<sup>3</sup>Det*|  R50-FPN |    1x   |  - |    -   | v1 |  70.86  |  -  | [Baidu:0518](https://pan.baidu.com/s/1kWg-bz2KjDcI-s_IWvUE6A)/[Google]() |

- `MS` means multiple scale image split.
- `RR` means random rotation.
