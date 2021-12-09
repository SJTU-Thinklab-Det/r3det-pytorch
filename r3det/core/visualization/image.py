import math

import cv2
import numpy as np
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val


# 旋转框可视化
def imshow_det_rbboxes(img,
                       bboxes,
                       labels,
                       class_names=None,
                       score_thr=0,
                       bbox_color='green',
                       text_color='green',
                       thickness=2,
                       font_scale=0.5,
                       show=True,
                       win_name='',
                       wait_time=0,
                       out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    # assert bboxes.shape[0] == labels.shape[0]
    img = imread(img)

    if bboxes.shape[1] == 6:

        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]

        bbox_color = color_val(bbox_color)
        text_color = color_val(text_color)

        for bbox, label in zip(bboxes, labels):
            xc, yc, w, h, ag, p = bbox.tolist()
            wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
            hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
            p1 = (xc - wx - hx, yc - wy - hy)
            p2 = (xc + wx - hx, yc + wy - hy)
            p3 = (xc + wx + hx, yc + wy + hy)
            p4 = (xc - wx + hx, yc - wy + hy)
            ps = np.int0(np.array([p1, p2, p3, p4]))
            cv2.drawContours(img, [ps], -1, bbox_color, thickness=thickness)
        if out_file is not None:
            imwrite(img, out_file)
    else:
        if out_file is not None:
            imwrite(img, out_file)
