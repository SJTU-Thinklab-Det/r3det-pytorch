import numpy as np
import torch

from . import nms_rotated_ext


def obb2hbb(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obboxes (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=1)
    return torch.cat([center - bias, center + bias], dim=1)


def obb_nms(dets, iou_thr, device_id=None):
    """Compute the NMS of oriented bboxes."""
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.numel() == 0:
        inds = dets_th.new_zeros(0, dtype=torch.int64)
    else:
        # same bug will happen when bboxes is too small
        too_small = dets_th[:, [2, 3]].min(1)[0] < 0.001
        if too_small.all():
            inds = dets_th.new_zeros(0, dtype=torch.int64)
        else:
            ori_inds = torch.arange(dets_th.size(0))
            ori_inds = ori_inds[~too_small]
            dets_th = dets_th[~too_small]

            bboxes, scores = dets_th[:, :5], dets_th[:, 5]
            inds = nms_rotated_ext.nms_rotated(bboxes, scores, iou_thr)
            inds = ori_inds[inds]

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def poly_nms(dets, iou_thr, device_id=None):
    """Compute the NMS of polygons."""
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.device == torch.device('cpu'):
        raise NotImplementedError
    inds = nms_rotated_ext.nms_poly(dets_th.float(), iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def obb_batched_nms(bboxes, scores, inds, nms_thr, class_agnostic=False):
    """Compute the NMS of oriented bboxes in batches."""
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        hbboxes = obb2hbb(bboxes) if bboxes.size(-1) == 5 else bboxes
        max_coordinate = hbboxes.max() - hbboxes.min()
        offsets = inds.to(bboxes) * (max_coordinate + 1)

        if bboxes.size(-1) == 5:
            bboxes_for_nms = bboxes.clone()
            bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
        else:
            bboxes_for_nms = bboxes + offsets[:, None]

    dets, keep = obb_nms(
        torch.cat([bboxes_for_nms, scores[:, None]], -1), nms_thr)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep
