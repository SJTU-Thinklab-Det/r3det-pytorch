import numpy as np
import torch

from . import rnms_ext


def rnms(dets, nms_thr, device_id=None):
    """Compute NMS of oriented bboxes."""
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = rnms_ext.rnms(dets_th, nms_thr)
        else:
            inds = rnms_ext.rnms(dets_th, nms_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def batched_rnms(bboxes, scores, inds, nms_thr, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all bboxes,
            regardless of the predicted class

    Returns:
        tuple: kept bboxes and indice.
    """
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max()
        offsets = inds.to(bboxes) * (max_coordinate + 1)
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] += offsets[:, None]
    dets, keep = rnms(
        torch.cat([bboxes_for_nms, scores[:, None]], -1), nms_thr)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep
