import torch
from mmcv.ops import nms_rotated

from r3det.ops import batched_rnms, ml_nms_rotated, obb_batched_nms


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): NMS
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    nms_version = nms.get('type', 'v1')

    if nms_version == 'mmcv':

        labels = torch.arange(num_classes, dtype=torch.long)
        labels = labels.view(1, -1).expand_as(scores)

        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            # remove low scoring boxes
            valid_mask = scores > score_thr
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        else:
            # TensorRT NMS plugin has invalid output filled with -1
            # add dummy data to make detection output correct.
            bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 5)], dim=0)
            scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
            labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

        if bboxes.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, labels, inds
            else:
                return dets, labels

        dets, keep = nms_rotated(bboxes, scores, nms.iou_thr, labels.cuda())

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        if return_inds:
            return dets, labels[keep], keep
        else:
            return dets, labels[keep]

    else:
        valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero(as_tuple=False)[:, 1]

        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 6))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
            return bboxes, labels

        if nms_version == 'v1':
            dets, keep = batched_rnms(bboxes, scores, labels, nms.iou_thr)
        elif nms_version == 'v3':
            dets, keep = obb_batched_nms(bboxes, scores, labels, nms.iou_thr)
        elif nms_version == 'v2':
            keep = ml_nms_rotated(bboxes, scores, labels, nms.iou_thr)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            if keep.size(0) > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                scores = scores[inds]
                labels = labels[inds]
            return torch.cat([bboxes, scores[:, None]], 1), labels

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        return dets, labels[keep]
