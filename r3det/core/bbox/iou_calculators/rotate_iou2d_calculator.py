from mmcv.ops import box_iou_rotated
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS

from r3det.ops import obb_overlaps, rbbox_iou


@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D_v1(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='v1'):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'v1'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps_v1(bboxes1.contiguous(), bboxes2.contiguous(),
                                 mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps_v1(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return rbbox_iou(bboxes1, bboxes2, is_aligned, (mode == 'iof'))


@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D_v2(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='v1'):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'v1'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps_v2(bboxes1.contiguous(), bboxes2.contiguous(),
                                 mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps_v2(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return box_iou_rotated(bboxes1, bboxes2, mode, is_aligned)


@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D_v3(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='v1'):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'v1'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps_v3(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps_v3(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return obb_overlaps(bboxes1, bboxes2, mode, is_aligned)
