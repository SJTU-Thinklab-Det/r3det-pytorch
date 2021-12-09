import mmcv
import numpy as np
import torch
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

pi = np.pi


@BBOX_CODERS.register_module()
class DeltaXYWHAOBBoxCoder(BaseBBoxCoder):
    """Delta XYWHA OBBox coder.

    this coder is used for rotated objects detection (for example on task1 of
     DOTA dataset). this coder encodes bbox (xc, yc, w, h, a) into delta
      (dx, dy, dw, dh, da) and decodes delta (dx, dy, dw, dh, da) back to
      original bbox (xc, yc, w, h, a).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='v1',
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.angle_range = angle_range

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        if self.angle_range == 'v1':
            return bbox2delta_v1(bboxes, gt_bboxes, self.means, self.stds)
        elif self.angle_range == 'v2':
            return bbox2delta_v2(bboxes, gt_bboxes, self.means, self.stds)
        elif self.angle_range == 'v3':
            return bbox2delta_v3(bboxes, gt_bboxes, self.means, self.stds)
        else:
            raise NotImplementedError

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 5) or (N, 5)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, num_classes * 5) or (B, N, 5) or
               (N, num_classes * 5) or (N, 5). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if self.angle_range == 'v1':
            return delta2bbox_v1(bboxes, pred_bboxes, self.means, self.stds,
                                 max_shape, wh_ratio_clip, self.add_ctr_clamp,
                                 self.ctr_clamp)
        elif self.angle_range == 'v2':
            return delta2bbox_v2(bboxes, pred_bboxes, self.means, self.stds,
                                 wh_ratio_clip)
        elif self.angle_range == 'v3':
            return delta2bbox_v3(bboxes, pred_bboxes, self.means, self.stds,
                                 wh_ratio_clip)
        else:
            raise NotImplementedError


@mmcv.jit(coderize=True)
def bbox2delta_v1(proposals,
                  gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t.

    gt.
    We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.
    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, pa = (proposals[:, i] for i in range(5))
    gx, gy, gw, gh, ga = (gt[:, i] for i in range(5))
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = ga - pa
    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


@mmcv.jit(coderize=True)
def delta2bbox_v1(rois,
                  deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  max_shape=None,
                  wh_ratio_clip=16 / 1000,
                  add_ctr_clamp=False,
                  ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.
    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.
    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 5).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'v1'.
    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    # Compute rotated angle of each roi
    pa = rois[:, 4].unsqueeze(1).expand_as(da)
    dx_width = pw * dx
    dy_height = ph * dy
    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Compute angle
    ga = pa + da
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)
    rbboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())
    return rbboxes


@mmcv.jit(coderize=True)
def bbox2delta_v2(proposals,
                  gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t.

    gt.
    We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.
    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)
    gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

    dx = (torch.cos(ptheta) * (gx - px) + torch.sin(ptheta) * (gy - py)) / pw
    dy = (-torch.sin(ptheta) * (gx - px) + torch.cos(ptheta) * (gy - py)) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dtheta = gtheta - ptheta
    dtheta = (dtheta + pi / 4) % pi - pi / 4
    dtheta /= pi
    deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


@mmcv.jit(coderize=True)
def delta2bbox_v2(proposals,
                  deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.
    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.
    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 5).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'v1'.
    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dtheta = denorm_deltas[:, 4::5] * pi
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)

    px = px.unsqueeze(1).expand_as(dx)
    py = py.unsqueeze(1).expand_as(dy)
    pw = pw.unsqueeze(1).expand_as(dw)
    ph = ph.unsqueeze(1).expand_as(dh)
    ptheta = ptheta.unsqueeze(1).expand_as(dtheta)

    gx = dx * pw * torch.cos(ptheta) - dy * ph * torch.sin(ptheta) + px
    gy = dx * pw * torch.sin(ptheta) + dy * ph * torch.cos(ptheta) + py
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gtheta = dtheta + ptheta
    gtheta = (gtheta + pi / 4) % pi - pi / 4

    bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view_as(deltas)
    return bboxes


@mmcv.jit(coderize=True)
def bbox2delta_v3(proposals,
                  gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t.

    gt.
    We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.
    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)
    gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

    dtheta1 = gtheta - ptheta
    dtheta1 = (dtheta1 + pi / 2) % pi - pi / 2
    dtheta2 = gtheta - ptheta + pi / 2
    dtheta2 = (dtheta2 + pi / 2) % pi - pi / 2
    abs_dtheta1 = torch.abs(dtheta1)
    abs_dtheta2 = torch.abs(dtheta2)

    gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
    gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
    dtheta = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
    dx = (torch.cos(-ptheta) * (gx - px) + torch.sin(-ptheta) * (gy - py)) / pw
    dy = (-torch.sin(-ptheta) * (gx - px) + torch.cos(-ptheta) *
          (gy - py)) / ph
    dw = torch.log(gw_regular / pw)
    dh = torch.log(gh_regular / ph)
    deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


@mmcv.jit(coderize=True)
def delta2bbox_v3(proposals,
                  deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.
    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.
    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 5).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'v1'.
    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dtheta = denorm_deltas[:, 4::5]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)

    px = px.unsqueeze(1).expand_as(dx)
    py = py.unsqueeze(1).expand_as(dy)
    pw = pw.unsqueeze(1).expand_as(dw)
    ph = ph.unsqueeze(1).expand_as(dh)
    ptheta = ptheta.unsqueeze(1).expand_as(dtheta)

    gx = dx * pw * torch.cos(-ptheta) - dy * ph * torch.sin(-ptheta) + px
    gy = dx * pw * torch.sin(-ptheta) + dy * ph * torch.cos(-ptheta) + py
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gtheta = dtheta + ptheta

    w_regular = torch.where(gw > gh, gw, gh)
    h_regular = torch.where(gw > gh, gh, gw)
    theta_regular = torch.where(gw > gh, gtheta, gtheta + pi / 2)
    theta_regular = (theta_regular + pi / 2) % pi - pi / 2
    bboxes = torch.stack([gx, gy, w_regular, h_regular, theta_regular],
                         dim=-1).view_as(deltas)
    return bboxes
