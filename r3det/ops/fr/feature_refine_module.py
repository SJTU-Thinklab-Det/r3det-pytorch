import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import feature_refine_cuda


class FeatureRefineFunction(Function):
    """Feature refine class."""

    @staticmethod
    def forward(ctx, features, best_rbboxes, spatial_scale, points=1):
        """Forward function."""
        ctx.spatial_scale = spatial_scale
        ctx.points = points
        ctx.save_for_backward(best_rbboxes)
        assert points in [1, 5]
        assert features.is_cuda
        output = torch.zeros_like(features)
        feature_refine_cuda.forward(features, best_rbboxes, spatial_scale,
                                    points, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward function."""
        best_rbboxes = ctx.saved_tensors[0]
        points = ctx.points
        spatial_scale = ctx.spatial_scale
        assert grad_output.is_cuda
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(grad_output)
            feature_refine_cuda.backward(grad_output.contiguous(),
                                         best_rbboxes, spatial_scale, points,
                                         grad_input)
        return grad_input, None, None, None


feature_refine = FeatureRefineFunction.apply


class FR(nn.Module):
    """FR module."""

    def __init__(self, spatial_scale, points=1):
        super(FR, self).__init__()
        self.spatial_scale = float(spatial_scale)
        self.points = points

    def forward(self, features, best_rbboxes):
        """Forward function."""
        return feature_refine(features, best_rbboxes, self.spatial_scale,
                              self.points)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(spatial_scale={self.spatial_scale},' \
                      f' points={self.points})'
        return format_str


class FeatureRefineModule(nn.Module):
    """Feature refine module."""

    def __init__(self,
                 in_channels,
                 featmap_strides,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FeatureRefineModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of feature refine module."""
        self.fr = nn.ModuleList(
            [FR(spatial_scale=1 / s) for s in self.featmap_strides])
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1)

    def init_weights(self):
        """Initialize weights of feature refine module."""
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def forward(self, x, best_rbboxes):
        """
        Args:
            x (list[Tensor]):
                feature maps of multiple scales
            best_rbboxes (list[list[Tensor]]):
                best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [
            torch.cat(best_rbbox) for best_rbbox in zip(*best_rbboxes)
        ]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(
                x, mlvl_rbboxes, self.fr):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = fr_scale(feat_scale, best_rbboxes_scale)
            out.append(x_scale + feat_refined_scale)
        return out
