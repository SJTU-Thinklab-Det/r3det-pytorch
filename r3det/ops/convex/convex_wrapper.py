from torch.autograd import Function

from . import convex_ext


class ConvexSortFunction(Function):
    """Convex sort class."""

    @staticmethod
    def forward(ctx, pts, masks, circular):
        """Forward function."""
        idx = convex_ext.convex_sort(pts, masks, circular)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return ()


convex_sort_func = ConvexSortFunction.apply


def convex_sort(pts, masks, circular=True):
    """convex sort function."""
    return convex_sort_func(pts, masks, circular)
