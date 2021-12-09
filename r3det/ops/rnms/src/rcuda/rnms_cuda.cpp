// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")

at::Tensor nmsr_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor rnms_cuda(const at::Tensor& dets, const float threshold) {
    CHECK_CUDA(dets);
    if (dets.numel() == 0)
        return at::empty({ 0 }, dets.options().dtype(at::kLong).device(at::kCPU));
    return nmsr_cuda(dets, threshold);
}
