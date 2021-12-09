// Modified from https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx, Soft-NMS is added
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

torch::Tensor rnms_cpu(const torch::Tensor &dets_tsr, const float threshold);


at::Tensor rnms_cuda(const at::Tensor& dets, const float threshold);


at::Tensor rnms(const at::Tensor& dets, const float threshold){
  if (dets.device().is_cuda()) {
    return rnms_cuda(dets, threshold);
  }
  return rnms_cpu(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rnms", &rnms, "non-maximum suppression for rotated bounding boxes");
}
