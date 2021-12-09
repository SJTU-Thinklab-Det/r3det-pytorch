#include <ATen/ATen.h>
#include <torch/extension.h>


at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof);


at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof);


inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
    return box_iou_rotated_cuda(
        boxes1.contiguous(),
	boxes2.contiguous(),
	iou_or_iof);
  }
    return box_iou_rotated_cpu(
        boxes1.contiguous(),
    boxes2.contiguous(),
    iou_or_iof);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("overlaps", box_iou_rotated, "calculate iou or iof of two group boxes");
}
