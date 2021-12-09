#include <torch/extension.h>

torch::Tensor mat_iou_iof_launcher(torch::Tensor rb1, torch::Tensor rb2, bool iof);
torch::Tensor vec_iou_iof_launcher(torch::Tensor rb1, torch::Tensor rb2, bool iof);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor mat_iou_iof(torch::Tensor rb1, torch::Tensor rb2, bool iof) {
    CHECK_INPUT(rb1);
    CHECK_INPUT(rb2);
    return mat_iou_iof_launcher(rb1, rb2, iof);
}

torch::Tensor vec_iou_iof(torch::Tensor rb1, torch::Tensor rb2, bool iof) {
    CHECK_INPUT(rb1);
    CHECK_INPUT(rb2);
    return vec_iou_iof_launcher(rb1, rb2, iof);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_iou_iof", &mat_iou_iof,
          "Union area calculation across 2 batches of rotated boxes (CUDA)");
    m.def("vec_iou_iof", &vec_iou_iof,
          "Union area calculation between 2 batches of rotated boxes (CUDA)");
}
