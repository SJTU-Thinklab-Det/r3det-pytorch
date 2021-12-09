#include <torch/extension.h>

int FRForwardLauncher(
    const torch::Tensor features,
    const torch::Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    torch::Tensor output);

int FRBackwardLauncher(
    const torch::Tensor top_grad,
    const torch::Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    torch::Tensor bottom_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

int feature_refine_forward(
    const torch::Tensor features,
    const torch::Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    torch::Tensor output) {

    CHECK_INPUT(features);
    CHECK_INPUT(best_bboxes);
    CHECK_INPUT(output);

    return FRForwardLauncher(
        features,
        best_bboxes,
        spatial_scale,
        points,
        output
    );
}

int feature_refine_backward(
    const torch::Tensor top_grad,
    const torch::Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    torch::Tensor bottom_grad) {

    CHECK_INPUT(top_grad);
    CHECK_INPUT(best_bboxes);
    CHECK_INPUT(bottom_grad);

    return FRBackwardLauncher(
        top_grad,
        best_bboxes,
        spatial_scale,
        points,
        bottom_grad
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &feature_refine_forward, "Feature Refine forward (CUDA)");
    m.def("backward", &feature_refine_backward, "Feature Refine backward (CUDA)");
}
