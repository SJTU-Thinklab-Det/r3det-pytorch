#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return std::min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t* bottom_data,
    const int height, const int width,
    scalar_t y, scalar_t x) {
    // deal with cases that inverse elements are out of feature map boundary
    // if the feature map's size is [height, width], then its valid pixel
    // coordinates range is: x in [0, width-1] y in [0, height-1]
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        return 0;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (scalar_t)y_low;
    }
    else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (scalar_t)x_low;
    }
    else {
        x_high = x_low + 1;
    }

    scalar_t ly = y - y_low;
    scalar_t lx = x - x_low;
    scalar_t hy = 1. - ly;
    scalar_t hx = 1. - lx;
    // do bilinear interpolation
    scalar_t lt = bottom_data[y_low * width + x_low];
    scalar_t rt = bottom_data[y_low * width + x_high];
    scalar_t lb = bottom_data[y_high * width + x_low];
    scalar_t rb = bottom_data[y_high * width + x_high];
    scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

    return val;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
    scalar_t y, scalar_t x,
    scalar_t& w1, scalar_t& w2,
    scalar_t& w3, scalar_t& w4,
    int& x_low, int& x_high,
    int& y_low, int& y_high) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int)y;
    x_low = (int)x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (scalar_t)y_low;
    }
    else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (scalar_t)x_low;
    }
    else {
        x_high = x_low + 1;
    }

    scalar_t ly = y - y_low;
    scalar_t lx = x - x_low;
    scalar_t hy = 1. - ly;
    scalar_t hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename scalar_t>
__global__ void feature_refine_forward_kernel(
    const int nthreads,
    const int points,
    const scalar_t* bottom_data,
    const scalar_t* best_bboxes,  // of shape (n, h, w, 5)
    const scalar_t spatial_scale, const int channels, const int height,
    const int width, scalar_t* top_data) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, h, w) is an element in the aligned output
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height /
            channels;  // refers to the n^th image within a minibatch

        const scalar_t* bbox_offset =
            best_bboxes + ((n * height + h) * width + w) * 5;
        // for rbbox, there are 5 entries: [x_ctr, y_ctr, w, h, ang]
        scalar_t roi_y = bbox_offset[0] * spatial_scale;
        scalar_t roi_x = bbox_offset[1] * spatial_scale;

        scalar_t px[5] = { roi_x, 0, 0, 0, 0 };
        scalar_t py[5] = { roi_y, 0, 0, 0, 0 };

        if (points > 1) {
            scalar_t roi_w = bbox_offset[2] * spatial_scale;
            scalar_t roi_h = bbox_offset[3] * spatial_scale;
            scalar_t roi_a = bbox_offset[4];

            scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
            scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
            scalar_t wx = cosa * w_2, wy = sina * w_2;
            scalar_t hx = -sina * h_2, hy = cosa * h_2;

            px[1] = roi_x + wx + hx; py[1] = roi_y + wy + hy;
            px[2] = roi_x - wx + hx; py[2] = roi_y - wy + hy;
            px[3] = roi_x - wx - hx; py[3] = roi_y - wy - hy;
            px[4] = roi_x + wx - hx; py[4] = roi_y + wy - hy;
        }

        const scalar_t* offset_bottom_data =
            bottom_data + (n * channels + c) * height * width;

        scalar_t output_val = bottom_data[index];
        for (int i = 0; i < points; i++) {
            output_val += bilinear_interpolate<scalar_t>(offset_bottom_data, height,
                width, py[i], px[i]);
        }
        top_data[index] = output_val;
    }
}

template <typename scalar_t>
__global__ void feature_refine_backward_kernel(
    const int nthreads,
    const int points,
    const scalar_t* top_diff,
    const scalar_t* best_bboxes,  // of shape (n, h, w, 5)
    const scalar_t spatial_scale, const int channels, const int height,
    const int width, scalar_t* bottom_diff) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, h, w) is an element in the input diff
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height /
            channels;  // refers to the n^th image within a minibatch

        const scalar_t* bbox_offset =
            best_bboxes + ((n * height + h) * width + w) * 5;
        // for rbbox, there are 5 entries: [x_ctr, y_ctr, w, h, ang]
        scalar_t roi_y = bbox_offset[0] * spatial_scale;
        scalar_t roi_x = bbox_offset[1] * spatial_scale;

        scalar_t px[5] = { roi_x, 0, 0, 0, 0 };
        scalar_t py[5] = { roi_y, 0, 0, 0, 0 };

        if (points > 1) {
            scalar_t roi_w = bbox_offset[2] * spatial_scale;
            scalar_t roi_h = bbox_offset[3] * spatial_scale;
            scalar_t roi_a = bbox_offset[4];

            scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
            scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
            scalar_t wx = cosa * w_2, wy = sina * w_2;
            scalar_t hx = -sina * h_2, hy = cosa * h_2;

            px[1] = roi_x + wx + hx; py[1] = roi_y + wy + hy;
            px[2] = roi_x - wx + hx; py[2] = roi_y - wy + hy;
            px[3] = roi_x - wx - hx; py[3] = roi_y - wy - hy;
            px[4] = roi_x + wx - hx; py[4] = roi_y + wy - hy;
        }

        scalar_t* offset_bottom_diff =
            bottom_diff + (n * channels + c) * height * width;
        scalar_t value_top_diff = top_diff[index];

        atomicAdd(bottom_diff + index, value_top_diff);
        for (int i = 0; i < points; i++) {
            scalar_t w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            bilinear_interpolate_gradient<scalar_t>(height, width, py[i], px[i], w1,
                w2, w3, w4, x_low, x_high, y_low,
                y_high);
            scalar_t g1 = value_top_diff * w1;
            scalar_t g2 = value_top_diff * w2;
            scalar_t g3 = value_top_diff * w3;
            scalar_t g4 = value_top_diff * w4;
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
            }
        }
    }
}

int FRForwardLauncher(const torch::Tensor features,
    const torch::Tensor best_bboxes,  // of shape (n, h, w, 5)
    const float spatial_scale,
    const int points,
    torch::Tensor output) {
    const int output_size = features.numel();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.scalar_type(), "FRForwardLaucherFun", ([&] {
            const scalar_t* bottom_data = features.data_ptr<scalar_t>();
            const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
            scalar_t* top_data = output.data_ptr<scalar_t>();

            feature_refine_forward_kernel<scalar_t>
                << <GET_BLOCKS(output_size), THREADS_PER_BLOCK >> > (
                    output_size, points, bottom_data, bboxes_data, scalar_t(spatial_scale),
                    features.size(1), features.size(2), features.size(3), top_data);
            }));
    return 1;
}

int FRBackwardLauncher(const torch::Tensor top_grad,
    const torch::Tensor best_bboxes,
    const float spatial_scale,
    const int points,
    torch::Tensor bottom_grad) {
    const int output_size = top_grad.numel();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_grad.scalar_type(), "FRBackwardLaucherFun", ([&] {
            const scalar_t* top_diff = top_grad.data_ptr<scalar_t>();
            const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
            scalar_t* bottom_diff = bottom_grad.data_ptr<scalar_t>();

            feature_refine_backward_kernel<scalar_t>
                << <GET_BLOCKS(output_size), THREADS_PER_BLOCK >> > (
                    output_size, points, top_diff, bboxes_data, scalar_t(spatial_scale),
                    top_grad.size(1), top_grad.size(2), top_grad.size(3),
                    bottom_diff);
            }));
    return 1;
}
