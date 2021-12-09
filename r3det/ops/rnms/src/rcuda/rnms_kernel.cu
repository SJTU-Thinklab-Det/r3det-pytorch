// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cmath>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename scalar_t>
struct Point {
    scalar_t x, y;
    __device__ Point() : x(0), y(0) {}
    __device__ Point(scalar_t x, scalar_t y) : x(x), y(y) {}
    __device__ scalar_t dot(const Point<scalar_t>& vec) const {
        return this->x * vec.x + this->y * vec.y;
    }
    __device__ scalar_t cross(const Point<scalar_t>& vec) const {
        return this->x * vec.y - vec.x * this->y;
    }
    __device__ Point<scalar_t> operator-(
        const Point<scalar_t>& vec) const {
        return Point(this->x - vec.x, this->y - vec.y);
    }
    __device__ Point<scalar_t> operator-=(
        const Point<scalar_t>& vec) {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }
    __device__ Point<scalar_t> operator+(
        const Point<scalar_t>& vec) const {
        return Point(this->x + vec.x, this->y + vec.y);
    }
    __device__ Point<scalar_t> operator+=(
        const Point<scalar_t>& vec) {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }
    __device__ bool operator<(
        const Point<scalar_t>& vec) const {
        if ((this->x == 0 && this->y == 0) && (vec.x != 0 || vec.y != 0))
            return true;
        return this->cross(vec) > 0;
    }
};

template <typename scalar_t>
__device__ Point<scalar_t> operator*(scalar_t a,
    const Point<scalar_t>& p) {
    return Point<scalar_t>(a * p.x, a * p.y);
}

template <typename scalar_t>
struct LinSeg {
    Point<scalar_t> x1, x2;
    __device__ LinSeg() {}
    __device__ LinSeg(const Point<scalar_t>& x1, const Point<scalar_t>& x2)
        : x1(x1), x2(x2) {}
    __device__ int InterSectWith(const LinSeg<scalar_t>& linseg,
        Point<scalar_t>* ps) {
        Point<scalar_t> a1 = this->x1, a2 = this->x2, b1 = linseg.x1,
            b2 = linseg.x2;
        /*
         intersection point
         A=a2-a1, B=b2-b1, C=a1-b1
         [C.x] = [-A.x  B.x] * [s]
         [C.y]   [-A.y  B.y]   [t]
         */
        Point<scalar_t> A = a2 - a1, B = b2 - b1, C = a1 - b1;
        if (C.x == 0 && C.y == 0) {
            ps[0] = a1;
            return 1;
        }
        scalar_t D = -A.cross(B);
        if (D != 0) {  // not parallel, may intersect.
            scalar_t s = C.cross(B) / D;
            scalar_t t = -A.cross(C) / D;
            if (0 <= s && s < 1 && 0 <= t &&
                t < 1) {  // head vertex does not count.
                ps[0] = a1 + s * A;
                return 1;
            }
            else {
                return 0;
            }
        }
        else {                    // check colinearity: |A*C|=0
            if (A.cross(C) != 0) {  // not colinear
                return 0;
            }
            else {
                int p_cnt = 0;
                // colinear overlap: only tail vertices count.
                scalar_t BdtC = B.dot(C);      // (b2-b1)*(a1-b1)
                scalar_t BdtB = B.dot(B);      // (b2-b1)*(b2-b1)
                scalar_t AdtnC = -A.dot(C);    // (a2-a1)*(b1-a1)
                scalar_t AdtA = A.dot(A);      // (a2-a1)*(a2-a1)
                if (BdtC >= 0 && BdtC < BdtB)  // a1 between b2 and b1
                    ps[p_cnt++] = a1;
                if (AdtnC >= 0 && AdtnC < AdtA)  // b1 between a2 and a1
                    ps[p_cnt++] = b1;
                return p_cnt;
            }
        }
    }
};

template <typename scalar_t>
__device__ void rbbox2points(const scalar_t* const rb,
    Point<scalar_t>* vs) {
    scalar_t x = rb[0], y = rb[1], w_2 = rb[2] / 2, h_2 = rb[3] / 2, a = rb[4];
    scalar_t cosa = cosf(a), sina = sinf(a);
    scalar_t wx = cosa * w_2, wy = sina * w_2;
    scalar_t hx = -sina * h_2, hy = cosa * h_2;

    vs[0] = Point<scalar_t>(x + wx + hx, y + wy + hy);
    vs[1] = Point<scalar_t>(x - wx + hx, y - wy + hy);
    vs[2] = Point<scalar_t>(x - wx - hx, y - wy - hy);
    vs[3] = Point<scalar_t>(x + wx - hx, y + wy - hy);
}

template <typename scalar_t>
__device__ int vertex_in_rbbox(Point<scalar_t>* v1,
    Point<scalar_t>* v2,
    Point<scalar_t>* ps) {
    Point<scalar_t> center = (scalar_t)0.5 * (v2[0] + v2[2]);
    Point<scalar_t> w_vec = (scalar_t)0.5 * (v2[1] - v2[0]);
    Point<scalar_t> h_vec = (scalar_t)0.5 * (v2[2] - v2[1]);
    scalar_t h_vec_2 = h_vec.dot(h_vec);
    scalar_t w_vec_2 = w_vec.dot(w_vec);
    int p_cnt = 0;
    for (int i = 0; i < 4; i++) {
        Point<scalar_t> pr = v1[i] - center;
        if (std::abs(pr.dot(h_vec)) < h_vec_2 &&
            std::abs(pr.dot(w_vec)) < w_vec_2) {
            ps[p_cnt++] = v1[i];
        }
    }
    return p_cnt;
}

template <typename scalar_t>
__device__ int rbbox_border_intsec(Point<scalar_t>* v1,
    Point<scalar_t>* v2,
    Point<scalar_t>* ps) {
    LinSeg<scalar_t> rb1[4] = {
        {v1[0], v1[1]}, {v1[1], v1[2]}, {v1[2], v1[3]}, {v1[3], v1[0]} };
    LinSeg<scalar_t> rb2[4] = {
        {v2[0], v2[1]}, {v2[1], v2[2]}, {v2[2], v2[3]}, {v2[3], v2[0]} };
    int p_cnt = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            p_cnt += rb1[i].InterSectWith(rb2[j], ps + p_cnt);
        }
    return p_cnt;
}

template <typename scalar_t>
__device__ scalar_t area(Point<scalar_t> *vs_dirty, int p_cnt_dirty) {
    const scalar_t numthres = (scalar_t) 1e-2;
    Point<scalar_t> vs[16];
    vs[0] = {0, 0};
    int p_cnt = 1;
    // set vs[0] the reference point
    for (int i = 1; i < p_cnt_dirty; i++) {
        bool clean = true;
        vs_dirty[i] -= vs_dirty[0];
        for (int j = 0; j < p_cnt; j++) {
            Point<scalar_t> diff = vs_dirty[i] - vs[j];
            if (std::abs(diff.x) < numthres && std::abs(diff.y) < numthres) {
                clean = false;
                break;
            }
        }
        if (clean) {
            vs[p_cnt++] = vs_dirty[i];
        }
    }
    // sort
    for (int i = 1; i < p_cnt; i++) {
        vs[0] = vs[i];
        int j;
        for (j = i - 1; vs[0] < vs[j]; j--)
            vs[j + 1] = vs[j];
        vs[j + 1] = vs[0];
    }
    // calculate area
    scalar_t a = 0;
    vs[0] = {0, 0};
    for (int i = 1; i < p_cnt; i++)
        a += vs[i].cross(vs[(i + 1) % p_cnt]);
    return a / 2;
}

template <typename scalar_t>
__device__ scalar_t devIoU(
    const scalar_t* const rb1_p,
    const scalar_t* const rb2_p) {
    Point<scalar_t> v1[4], v2[4], u[16];
    rbbox2points(rb1_p, v1);
    rbbox2points(rb2_p, v2);
    int p_cnt = 0;
    // add rbbox's vertices inside the other one
    p_cnt += vertex_in_rbbox(v1, v2, u + p_cnt);
    p_cnt += vertex_in_rbbox(v2, v1, u + p_cnt);
    // add rect border line segment intersection points
    p_cnt += rbbox_border_intsec(v1, v2, u + p_cnt);
    if (p_cnt >= 3) {
        scalar_t s1 = rb1_p[2] * rb1_p[3];
        scalar_t s2 = rb2_p[2] * rb2_p[3];
        scalar_t su = area(u, p_cnt);
        su = min(su, s1);
        su = min(su, s2);
        su = max(su, (scalar_t)0);
        return su / (s1 + s2 - su);
    }
    else {
        return (scalar_t)0;
    }
}

__global__ void nmsr_kernel(const int n_boxes, const float nms_overlap_thresh,
    const float* dev_boxes, unsigned long long* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];

    auto block_boxes_p = block_boxes + threadIdx.x * 5;
    auto dev_boxes_p = dev_boxes + (threadsPerBlock * col_start + threadIdx.x) * 6;
    if (threadIdx.x < col_size) {
        block_boxes_p[0] = dev_boxes_p[0];
        block_boxes_p[1] = dev_boxes_p[1];
        block_boxes_p[2] = dev_boxes_p[2];
        block_boxes_p[3] = dev_boxes_p[3];
        block_boxes_p[4] = dev_boxes_p[4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float* cur_box = dev_boxes + cur_box_idx * 6;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

// boxes is a N x 6 tensor
at::Tensor nmsr_cuda(const at::Tensor boxes, float nms_overlap_thresh) {

    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(boxes.device());

    using scalar_t = float;
    AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
    auto scores = boxes.select(1, 5);
    auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
    auto boxes_sorted = boxes.index_select(0, order_t);

    int boxes_num = boxes.size(0);

    const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

    scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

    THCState* state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

    unsigned long long* mask_dev = NULL;
    //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
    //                      boxes_num * col_blocks * sizeof(unsigned long long)));

    mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

    dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
        THCCeilDiv(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    nmsr_kernel <<<blocks, threads>>> (boxes_num,
        nms_overlap_thresh,
        boxes_dev,
        mask_dev);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    THCudaCheck(cudaMemcpy(&mask_host[0],
        mask_dev,
        sizeof(unsigned long long) * boxes_num * col_blocks,
        cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep = at::empty({ boxes_num }, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t* keep_out = keep.data<int64_t>();

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
            unsigned long long* p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }

    THCudaFree(state, mask_dev);
    // TODO improve this part
    return std::get<0>(order_t.index({
                         keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                           order_t.device(), keep.scalar_type())
        }).sort(0, false));
}
