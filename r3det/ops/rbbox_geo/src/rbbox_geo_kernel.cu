#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <vector>

#define MAX_BLOCKS 65000
#define MAX_THREADS_PER_BLOCK 768

void getdim(dim3 &grid, dim3 &block,
    const std::vector<int> &shape,
    int grid_d_mul = 65000, int block_d_mul = 1024) {
    int block_d[3] = { 1, 1, 1 };
    int grid_d[3] = { 1, 1, 1 };
    std::vector<std::pair<int, int>> shape_pair;
    for (int i = 0; i < shape.size(); i++)
        shape_pair.push_back(std::make_pair(shape[i], i));
    std::sort(shape_pair.begin(), shape_pair.end());
    for (int i = 0; i < shape.size(); i++) {
        int shape_d = shape_pair[i].first, ind = shape_pair[i].second;
        int bd = std::pow(block_d_mul, 1.0 / (shape.size() - i));
        bd = std::min(shape_d, bd);  // at most!
        block_d[ind] = bd;
        block_d_mul /= bd;
        int gd = std::pow(grid_d_mul, 1.0 / (shape.size() - i));
        gd = std::min((shape_d + bd - 1) / bd, gd);  // at most!
        grid_d[ind] = gd;
        grid_d_mul /= gd;
    }
    block.x = block_d[0], block.y = block_d[1], block.z = block_d[2];
    grid.x = grid_d[0], grid.y = grid_d[1], grid.z = grid_d[2];
}

#define CUDA_1D_KERNEL_LOOP(i, ni)                                 \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ni; \
         i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, j, ni, nj)                              \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ni;     \
         i += blockDim.x * gridDim.x)                                  \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < nj; \
             j += blockDim.y * gridDim.y)

template <typename scalar_t>
struct Point {
    scalar_t x, y;
    __device__ Point() : x(0), y(0) {}
    __device__ Point(scalar_t x, scalar_t y) : x(x), y(y) {}
    __device__ scalar_t dot(const Point<scalar_t> &vec) const {
        return this->x * vec.x + this->y * vec.y;
    }
    __device__ scalar_t cross(const Point<scalar_t> &vec) const {
        return this->x * vec.y - vec.x * this->y;
    }
    __device__ Point<scalar_t> operator-(
        const Point<scalar_t> &vec) const {
        return Point(this->x - vec.x, this->y - vec.y);
    }
    __device__ Point<scalar_t> operator-=(
        const Point<scalar_t> &vec) {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }
    __device__ Point<scalar_t> operator+(
        const Point<scalar_t> &vec) const {
        return Point(this->x + vec.x, this->y + vec.y);
    }
    __device__ Point<scalar_t> operator+=(
        const Point<scalar_t> &vec) {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }
    __device__ bool operator<(
        const Point<scalar_t> &vec) const {
        if ((this->x == 0 && this->y == 0) && (vec.x != 0 || vec.y != 0))
            return true;
        return this->cross(vec) > 0;
    }
};

template <typename scalar_t>
__device__ Point<scalar_t> operator*(scalar_t a,
    const Point<scalar_t> &p) {
    return Point<scalar_t>(a * p.x, a * p.y);
}

template <typename scalar_t>
struct LinSeg {
    Point<scalar_t> x1, x2;
    __device__ LinSeg() {}
    __device__ LinSeg(const Point<scalar_t> &x1, const Point<scalar_t> &x2)
        : x1(x1), x2(x2) {}
    __device__ int InterSectWith(const LinSeg<scalar_t> &linseg,
        Point<scalar_t> *ps) {
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
__device__ void rbbox2points(const scalar_t *const rb,
    Point<scalar_t> *vs) {
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
__device__ int vertex_in_rbbox(Point<scalar_t> *v1,
    Point<scalar_t> *v2,
    Point<scalar_t> *ps) {
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
__device__ int rbbox_border_intsec(Point<scalar_t> *v1,
    Point<scalar_t> *v2,
    Point<scalar_t> *ps) {
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
__global__ void mat_iou_iof_kernel(
    const scalar_t * const rb1,  // (n1, 5)
    const scalar_t * const rb2,  // (n2, 5)
    scalar_t * const res,  // (n1, n2)
    const int n1,
    const int n2,
    bool iof) {
    CUDA_2D_KERNEL_LOOP(i, j, n1, n2) {
        auto rb1_p = rb1 + 5 * i;
        auto rb2_p = rb2 + 5 * j;
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
            if (iof) {
                res[i * n2 + j] = su / s1;
            }
            else {
                res[i * n2 + j] = su / (s1 + s2 - su);
            }
        }
        else {
            res[i * n2 + j] = (scalar_t)0;
        }
    }
}

template <typename scalar_t>
__global__ void vec_iou_iof_kernel(
    const scalar_t * const rb1,  // (n, 5) or (1, 5)
    const scalar_t * const rb2,  // (n, 5) or (1, 5)
    scalar_t * const res,  // (n,)
    const int n1,
    const int n2,
    bool iof) {
    size_t n = max(n1, n2);
    CUDA_1D_KERNEL_LOOP(i, n) {
        auto rb1_p = rb1 + (i % n1)*5;
        auto rb2_p = rb2 + (i % n2)*5;
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
            if (iof) {
                res[i] = su / s1;
            }
            else {
                res[i] = su / (s1 + s2 - su);
            }
        }
        else {
            res[i] = (scalar_t)0;
        }
    }
}

torch::Tensor mat_iou_iof_launcher(torch::Tensor rb1, torch::Tensor rb2, bool iof) {
    int n1 = rb1.size(0);
    int n2 = rb2.size(0);
    torch::Tensor res =
        torch::empty({ n1, n2 }, torch::dtype(rb1.dtype()).device(torch::kCUDA));
    dim3 grid, block;
    getdim(grid, block, { n1, n2 }, MAX_BLOCKS, MAX_THREADS_PER_BLOCK);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rb1.scalar_type(), "mat_iou_iof_launcher_fun", ([&] {
            mat_iou_iof_kernel<scalar_t> <<<grid, block>>> (
                rb1.data_ptr<scalar_t>(),
                rb2.data_ptr<scalar_t>(),
                res.data_ptr<scalar_t>(),
                n1,
                n2,
                iof);
            }));
    return res;
}

torch::Tensor vec_iou_iof_launcher(torch::Tensor rb1, torch::Tensor rb2, bool iof) {
    int n1 = rb1.size(0);
    int n2 = rb2.size(0);
    int n = std::max(n1, n2);
    torch::Tensor res =
        torch::empty({ n }, torch::dtype(rb1.dtype()).device(torch::kCUDA));
    dim3 grid(1), block(1);
    getdim(grid, block, { n }, MAX_BLOCKS, MAX_THREADS_PER_BLOCK);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rb1.scalar_type(), "vec_iou_iof_launcher_fun", ([&] {
            vec_iou_iof_kernel<scalar_t> <<<grid, block>>> (
                rb1.data_ptr<scalar_t>(),
                rb2.data_ptr<scalar_t>(),
                res.data_ptr<scalar_t>(),
                n1,
                n2,
                iof);
            }));
    return res;
}
