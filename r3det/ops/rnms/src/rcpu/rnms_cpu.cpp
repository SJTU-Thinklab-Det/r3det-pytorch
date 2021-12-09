// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CPU(x);      \
    CHECK_CONTIGUOUS(x)

template <typename scalar_t>
struct Point
{
    scalar_t x, y;
    Point() : x(0), y(0) {}
    Point(scalar_t x, scalar_t y) : x(x), y(y) {}
    scalar_t dot(const Point<scalar_t> &vec) const
    {
        return this->x * vec.x + this->y * vec.y;
    }
    scalar_t cross(const Point<scalar_t> &vec) const
    {
        return this->x * vec.y - vec.x * this->y;
    }
    Point<scalar_t> operator-(
        const Point<scalar_t> &vec) const
    {
        return Point(this->x - vec.x, this->y - vec.y);
    }
    Point<scalar_t> operator-=(
        const Point<scalar_t> &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }
    Point<scalar_t> operator+(
        const Point<scalar_t> &vec) const
    {
        return Point(this->x + vec.x, this->y + vec.y);
    }
    Point<scalar_t> operator+=(
        const Point<scalar_t> &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }
    bool operator<(
        const Point<scalar_t> &vec) const
    {
        if ((this->x == 0 && this->y == 0) && (vec.x != 0 || vec.y != 0))
            return true;
        return this->cross(vec) > 0;
    }
};

template <typename scalar_t>
Point<scalar_t> operator*(scalar_t a,
                          const Point<scalar_t> &p)
{
    return Point<scalar_t>(a * p.x, a * p.y);
}

template <typename scalar_t>
struct LinSeg
{
    Point<scalar_t> x1, x2;
    LinSeg() {}
    LinSeg(const Point<scalar_t> &x1, const Point<scalar_t> &x2)
        : x1(x1), x2(x2) {}
    int InterSectWith(const LinSeg<scalar_t> &linseg,
                      Point<scalar_t> *ps)
    {
        Point<scalar_t> a1 = this->x1, a2 = this->x2, b1 = linseg.x1,
                        b2 = linseg.x2;
        /*
         intersection point
         A=a2-a1, B=b2-b1, C=a1-b1
         [C.x] = [-A.x  B.x] * [s]
         [C.y]   [-A.y  B.y]   [t]
         */
        Point<scalar_t> A = a2 - a1, B = b2 - b1, C = a1 - b1;
        if (C.x == 0 && C.y == 0)
        {
            ps[0] = a1;
            return 1;
        }
        scalar_t D = -A.cross(B);
        if (D != 0)
        { // not parallel, may intersect.
            scalar_t s = C.cross(B) / D;
            scalar_t t = -A.cross(C) / D;
            if (0 <= s && s < 1 && 0 <= t &&
                t < 1)
            { // head vertex does not count.
                ps[0] = a1 + s * A;
                return 1;
            }
            else
            {
                return 0;
            }
        }
        else
        { // check colinearity: |A*C|=0
            if (A.cross(C) != 0)
            { // not colinear
                return 0;
            }
            else
            {
                int p_cnt = 0;
                // colinear overlap: only tail vertices count.
                scalar_t BdtC = B.dot(C);     // (b2-b1)*(a1-b1)
                scalar_t BdtB = B.dot(B);     // (b2-b1)*(b2-b1)
                scalar_t AdtnC = -A.dot(C);   // (a2-a1)*(b1-a1)
                scalar_t AdtA = A.dot(A);     // (a2-a1)*(a2-a1)
                if (BdtC >= 0 && BdtC < BdtB) // a1 between b2 and b1
                    ps[p_cnt++] = a1;
                if (AdtnC >= 0 && AdtnC < AdtA) // b1 between a2 and a1
                    ps[p_cnt++] = b1;
                return p_cnt;
            }
        }
    }
};

template <typename scalar_t>
void rbbox2points(const scalar_t *const rb,
                  Point<scalar_t> *vs)
{
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
int vertex_in_rbbox(Point<scalar_t> *v1,
                    Point<scalar_t> *v2,
                    Point<scalar_t> *ps)
{
    Point<scalar_t> center = (scalar_t)0.5 * (v2[0] + v2[2]);
    Point<scalar_t> w_vec = (scalar_t)0.5 * (v2[1] - v2[0]);
    Point<scalar_t> h_vec = (scalar_t)0.5 * (v2[2] - v2[1]);
    scalar_t h_vec_2 = h_vec.dot(h_vec);
    scalar_t w_vec_2 = w_vec.dot(w_vec);
    int p_cnt = 0;
    for (int i = 0; i < 4; i++)
    {
        Point<scalar_t> pr = v1[i] - center;
        if (std::abs(pr.dot(h_vec)) < h_vec_2 &&
            std::abs(pr.dot(w_vec)) < w_vec_2)
        {
            ps[p_cnt++] = v1[i];
        }
    }
    return p_cnt;
}

template <typename scalar_t>
int rbbox_border_intsec(Point<scalar_t> *v1,
                        Point<scalar_t> *v2,
                        Point<scalar_t> *ps)
{
    LinSeg<scalar_t> rb1[4] = {
        {v1[0], v1[1]}, {v1[1], v1[2]}, {v1[2], v1[3]}, {v1[3], v1[0]}};
    LinSeg<scalar_t> rb2[4] = {
        {v2[0], v2[1]}, {v2[1], v2[2]}, {v2[2], v2[3]}, {v2[3], v2[0]}};
    int p_cnt = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            p_cnt += rb1[i].InterSectWith(rb2[j], ps + p_cnt);
        }
    return p_cnt;
}

template <typename scalar_t>
scalar_t area(Point<scalar_t> *vs_dirty, int p_cnt_dirty)
{
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
torch::Tensor nmsr_cpu_kernel(const torch::Tensor &dets_tsr, const float threshold)
{
    if (dets_tsr.numel() == 0)
    {
        return torch::empty({0}, dets_tsr.options().dtype(torch::kLong).device(torch::kCPU));
    }

    auto score_tsr = dets_tsr.select(1, 5).contiguous();

    auto order_tsr = std::get<1>(score_tsr.sort(0, /* descending=*/true));

    auto ndets = dets_tsr.size(0);
    torch::Tensor suppressed_tsr = torch::zeros({ndets}, dets_tsr.options().dtype(torch::kByte).device(torch::kCPU));

    auto dets_p = dets_tsr.data_ptr<scalar_t>();
    auto order_p = order_tsr.data_ptr<int64_t>();
    auto suppressed_p = suppressed_tsr.data_ptr<uint8_t>();

    for (int64_t _i = 0; _i < ndets; _i++)
    {
        auto i = order_p[_i];
        if (suppressed_p[i] == 1)
            continue;

        for (int64_t _j = _i + 1; _j < ndets; _j++)
        {
            auto j = order_p[_j];
            if (suppressed_p[j] == 1)
                continue;

            auto rb1_p = dets_p + i * 6;
            auto rb2_p = dets_p + j * 6;
            Point<scalar_t> v1[4], v2[4], u[16];
            rbbox2points(rb1_p, v1);
            rbbox2points(rb2_p, v2);
            int p_cnt = 0;
            // add rbbox's vertices inside the other one
            p_cnt += vertex_in_rbbox(v1, v2, u + p_cnt);
            p_cnt += vertex_in_rbbox(v2, v1, u + p_cnt);
            // add rect border line segment intersection points
            p_cnt += rbbox_border_intsec(v1, v2, u + p_cnt);

            scalar_t iou = static_cast<scalar_t>(0);
            if (p_cnt >= 3)
            {
                scalar_t s1 = rb1_p[2] * rb1_p[3];
                scalar_t s2 = rb2_p[2] * rb2_p[3];
                scalar_t su = area(u, p_cnt);
                su = std::min(su, s1);
                su = std::min(su, s2);
                su = std::max(su, static_cast<scalar_t>(0));
                iou = su / (s1 + s2 - su);
            }
            if (iou >= threshold)
                suppressed_p[j] = 1;
        }
    }
    return torch::nonzero(suppressed_tsr == 0).squeeze(1);
}

torch::Tensor rnms_cpu(const torch::Tensor &dets_tsr, const float threshold)
{
    CHECK_INPUT(dets_tsr);
    torch::Tensor res_tsr;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        dets_tsr.scalar_type(), "nmsr", ([&] {
            res_tsr = nmsr_cpu_kernel<scalar_t>(dets_tsr, threshold);
        }));
    return res_tsr;
}
