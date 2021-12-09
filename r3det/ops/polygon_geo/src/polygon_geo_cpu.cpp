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
    scalar_t dot(const Point<scalar_t>& vec) const
    {
        return this->x * vec.x + this->y * vec.y;
    }
    scalar_t cross(const Point<scalar_t>& vec) const
    {
        return this->x * vec.y - vec.x * this->y;
    }
    Point<scalar_t> operator-(
        const Point<scalar_t>& vec) const
    {
        return Point(this->x - vec.x, this->y - vec.y);
    }
    Point<scalar_t> operator-=(
        const Point<scalar_t>& vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }
    Point<scalar_t> operator+(
        const Point<scalar_t>& vec) const
    {
        return Point(this->x + vec.x, this->y + vec.y);
    }
    Point<scalar_t> operator+=(
        const Point<scalar_t>& vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }
    bool operator<(
        const Point<scalar_t>& vec) const
    {
        if ((this->x == 0 && this->y == 0) && (vec.x != 0 || vec.y != 0))
            return true;
        return this->cross(vec) > 0;
    }
};

template <typename scalar_t>
Point<scalar_t> operator*(scalar_t a,
    const Point<scalar_t>& p)
{
    return Point<scalar_t>(a * p.x, a * p.y);
}

template <typename scalar_t>
struct LinSeg
{
    Point<scalar_t> x1, x2;
    LinSeg() {}
    LinSeg(const Point<scalar_t>& x1, const Point<scalar_t>& x2)
        : x1(x1), x2(x2) {}
    int InterSectWith(const LinSeg<scalar_t>& linseg,
        Point<scalar_t>* ps)
    {
        Point<scalar_t> a1 = this->x1, a2 = this->x2, b1 = linseg.x1,
            b2 = linseg.x2;
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
void polygon2points(const scalar_t* const poly,
    Point<scalar_t>* vs)
{
    scalar_t x1 = poly[0], y1 = poly[1];
    scalar_t x2 = poly[2], y2 = poly[3];
    scalar_t x3 = poly[4], y3 = poly[5];
    scalar_t x4 = poly[6], y4 = poly[7];

    vs[0] = Point<scalar_t>(x1, y1);
    vs[1] = Point<scalar_t>(x2, y2);
    vs[2] = Point<scalar_t>(x3, y3);
    vs[3] = Point<scalar_t>(x4, y4);
    // insertion sort
    for (int i = 2; i < 4; i++)
    {
        Point<scalar_t> pt = vs[i];
        int j;
        for (j = i - 1; (pt - vs[0]) < (vs[j] - vs[0]); j--)
            vs[j + 1] = vs[j];
        vs[j + 1] = pt;
    }
}

template <typename scalar_t>
int vertex_in_polygon(Point<scalar_t>* v1,
    Point<scalar_t>* v2,
    Point<scalar_t>* ps)
{
    int p_cnt = 0;
    for (int i = 0; i < 4; i++)
    {
        bool inside_flag = true;
        for (int j = 0; j < 4; j++) {
            Point<scalar_t> pr = v1[i] - v2[j];
            Point<scalar_t> pb = v2[(j + 1) % 4] - v2[j];
            if (pr < pb) {
                inside_flag = false;
                break;
            }
        }
        if (inside_flag)
        {
            ps[p_cnt++] = v1[i];
        }
    }
    return p_cnt;
}

template <typename scalar_t>
int polygon_border_intsec(Point<scalar_t>* v1,
    Point<scalar_t>* v2,
    Point<scalar_t>* ps)
{
    LinSeg<scalar_t> rb1[4] = {
        {v1[0], v1[1]}, {v1[1], v1[2]}, {v1[2], v1[3]}, {v1[3], v1[0]} };
    LinSeg<scalar_t> rb2[4] = {
        {v2[0], v2[1]}, {v2[1], v2[2]}, {v2[2], v2[3]}, {v2[3], v2[0]} };
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
torch::Tensor polygon_iou_kernel(const torch::Tensor& a_tsr, const torch::Tensor& b_tsr)
{
    auto na = a_tsr.size(0);
    auto nb = b_tsr.size(0);
    torch::Tensor iou_tsr = torch::zeros({ na, nb }, a_tsr.options().device(torch::kCPU));

    auto a_p = a_tsr.data_ptr<scalar_t>();
    auto b_p = b_tsr.data_ptr<scalar_t>();
    auto iou_p = iou_tsr.data_ptr<scalar_t>();

    for (int64_t i = 0; i < na; i++)
    {
        for (int64_t j = 0; j < nb; j++)
        {
            auto poly1_p = a_p + i * 8;
            auto poly2_p = b_p + j * 8;
            Point<scalar_t> v1[4], v2[4], u[16];
            polygon2points(poly1_p, v1);
            polygon2points(poly2_p, v2);
            int p_cnt = 0;
            // add rbbox's vertices inside the other one
            p_cnt += vertex_in_polygon(v1, v2, u + p_cnt);
            p_cnt += vertex_in_polygon(v2, v1, u + p_cnt);
            // add rect border line segment intersection points
            p_cnt += polygon_border_intsec(v1, v2, u + p_cnt);

            if (p_cnt >= 3)
            {
                scalar_t s1 = area(v1, 4);
                scalar_t s2 = area(v2, 4);
                scalar_t su = area(u, p_cnt);
                su = std::min(su, s1);
                su = std::min(su, s2);
                su = std::max(su, static_cast<scalar_t>(0));
                iou_p[i * nb + j] = su / (s1 + s2 - su);
            }
            else {
                iou_p[i * nb + j] = static_cast<scalar_t>(0);
            }
        }
    }
    return iou_tsr;
}

torch::Tensor polygon_iou(const torch::Tensor& a_tsr, const torch::Tensor& b_tsr)
{
    CHECK_INPUT(a_tsr);
    CHECK_INPUT(b_tsr);
    torch::Tensor res_tsr;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a_tsr.scalar_type(), "polygon_iou", ([&] {
            res_tsr = polygon_iou_kernel<scalar_t>(a_tsr, b_tsr);
            }));
    return res_tsr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polygon_iou", &polygon_iou,
        "IOU calculation across 2 batches of 4-points polygons (CPU)");
}
