/**
 * ga_types.cuh - Core Types and Inline Operations for Cl(3,1) Geometric Algebra
 *
 * Metric signature: (+, +, +, -)
 * Basis ordering (bitmap): 1, e1, e2, e12, e3, e13, e23, e123,
 *                           e4, e14, e24, e124, e34, e134, e234, e1234
 *
 * Conformal split:
 *   n0  = 0.5 * (e4 - e3)   (origin null vector)
 *   nInf = e3 + e4           (infinity null vector)
 */

#ifndef GA_TYPES_CUH
#define GA_TYPES_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Basis Indices
// ============================================================================
enum Basis {
    SCALAR = 0,
    E1, E2, E12, E3, E13, E23, E123,
    E4, E14, E24, E124, E34, E134, E234, E1234
};

// ============================================================================
// Multivector Type (16 floats)
// ============================================================================
struct __align__(16) Multivector {
    float v[16];

    __host__ __device__ float& operator[](int i) { return v[i]; }
    __host__ __device__ const float& operator[](int i) const { return v[i]; }
};

// ============================================================================
// Rotor Type (even subalgebra: scalar + 6 bivectors + pseudoscalar)
// ============================================================================
struct __align__(16) Rotor {
    float s;        // scalar
    float e12, e13, e23, e14, e24, e34;  // bivectors
    float e1234;    // pseudoscalar

    __host__ __device__ Rotor()
        : s(0), e12(0), e13(0), e23(0), e14(0), e24(0), e34(0), e1234(0) {}

    __host__ __device__ Rotor(float s_, float e12_, float e13_, float e23_,
                               float e14_, float e24_, float e34_, float e1234_)
        : s(s_), e12(e12_), e13(e13_), e23(e23_),
          e14(e14_), e24(e24_), e34(e34_), e1234(e1234_) {}
};

// ============================================================================
// Vector Type (3D Euclidean)
// ============================================================================
struct __align__(16) Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// ============================================================================
// Low-level basis blade operations
// ============================================================================

__host__ __device__ inline int grade(int bitmap) {
    int g = 0;
    for (int i = 0; i < 4; ++i) g += (bitmap >> i) & 1;
    return g;
}

__host__ __device__ inline float basis_gp_sign(int a, int b, int* out_idx) {
    // Compute geometric product sign for basis blades a and b
    // Metric: g1=1, g2=1, g3=1, g4=-1
    int common = a & b;
    float metric_sign = 1.0f;
    if (common & 1) metric_sign *=  1.0f;
    if (common & 2) metric_sign *=  1.0f;
    if (common & 4) metric_sign *=  1.0f;
    if (common & 8) metric_sign *= -1.0f;

    // Count swaps: for each bit in b, count bits in a that are greater
    int swaps = 0;
    for (int bi = 0; bi < 4; ++bi) {
        if ((b >> bi) & 1) {
            for (int ai = bi + 1; ai < 4; ++ai) {
                if ((a >> ai) & 1) swaps++;
            }
        }
    }
    float swap_sign = (swaps & 1) ? -1.0f : 1.0f;
    *out_idx = a ^ b;
    return metric_sign * swap_sign;
}

__host__ __device__ inline float basis_wp_sign(int a, int b, int* out_idx) {
    if (a & b) {
        *out_idx = -1;
        return 0.0f;
    }
    int swaps = 0;
    for (int bi = 0; bi < 4; ++bi) {
        if ((b >> bi) & 1) {
            for (int ai = bi + 1; ai < 4; ++ai) {
                if ((a >> ai) & 1) swaps++;
            }
        }
    }
    float swap_sign = (swaps & 1) ? -1.0f : 1.0f;
    *out_idx = a ^ b;
    return swap_sign;
}

__host__ __device__ inline float basis_ip_sign(int a, int b, int* out_idx) {
    int ga = grade(a);
    int gb = grade(b);
    if (ga > gb) {
        *out_idx = -1;
        return 0.0f;
    }
    int idx;
    float sign = basis_gp_sign(a, b, &idx);
    int gidx = grade(idx);
    if (gidx == gb - ga) {
        *out_idx = idx;
        return sign;
    }
    *out_idx = -1;
    return 0.0f;
}

__host__ __device__ inline float reversion_sign(int bitmap) {
    int g = grade(bitmap);
    int swaps = g * (g - 1) / 2;
    return (swaps & 1) ? -1.0f : 1.0f;
}

__host__ __device__ inline float involution_sign(int bitmap) {
    return (grade(bitmap) & 1) ? -1.0f : 1.0f;
}

// ============================================================================
// Inline Device/Host Operations
// ============================================================================

__host__ __device__ inline Multivector mv_zero() {
    Multivector mv;
    #pragma unroll
    for (int i = 0; i < 16; ++i) mv.v[i] = 0.0f;
    return mv;
}

__host__ __device__ inline Multivector mv_scalar(float s) {
    Multivector mv = mv_zero();
    mv.v[SCALAR] = s;
    return mv;
}

__host__ __device__ inline Multivector mv_vector(float x, float y, float z, float w) {
    Multivector mv = mv_zero();
    mv.v[E1] = x; mv.v[E2] = y; mv.v[E3] = z; mv.v[E4] = w;
    return mv;
}

__host__ __device__ inline Multivector mv_from_rotor(const Rotor& r) {
    Multivector mv = mv_zero();
    mv.v[SCALAR] = r.s;
    mv.v[E12] = r.e12; mv.v[E13] = r.e13; mv.v[E23] = r.e23;
    mv.v[E14] = r.e14; mv.v[E24] = r.e24; mv.v[E34] = r.e34;
    mv.v[E1234] = r.e1234;
    return mv;
}

__host__ __device__ inline Rotor rotor_from_mv(const Multivector& mv) {
    return Rotor(mv.v[SCALAR], mv.v[E12], mv.v[E13], mv.v[E23],
                 mv.v[E14], mv.v[E24], mv.v[E34], mv.v[E1234]);
}

__host__ __device__ inline Multivector geometric_product(const Multivector& a,
                                                          const Multivector& b) {
    Multivector r = mv_zero();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (a.v[i] == 0.0f) continue;
        float ai = a.v[i];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (b.v[j] == 0.0f) continue;
            int idx;
            float sign = basis_gp_sign(i, j, &idx);
            r.v[idx] += ai * b.v[j] * sign;
        }
    }
    return r;
}

__host__ __device__ inline Multivector wedge_product(const Multivector& a,
                                                      const Multivector& b) {
    Multivector r = mv_zero();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (a.v[i] == 0.0f) continue;
        float ai = a.v[i];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (b.v[j] == 0.0f) continue;
            int idx;
            float sign = basis_wp_sign(i, j, &idx);
            if (idx < 0) continue;
            r.v[idx] += ai * b.v[j] * sign;
        }
    }
    return r;
}

__host__ __device__ inline Multivector inner_product(const Multivector& a,
                                                      const Multivector& b) {
    Multivector r = mv_zero();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (a.v[i] == 0.0f) continue;
        float ai = a.v[i];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (b.v[j] == 0.0f) continue;
            int idx;
            float sign = basis_ip_sign(i, j, &idx);
            if (idx < 0) continue;
            r.v[idx] += ai * b.v[j] * sign;
        }
    }
    return r;
}

__host__ __device__ inline Multivector mv_add(const Multivector& a,
                                               const Multivector& b) {
    Multivector r;
    #pragma unroll
    for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] + b.v[i];
    return r;
}

__host__ __device__ inline Multivector mv_sub(const Multivector& a,
                                               const Multivector& b) {
    Multivector r;
    #pragma unroll
    for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] - b.v[i];
    return r;
}

__host__ __device__ inline Multivector mv_scale(const Multivector& a, float s) {
    Multivector r;
    #pragma unroll
    for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] * s;
    return r;
}

__host__ __device__ inline Multivector mv_reverse(const Multivector& a) {
    Multivector r;
    #pragma unroll
    for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] * reversion_sign(i);
    return r;
}

__host__ __device__ inline Multivector mv_involute(const Multivector& a) {
    Multivector r;
    #pragma unroll
    for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] * involution_sign(i);
    return r;
}

__host__ __device__ inline float mv_norm_sq(const Multivector& a) {
    // Scalar part of a * reverse(a)
    Multivector ar = mv_reverse(a);
    Multivector p = geometric_product(a, ar);
    return p.v[SCALAR];
}

__host__ __device__ inline float mv_norm(const Multivector& a) {
    return sqrtf(fabsf(mv_norm_sq(a)));
}

__host__ __device__ inline Multivector mv_normalize(const Multivector& a) {
    float n = mv_norm(a);
    return (n > 1e-7f) ? mv_scale(a, 1.0f / n) : mv_zero();
}

// ============================================================================
// Conformal Embedding
// ============================================================================

/**
 * Embed a 3D Euclidean point into Cl(3,1) conformal space.
 *
 * This uses a 4D projective-conformal model where:
 *   n0  = 0.5*(e4 - e3)   (origin null vector)
 *   nInf = e3 + e4        (infinity null vector)
 *
 * The point is mapped to a null vector:
 *   P = x*e1 + y*e2 + z*n0 + 0.5*(x^2+y^2+z^2)*nInf
 *
 * Note: In Cl(3,1) with 16 components, this provides a projective
 * conformal embedding. The Euclidean distance between two points
 * is recovered via the inner product.
 */
__host__ __device__ inline Multivector conformal_embed(float x, float y, float z) {
    Multivector mv = mv_zero();
    float half_norm_sq = 0.5f * (x*x + y*y + z*z);
    // P = x*e1 + y*e2 + z*n0 + 0.5*r^2*nInf
    // n0  = 0.5*(e4 - e3)  =>  coeff on e3: -z/2, coeff on e4: z/2
    // nInf = e3 + e4       =>  coeff on e3: r^2/2, coeff on e4: r^2/2
    mv.v[E1] = x;
    mv.v[E2] = y;
    mv.v[E3] = -0.5f * z + half_norm_sq;   // from n0 and nInf
    mv.v[E4] =  0.5f * z + half_norm_sq;
    return mv;
}

__host__ __device__ inline Multivector conformal_embed(const Vec3& p) {
    return conformal_embed(p.x, p.y, p.z);
}

/**
 * Extract the 3D Euclidean point from a conformal null vector.
 * Returns (x, y, z) by projecting out the n0/nInf components.
 */
__host__ __device__ inline Vec3 conformal_extract(const Multivector& mv) {
    Vec3 p;
    p.x = mv.v[E1];
    p.y = mv.v[E2];
    // z = coefficient difference: e4 - e3 gives z
    p.z = mv.v[E4] - mv.v[E3];
    return p;
}

// ============================================================================
// Rotor Operations
// ============================================================================

/**
 * Construct a rotor from an axis-angle representation.
 * Axis must be a unit vector. Angle is in radians.
 *
 * R = cos(angle/2) - sin(angle/2) * B
 * where B is the unit bivector representing the rotation plane.
 */
__host__ __device__ inline Rotor rotor_from_axis_angle(const Vec3& axis, float angle) {
    float half = 0.5f * angle;
    float c = cosf(half);
    float s = sinf(half);
    Rotor r;
    r.s = c;
    // R = cos(θ/2) - sin(θ/2) * B
    // where B = I*n = axis_x*e23 + axis_y*e13 + axis_z*e12
    // for 3D rotations in the positive-definite subspace.
    // Using R v R̃ with B as above gives right-handed rotation.
    r.e23 = -axis.x * s;
    r.e13 =  axis.y * s;
    r.e12 = -axis.z * s;
    r.e14 = 0; r.e24 = 0; r.e34 = 0; r.e1234 = 0;
    return r;
}

__host__ __device__ inline float rotor_norm_sq(const Rotor& r) {
    return r.s*r.s + r.e12*r.e12 + r.e13*r.e13 + r.e23*r.e23
         + r.e14*r.e14 + r.e24*r.e24 + r.e34*r.e34 + r.e1234*r.e1234;
}

__host__ __device__ inline Rotor rotor_normalize(const Rotor& r) {
    float n = sqrtf(rotor_norm_sq(r));
    if (n < 1e-7f) return Rotor();
    float inv = 1.0f / n;
    return Rotor(r.s*inv, r.e12*inv, r.e13*inv, r.e23*inv,
                 r.e14*inv, r.e24*inv, r.e34*inv, r.e1234*inv);
}

__host__ __device__ inline Rotor rotor_reverse(const Rotor& r) {
    // Reverse: scalar and pseudoscalar unchanged, bivectors flip sign
    return Rotor(r.s, -r.e12, -r.e13, -r.e23,
                 -r.e14, -r.e24, -r.e34, r.e1234);
}

__host__ __device__ inline Rotor rotor_geometric(const Rotor& a, const Rotor& b) {
    // Even subalgebra product
    Multivector ma = mv_from_rotor(a);
    Multivector mb = mv_from_rotor(b);
    Multivector mc = geometric_product(ma, mb);
    return rotor_from_mv(mc);
}

/**
 * Spherical linear interpolation between two rotors.
 * t in [0, 1]
 */
__host__ __device__ inline Rotor rotor_slerp(const Rotor& a, const Rotor& b, float t) {
    // Convert to multivectors for dot product
    Multivector ma = mv_from_rotor(a);
    Multivector mb = mv_from_rotor(b);
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) dot += ma.v[i] * mb.v[i];

    Rotor b_ = b;
    if (dot < 0.0f) {
        dot = -dot;
        b_ = rotor_from_mv(mv_scale(mb, -1.0f));
    }

    if (dot > 0.9995f) {
        // Linear interpolation for very close rotors
        Rotor res;
        res.s     = a.s     + t * (b_.s     - a.s);
        res.e12   = a.e12   + t * (b_.e12   - a.e12);
        res.e13   = a.e13   + t * (b_.e13   - a.e13);
        res.e23   = a.e23   + t * (b_.e23   - a.e23);
        res.e14   = a.e14   + t * (b_.e14   - a.e14);
        res.e24   = a.e24   + t * (b_.e24   - a.e24);
        res.e34   = a.e34   + t * (b_.e34   - a.e34);
        res.e1234 = a.e1234 + t * (b_.e1234 - a.e1234);
        return rotor_normalize(res);
    }

    float theta0 = acosf(dot);
    float theta = theta0 * t;
    float s0 = cosf(theta) - dot * sinf(theta) / sinf(theta0);
    float s1 = sinf(theta) / sinf(theta0);

    Rotor res;
    res.s     = a.s     * s0 + b_.s     * s1;
    res.e12   = a.e12   * s0 + b_.e12   * s1;
    res.e13   = a.e13   * s0 + b_.e13   * s1;
    res.e23   = a.e23   * s0 + b_.e23   * s1;
    res.e14   = a.e14   * s0 + b_.e14   * s1;
    res.e24   = a.e24   * s0 + b_.e24   * s1;
    res.e34   = a.e34   * s0 + b_.e34   * s1;
    res.e1234 = a.e1234 * s0 + b_.e1234 * s1;
    return rotor_normalize(res);
}

/**
 * Apply a rotor to a multivector: R * a * reverse(R)
 */
__host__ __device__ inline Multivector rotor_apply(const Rotor& r, const Multivector& a) {
    Multivector mr = mv_from_rotor(r);
    Multivector mrr = mv_reverse(mr);
    return geometric_product(geometric_product(mr, a), mrr);
}

/**
 * Apply a rotor to a 3D vector (treated as a grade-1 multivector).
 */
__host__ __device__ inline Vec3 rotor_apply_vec3(const Rotor& r, const Vec3& v) {
    Multivector mv = mv_vector(v.x, v.y, v.z, 0.0f);
    Multivector out = rotor_apply(r, mv);
    return Vec3(out.v[E1], out.v[E2], out.v[E3]);
}

// ============================================================================
// Reflection and Projection
// ============================================================================

/**
 * Reflect a multivector in a hyperplane with normal vector n.
 * Reflection: -n * a * n^{-1}
 * For unit vector n: -n * a * n
 */
__host__ __device__ inline Multivector reflect(const Multivector& a,
                                                const Multivector& n) {
    Multivector nn = geometric_product(n, n);
    float nsq = nn.v[SCALAR];
    if (fabsf(nsq) < 1e-7f) return a;
    float inv_nsq = 1.0f / nsq;
    Multivector n_inv = mv_scale(n, inv_nsq);
    Multivector t = geometric_product(n, a);
    t = geometric_product(t, n_inv);
    return mv_scale(t, -1.0f);
}

/**
 * Project a multivector onto a blade B.
 * Projection: (a _| B) * B^{-1}
 */
__host__ __device__ inline Multivector project_onto(const Multivector& a,
                                                     const Multivector& B) {
    Multivector B_rev = mv_reverse(B);
    Multivector BB = geometric_product(B, B_rev);
    float Bsq = BB.v[SCALAR];
    if (fabsf(Bsq) < 1e-7f) return mv_zero();
    float inv_Bsq = 1.0f / Bsq;
    Multivector t = inner_product(a, B);
    t = geometric_product(t, B_rev);
    return mv_scale(t, inv_Bsq);
}

/**
 * Reject a multivector from a blade B.
 * Rejection: a - project(a onto B)
 */
__host__ __device__ inline Multivector reject_from(const Multivector& a,
                                                    const Multivector& B) {
    return mv_sub(a, project_onto(a, B));
}

#endif // GA_TYPES_CUH
