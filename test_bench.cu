/**
 * test_bench.cu - Comprehensive Test Suite for Cl(3,1) GPU GA Library
 *
 * Tests: geometric product, wedge, inner product, conformal embedding,
 *        rotor operations (axis-angle, slerp), reflection, projection,
 *        and batch kernel operations.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "ga_kernel.cuh"

// ============================================================================
// Test Utilities
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _diff = fabsf((a) - (b)); \
    if (_diff > (tol)) { \
        fprintf(stderr, "[FAIL] %s: expected %.6f, got %.6f (diff=%.6f)\n", \
                msg, (float)(b), (float)(a), _diff); \
        g_tests_failed++; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

#define ASSERT_MV_NEAR(mv1, mv2, tol, msg) do { \
    for (int _i = 0; _i < 16; ++_i) { \
        float _diff = fabsf((mv1).v[_i] - (mv2).v[_i]); \
        if (_diff > (tol)) { \
            fprintf(stderr, "[FAIL] %s [comp %d]: expected %.6f, got %.6f (diff=%.6f)\n", \
                    msg, _i, (mv2).v[_i], (mv1).v[_i], _diff); \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } \
} while(0)

#define RUN_TEST(name) do { \
    printf("\n--- Running: %s ---\n", #name); \
    name(); \
    cudaDeviceSynchronize(); \
    CHECK_CUDA(cudaGetLastError()); \
} while(0)

// ============================================================================
// Reference Host Implementations (for verification)
// ============================================================================

__host__ Multivector h_geometric_product(const Multivector& a, const Multivector& b) {
    // Use device functions on host (they are __host__ __device__)
    return geometric_product(a, b);
}

// ============================================================================
// Individual Unit Tests
// ============================================================================

void test_basis_vector_squares() {
    // e1^2 = 1, e2^2 = 1, e3^2 = 1, e4^2 = -1
    Multivector e1 = mv_vector(1,0,0,0);
    Multivector e2 = mv_vector(0,1,0,0);
    Multivector e3 = mv_vector(0,0,1,0);
    Multivector e4 = mv_vector(0,0,0,1);

    Multivector r1 = h_geometric_product(e1, e1);
    Multivector r2 = h_geometric_product(e2, e2);
    Multivector r3 = h_geometric_product(e3, e3);
    Multivector r4 = h_geometric_product(e4, e4);

    ASSERT_NEAR(r1.v[SCALAR], 1.0f, 1e-5f, "e1^2");
    ASSERT_NEAR(r2.v[SCALAR], 1.0f, 1e-5f, "e2^2");
    ASSERT_NEAR(r3.v[SCALAR], 1.0f, 1e-5f, "e3^2");
    ASSERT_NEAR(r4.v[SCALAR], -1.0f, 1e-5f, "e4^2");
}

void test_anticommutation() {
    // e1*e2 = e12, e2*e1 = -e12
    Multivector e1 = mv_vector(1,0,0,0);
    Multivector e2 = mv_vector(0,1,0,0);

    Multivector r12 = h_geometric_product(e1, e2);
    Multivector r21 = h_geometric_product(e2, e1);

    ASSERT_NEAR(r12.v[E12], 1.0f, 1e-5f, "e1*e2 = e12");
    ASSERT_NEAR(r21.v[E12], -1.0f, 1e-5f, "e2*e1 = -e12");
}

void test_mixed_signature() {
    // e3*e4 = e34, e4*e3 = -e34
    Multivector e3 = mv_vector(0,0,1,0);
    Multivector e4 = mv_vector(0,0,0,1);

    Multivector r34 = h_geometric_product(e3, e4);
    Multivector r43 = h_geometric_product(e4, e3);

    ASSERT_NEAR(r34.v[E34], 1.0f, 1e-5f, "e3*e4 = e34");
    ASSERT_NEAR(r43.v[E34], -1.0f, 1e-5f, "e4*e3 = -e34");
}

void test_wedge_product() {
    // e1 ^ e2 = e12
    Multivector e1 = mv_vector(1,0,0,0);
    Multivector e2 = mv_vector(0,1,0,0);

    Multivector w = wedge_product(e1, e2);
    ASSERT_NEAR(w.v[E12], 1.0f, 1e-5f, "e1^e2 = e12");
    ASSERT_NEAR(w.v[SCALAR], 0.0f, 1e-5f, "e1^e2 scalar part");

    // e1 ^ e1 = 0
    Multivector w2 = wedge_product(e1, e1);
    ASSERT_NEAR(w2.v[SCALAR], 0.0f, 1e-5f, "e1^e1 = 0");
}

void test_inner_product() {
    // e1 _| e1 = 1
    Multivector e1 = mv_vector(1,0,0,0);
    Multivector r = inner_product(e1, e1);
    ASSERT_NEAR(r.v[SCALAR], 1.0f, 1e-5f, "e1_|e1 = 1");

    // e12 _| e123 = -e3  (left contraction sign)
    Multivector e12 = mv_zero();
    e12.v[E12] = 1.0f;
    Multivector e123 = mv_zero();
    e123.v[E123] = 1.0f;
    Multivector r2 = inner_product(e12, e123);
    ASSERT_NEAR(r2.v[E3], -1.0f, 1e-5f, "e12_|e123 = -e3");
}

void test_reversion() {
    // Reverse(e12) = -e12
    Multivector e12 = mv_zero();
    e12.v[E12] = 1.0f;
    Multivector rev = mv_reverse(e12);
    ASSERT_NEAR(rev.v[E12], -1.0f, 1e-5f, "reverse(e12)");

    // Reverse(e123) = -e123
    Multivector e123 = mv_zero();
    e123.v[E123] = 1.0f;
    Multivector rev2 = mv_reverse(e123);
    ASSERT_NEAR(rev2.v[E123], -1.0f, 1e-5f, "reverse(e123)");
}

void test_rotor_axis_angle() {
    // Rotate vector (1,0,0) by 90 degrees around z-axis -> (0,1,0)
    Vec3 axis(0,0,1);
    float angle = 3.14159265f / 2.0f;  // 90 degrees
    Rotor r = rotor_from_axis_angle(axis, angle);
    r = rotor_normalize(r);

    Vec3 v(1,0,0);
    Vec3 out = rotor_apply_vec3(r, v);

    ASSERT_NEAR(out.x, 0.0f, 1e-4f, "rotate(1,0,0) around z by 90 deg: x");
    ASSERT_NEAR(out.y, 1.0f, 1e-4f, "rotate(1,0,0) around z by 90 deg: y");
    ASSERT_NEAR(out.z, 0.0f, 1e-4f, "rotate(1,0,0) around z by 90 deg: z");
}

void test_rotor_identity() {
    // Zero angle rotation should be identity
    Vec3 axis(1,0,0);
    Rotor r = rotor_from_axis_angle(axis, 0.0f);
    Vec3 v(1,2,3);
    Vec3 out = rotor_apply_vec3(r, v);

    ASSERT_NEAR(out.x, v.x, 1e-5f, "identity rotor: x");
    ASSERT_NEAR(out.y, v.y, 1e-5f, "identity rotor: y");
    ASSERT_NEAR(out.z, v.z, 1e-5f, "identity rotor: z");
}

void test_rotor_slerp() {
    // Slerp from identity to 90-degree z-rotation at t=0.5 -> 45-degree rotation
    Rotor id(1,0,0,0,0,0,0,0);
    Vec3 axis(0,0,1);
    Rotor r90 = rotor_from_axis_angle(axis, 3.14159265f / 2.0f);
    r90 = rotor_normalize(r90);

    Rotor r45 = rotor_slerp(id, r90, 0.5f);

    Vec3 v(1,0,0);
    Vec3 out = rotor_apply_vec3(r45, v);

    float expected_x = cosf(3.14159265f / 4.0f);
    float expected_y = sinf(3.14159265f / 4.0f);
    ASSERT_NEAR(out.x, expected_x, 1e-4f, "slerp 45 deg: x");
    ASSERT_NEAR(out.y, expected_y, 1e-4f, "slerp 45 deg: y");
    ASSERT_NEAR(out.z, 0.0f, 1e-4f, "slerp 45 deg: z");
}

void test_conformal_embed_roundtrip() {
    // 2D conformal roundtrip (z is not preserved in Cl(3,1) 2D CGA)
    Vec3 p(1.0f, 2.0f, 0.0f);
    Multivector mv = conformal_embed(p);
    Vec3 q = conformal_extract(mv);

    ASSERT_NEAR(q.x, p.x, 1e-5f, "conformal roundtrip: x");
    ASSERT_NEAR(q.y, p.y, 1e-5f, "conformal roundtrip: y");
    ASSERT_NEAR(q.z, 0.0f, 1e-5f, "conformal roundtrip: z (2D model)");
}

void test_conformal_null_vector() {
    // A conformally embedded 2D point must be a null vector (P^2 = 0)
    Multivector mv = conformal_embed(1.0f, 2.0f, 0.0f);
    Multivector sq = h_geometric_product(mv, mv);
    ASSERT_NEAR(sq.v[SCALAR], 0.0f, 1e-4f, "conformal 2D point is null");
}

void test_conformal_inner_product_distance() {
    // For two conformal points, P1·P2 = -0.5 * |p1-p2|^2
    Multivector p1 = conformal_embed(0.0f, 0.0f, 0.0f);
    Multivector p2 = conformal_embed(1.0f, 0.0f, 0.0f);
    Multivector ip = inner_product(p1, p2);
    // |p1-p2|^2 = 1, so P1·P2 = -0.5
    ASSERT_NEAR(ip.v[SCALAR], -0.5f, 1e-4f, "conformal inner product origin-to-unit-x");
}

void test_reflection() {
    // Reflect (1,0,0) in the yz-plane (normal = e1) -> (-1,0,0)
    Multivector n = mv_vector(1,0,0,0);
    Multivector a = mv_vector(1,0,0,0);
    Multivector r = reflect(a, n);

    ASSERT_NEAR(r.v[E1], -1.0f, 1e-5f, "reflect in yz-plane: e1");
    ASSERT_NEAR(r.v[E2], 0.0f, 1e-5f, "reflect in yz-plane: e2");
    ASSERT_NEAR(r.v[E3], 0.0f, 1e-5f, "reflect in yz-plane: e3");
}

void test_projection() {
    // Project e1 onto e1 -> e1
    Multivector e1 = mv_vector(1,0,0,0);
    Multivector r = project_onto(e1, e1);
    ASSERT_NEAR(r.v[E1], 1.0f, 1e-5f, "project e1 onto e1");
    ASSERT_NEAR(r.v[E2], 0.0f, 1e-5f, "project e1 onto e1 (e2)");
}

void test_rotor_composition() {
    // Compose two 90-degree rotations around z -> 180-degree rotation
    Vec3 axis(0,0,1);
    Rotor r90 = rotor_from_axis_angle(axis, 3.14159265f / 2.0f);
    Rotor r180 = rotor_geometric(r90, r90);
    r180 = rotor_normalize(r180);

    Vec3 v(1,0,0);
    Vec3 out = rotor_apply_vec3(r180, v);

    ASSERT_NEAR(out.x, -1.0f, 1e-4f, "compose 90+90 = 180: x");
    ASSERT_NEAR(out.y, 0.0f, 1e-4f, "compose 90+90 = 180: y");
    ASSERT_NEAR(out.z, 0.0f, 1e-4f, "compose 90+90 = 180: z");
}

// ============================================================================
// Batch Kernel Tests (GPU)
// ============================================================================

void test_batch_geometric_product() {
    const int N = 1024;
    std::vector<Multivector> h_a(N), h_b(N), h_out(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = mv_vector((float)i, 1.0f, 0.0f, 0.0f);
        h_b[i] = mv_vector(1.0f, (float)i, 0.0f, 0.0f);
    }

    Multivector *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(Multivector)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(Multivector)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(Multivector)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(Multivector), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(Multivector), cudaMemcpyHostToDevice));

    launch_batch_geometric_product(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(Multivector), cudaMemcpyDeviceToHost));

    // Verify first element: (i*e1 + e2) * (e1 + i*e2)
    for (int i = 0; i < N; ++i) {
        Multivector expected = h_geometric_product(h_a[i], h_b[i]);
        ASSERT_MV_NEAR(h_out[i], expected, 1e-4f, ("batch_gp [" + std::to_string(i) + "]").c_str());
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

void test_batch_wedge_product() {
    const int N = 512;
    std::vector<Multivector> h_a(N), h_b(N), h_out(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = mv_vector(1.0f, 0.0f, 0.0f, 0.0f);
        h_b[i] = mv_vector(0.0f, 1.0f, 0.0f, 0.0f);
    }

    Multivector *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(Multivector)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(Multivector)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(Multivector)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(Multivector), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(Multivector), cudaMemcpyHostToDevice));

    launch_batch_wedge_product(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(Multivector), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(h_out[i].v[E12], 1.0f, 1e-5f, "batch_wedge");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

void test_batch_rotor_apply() {
    const int N = 1024;
    std::vector<Rotor> h_R(N);
    std::vector<Multivector> h_a(N), h_out(N);

    Vec3 axis(0,0,1);
    for (int i = 0; i < N; ++i) {
        float angle = (float)i * 0.01f;
        h_R[i] = rotor_normalize(rotor_from_axis_angle(axis, angle));
        h_a[i] = mv_vector(1.0f, 0.0f, 0.0f, 0.0f);
    }

    Rotor* d_R;
    Multivector *d_a, *d_out;
    CHECK_CUDA(cudaMalloc(&d_R, N * sizeof(Rotor)));
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(Multivector)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(Multivector)));

    CHECK_CUDA(cudaMemcpy(d_R, h_R.data(), N * sizeof(Rotor), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(Multivector), cudaMemcpyHostToDevice));

    launch_batch_rotor_apply(d_R, d_a, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(Multivector), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        Multivector expected = rotor_apply(h_R[i], h_a[i]);
        ASSERT_MV_NEAR(h_out[i], expected, 1e-4f, ("batch_rotor [" + std::to_string(i) + "]").c_str());
    }

    cudaFree(d_R); cudaFree(d_a); cudaFree(d_out);
}

void test_batch_conformal_embed_extract() {
    const int N = 1024;
    std::vector<float> h_x(N), h_y(N), h_z(N);
    std::vector<Multivector> h_mv(N);
    std::vector<float> h_out_x(N), h_out_y(N), h_out_z(N);

    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)(i % 100) * 0.1f;
        h_y[i] = (float)(i % 50) * 0.2f;
        h_z[i] = (float)(i % 25) * 0.4f;
    }

    float *d_x, *d_y, *d_z;
    Multivector* d_mv;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mv, N * sizeof(Multivector)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_z, h_z.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch_batch_conformal_embed(d_x, d_y, d_z, d_mv, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *d_ox, *d_oy, *d_oz;
    CHECK_CUDA(cudaMalloc(&d_ox, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_oy, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_oz, N * sizeof(float)));

    launch_batch_conformal_extract(d_mv, d_ox, d_oy, d_oz, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_x.data(), d_ox, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_y.data(), d_oy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_z.data(), d_oz, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(h_out_x[i], h_x[i], 1e-4f, ("batch_embed_extract x [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out_y[i], h_y[i], 1e-4f, ("batch_embed_extract y [" + std::to_string(i) + "]").c_str());
        // z is not preserved in 2D conformal model
        ASSERT_NEAR(h_out_z[i], 0.0f, 1e-4f, ("batch_embed_extract z [" + std::to_string(i) + "]").c_str());
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mv);
    cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_oz);
}

void test_batch_rotor_slerp() {
    const int N = 512;
    std::vector<Rotor> h_a(N), h_b(N), h_out(N);
    std::vector<float> h_t(N);

    Rotor id(1,0,0,0,0,0,0,0);
    Vec3 axis(0,0,1);
    Rotor r90 = rotor_normalize(rotor_from_axis_angle(axis, 3.14159265f / 2.0f));

    for (int i = 0; i < N; ++i) {
        h_a[i] = id;
        h_b[i] = r90;
        h_t[i] = (float)i / (float)(N - 1);
    }

    Rotor *d_a, *d_b, *d_out;
    float* d_t;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(Rotor)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(Rotor)));
    CHECK_CUDA(cudaMalloc(&d_t, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(Rotor)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(Rotor), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(Rotor), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_t, h_t.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch_batch_rotor_slerp(d_a, d_b, d_t, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(Rotor), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        Rotor expected = rotor_slerp(h_a[i], h_b[i], h_t[i]);
        ASSERT_NEAR(h_out[i].s, expected.s, 1e-4f, ("batch_slerp s [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out[i].e12, expected.e12, 1e-4f, ("batch_slerp e12 [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out[i].e13, expected.e13, 1e-4f, ("batch_slerp e13 [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out[i].e23, expected.e23, 1e-4f, ("batch_slerp e23 [" + std::to_string(i) + "]").c_str());
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_t); cudaFree(d_out);
}

void test_batch_axis_angle_to_rotor() {
    const int N = 256;
    std::vector<Vec3> h_axes(N);
    std::vector<float> h_angles(N);
    std::vector<Rotor> h_out(N);

    for (int i = 0; i < N; ++i) {
        h_axes[i] = Vec3(0, 0, 1);
        h_angles[i] = (float)i * 0.05f;
    }

    Vec3* d_axes;
    float* d_angles;
    Rotor* d_out;
    CHECK_CUDA(cudaMalloc(&d_axes, N * sizeof(Vec3)));
    CHECK_CUDA(cudaMalloc(&d_angles, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(Rotor)));

    CHECK_CUDA(cudaMemcpy(d_axes, h_axes.data(), N * sizeof(Vec3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_angles, h_angles.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch_batch_axis_angle_to_rotor(d_axes, d_angles, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(Rotor), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        Rotor expected = rotor_from_axis_angle(h_axes[i], h_angles[i]);
        ASSERT_NEAR(h_out[i].s, expected.s, 1e-4f, ("batch_aa s [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out[i].e12, expected.e12, 1e-4f, ("batch_aa e12 [" + std::to_string(i) + "]").c_str());
        ASSERT_NEAR(h_out[i].e23, expected.e23, 1e-4f, ("batch_aa e23 [" + std::to_string(i) + "]").c_str());
    }

    cudaFree(d_axes); cudaFree(d_angles); cudaFree(d_out);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  GPU-GA-KERNEL Test Bench\n");
    printf("  Cl(3,1) Conformal Geometric Algebra\n");
    printf("========================================\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Host-side unit tests
    RUN_TEST(test_basis_vector_squares);
    RUN_TEST(test_anticommutation);
    RUN_TEST(test_mixed_signature);
    RUN_TEST(test_wedge_product);
    RUN_TEST(test_inner_product);
    RUN_TEST(test_reversion);
    RUN_TEST(test_rotor_axis_angle);
    RUN_TEST(test_rotor_identity);
    RUN_TEST(test_rotor_slerp);
    RUN_TEST(test_conformal_embed_roundtrip);
    RUN_TEST(test_conformal_null_vector);
    RUN_TEST(test_conformal_inner_product_distance);
    RUN_TEST(test_reflection);
    RUN_TEST(test_projection);
    RUN_TEST(test_rotor_composition);

    // GPU batch kernel tests
    RUN_TEST(test_batch_geometric_product);
    RUN_TEST(test_batch_wedge_product);
    RUN_TEST(test_batch_rotor_apply);
    RUN_TEST(test_batch_conformal_embed_extract);
    RUN_TEST(test_batch_rotor_slerp);
    RUN_TEST(test_batch_axis_angle_to_rotor);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("========================================\n");

    return (g_tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
