/**
 * ga_kernel.cu - CUDA Kernel Implementations for Cl(3,1) Batch Operations
 */

#include "ga_kernel.cuh"

// ============================================================================
// Kernel Implementations
// ============================================================================

__global__ void batch_geometric_product(const Multivector* a,
                                         const Multivector* b,
                                         Multivector* out,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = geometric_product(a[idx], b[idx]);
}

__global__ void batch_wedge_product(const Multivector* a,
                                     const Multivector* b,
                                     Multivector* out,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = wedge_product(a[idx], b[idx]);
}

__global__ void batch_inner_product(const Multivector* a,
                                     const Multivector* b,
                                     Multivector* out,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = inner_product(a[idx], b[idx]);
}

__global__ void batch_rotor_apply(const Rotor* R,
                                   const Multivector* a,
                                   Multivector* out,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = rotor_apply(R[idx], a[idx]);
}

__global__ void batch_conformal_embed(const float* x,
                                       const float* y,
                                       const float* z,
                                       Multivector* out,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = conformal_embed(x[idx], y[idx], z[idx]);
}

__global__ void batch_conformal_extract(const Multivector* mv,
                                         float* x,
                                         float* y,
                                         float* z,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Vec3 p = conformal_extract(mv[idx]);
    x[idx] = p.x;
    y[idx] = p.y;
    z[idx] = p.z;
}

__global__ void batch_reflect(const Multivector* nrm,
                               const Multivector* a,
                               Multivector* out,
                               int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    out[idx] = reflect(a[idx], nrm[idx]);
}

__global__ void batch_project(const Multivector* B,
                               const Multivector* a,
                               Multivector* out,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = project_onto(a[idx], B[idx]);
}

__global__ void batch_rotor_slerp(const Rotor* a,
                                   const Rotor* b,
                                   const float* t,
                                   Rotor* out,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = rotor_slerp(a[idx], b[idx], t[idx]);
}

__global__ void batch_axis_angle_to_rotor(const Vec3* axes,
                                           const float* angles,
                                           Rotor* out,
                                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = rotor_from_axis_angle(axes[idx], angles[idx]);
}

// ============================================================================
// Host Wrapper Implementations
// ============================================================================

static inline int blocks_needed(int n) {
    return (n + GA_BLOCK_SIZE - 1) / GA_BLOCK_SIZE;
}

void launch_batch_geometric_product(const Multivector* d_a,
                                     const Multivector* d_b,
                                     Multivector* d_out,
                                     int n) {
    batch_geometric_product<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_a, d_b, d_out, n);
}

void launch_batch_wedge_product(const Multivector* d_a,
                                 const Multivector* d_b,
                                 Multivector* d_out,
                                 int n) {
    batch_wedge_product<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_a, d_b, d_out, n);
}

void launch_batch_inner_product(const Multivector* d_a,
                                 const Multivector* d_b,
                                 Multivector* d_out,
                                 int n) {
    batch_inner_product<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_a, d_b, d_out, n);
}

void launch_batch_rotor_apply(const Rotor* d_R,
                               const Multivector* d_a,
                               Multivector* d_out,
                               int n) {
    batch_rotor_apply<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_R, d_a, d_out, n);
}

void launch_batch_conformal_embed(const float* d_x,
                                   const float* d_y,
                                   const float* d_z,
                                   Multivector* d_out,
                                   int n) {
    batch_conformal_embed<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_x, d_y, d_z, d_out, n);
}

void launch_batch_conformal_extract(const Multivector* d_mv,
                                     float* d_x,
                                     float* d_y,
                                     float* d_z,
                                     int n) {
    batch_conformal_extract<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_mv, d_x, d_y, d_z, n);
}

void launch_batch_reflect(const Multivector* d_norm,
                           const Multivector* d_a,
                           Multivector* d_out,
                           int n) {
    batch_reflect<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_norm, d_a, d_out, n);
}

void launch_batch_project(const Multivector* d_B,
                           const Multivector* d_a,
                           Multivector* d_out,
                           int n) {
    batch_project<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_B, d_a, d_out, n);
}

void launch_batch_rotor_slerp(const Rotor* d_a,
                               const Rotor* d_b,
                               const float* d_t,
                               Rotor* d_out,
                               int n) {
    batch_rotor_slerp<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_a, d_b, d_t, d_out, n);
}

void launch_batch_axis_angle_to_rotor(const Vec3* d_axes,
                                       const float* d_angles,
                                       Rotor* d_out,
                                       int n) {
    batch_axis_angle_to_rotor<<<blocks_needed(n), GA_BLOCK_SIZE>>>(d_axes, d_angles, d_out, n);
}
