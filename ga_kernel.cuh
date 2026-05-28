/**
 * ga_kernel.cuh - CUDA Kernel Declarations for Cl(3,1) Batch Operations
 */

#ifndef GA_KERNEL_CUH
#define GA_KERNEL_CUH

#include "ga_types.cuh"

// Batch size constants for kernel launch configuration
#define GA_BLOCK_SIZE 256

// ============================================================================
// Batch Kernels
// ============================================================================

/**
 * Batch geometric product: out[i] = a[i] * b[i]
 */
__global__ void batch_geometric_product(const Multivector* a,
                                         const Multivector* b,
                                         Multivector* out,
                                         int n);

/**
 * Batch wedge product: out[i] = a[i] ^ b[i]
 */
__global__ void batch_wedge_product(const Multivector* a,
                                     const Multivector* b,
                                     Multivector* out,
                                     int n);

/**
 * Batch inner product: out[i] = a[i] _| b[i]
 */
__global__ void batch_inner_product(const Multivector* a,
                                     const Multivector* b,
                                     Multivector* out,
                                     int n);

/**
 * Batch rotor application: out[i] = R * a[i] * reverse(R)
 */
__global__ void batch_rotor_apply(const Rotor* R,
                                   const Multivector* a,
                                   Multivector* out,
                                   int n);

/**
 * Batch conformal embedding: convert arrays of (x,y,z) to multivectors
 */
__global__ void batch_conformal_embed(const float* x,
                                       const float* y,
                                       const float* z,
                                       Multivector* out,
                                       int n);

/**
 * Batch conformal extraction: convert multivectors back to (x,y,z)
 */
__global__ void batch_conformal_extract(const Multivector* mv,
                                         float* x,
                                         float* y,
                                         float* z,
                                         int n);

/**
 * Batch reflection: out[i] = -n * a[i] * n^{-1}
 */
__global__ void batch_reflect(const Multivector* norm,
                               const Multivector* a,
                               Multivector* out,
                               int n);

/**
 * Batch projection: out[i] = project(a[i] onto B)
 */
__global__ void batch_project(const Multivector* B,
                               const Multivector* a,
                               Multivector* out,
                               int n);

/**
 * Batch rotor slerp: out[i] = slerp(a[i], b[i], t[i])
 */
__global__ void batch_rotor_slerp(const Rotor* a,
                                   const Rotor* b,
                                   const float* t,
                                   Rotor* out,
                                   int n);

/**
 * Batch axis-angle to rotor conversion
 */
__global__ void batch_axis_angle_to_rotor(const Vec3* axes,
                                           const float* angles,
                                           Rotor* out,
                                           int n);

// ============================================================================
// Host Wrapper Functions (convenience)
// ============================================================================

void launch_batch_geometric_product(const Multivector* d_a,
                                     const Multivector* d_b,
                                     Multivector* d_out,
                                     int n);

void launch_batch_wedge_product(const Multivector* d_a,
                                 const Multivector* d_b,
                                 Multivector* d_out,
                                 int n);

void launch_batch_inner_product(const Multivector* d_a,
                                 const Multivector* d_b,
                                 Multivector* d_out,
                                 int n);

void launch_batch_rotor_apply(const Rotor* d_R,
                               const Multivector* d_a,
                               Multivector* d_out,
                               int n);

void launch_batch_conformal_embed(const float* d_x,
                                   const float* d_y,
                                   const float* d_z,
                                   Multivector* d_out,
                                   int n);

void launch_batch_conformal_extract(const Multivector* d_mv,
                                     float* d_x,
                                     float* d_y,
                                     float* d_z,
                                     int n);

void launch_batch_reflect(const Multivector* d_norm,
                           const Multivector* d_a,
                           Multivector* d_out,
                           int n);

void launch_batch_project(const Multivector* d_B,
                           const Multivector* d_a,
                           Multivector* d_out,
                           int n);

void launch_batch_rotor_slerp(const Rotor* d_a,
                               const Rotor* d_b,
                               const float* d_t,
                               Rotor* d_out,
                               int n);

void launch_batch_axis_angle_to_rotor(const Vec3* d_axes,
                                       const float* d_angles,
                                       Rotor* d_out,
                                       int n);

#endif // GA_KERNEL_CUH
