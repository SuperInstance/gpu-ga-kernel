# gpu-ga-kernel

GPU-accelerated Cl(3,1) Conformal Geometric Algebra library.

## Overview

This is the CUDA/GPU counterpart to `ga-core`, implementing Cl(3,1) conformal geometric algebra with a 16-float multivector type. All core operations are implemented as CUDA kernels for high-throughput batch processing.

## Metric

Signature: **(+ + + &minus;)**

Basis ordering (bitmap): `1, e1, e2, e12, e3, e13, e23, e123, e4, e14, e24, e124, e34, e134, e234, e1234`

## Features

- **Multivector** type with 16 `float` components
- **Geometric Product** `a * b` (full Cl(3,1) multiplication table)
- **Wedge Product** `a ^ b`
- **Inner Product** (left contraction) `a _| b`
- **Rotor Operations**
  - Axis-angle construction
  - Composition via geometric product
  - Spherical linear interpolation (SLERP)
  - Application to multivectors and 3D vectors
- **Conformal Embedding**
  - Embed 2D Euclidean points as null vectors in Cl(3,1)
  - Extract Euclidean coordinates from conformal multivectors
  - (True 3D conformal geometry requires Cl(4,1) with 32 floats)
- **Reflection** in a hyperplane
- **Projection / Rejection** onto a blade
- **Batch Kernels** for all operations (GPU-parallel)
- **Host wrappers** for convenient kernel launching

## Building

Requires CUDA 11.5+ and `nvcc`.

```bash
make
```

To run tests:

```bash
make test
```

For debug build:

```bash
make DEBUG=1
```

## Files

| File | Description |
|------|-------------|
| `ga_types.cuh` | Core types, constants, and inline device/host operations |
| `ga_kernel.cuh` | CUDA kernel declarations and host wrapper prototypes |
| `ga_kernel.cu` | Kernel implementations and host wrapper functions |
| `test_bench.cu` | Comprehensive test suite |
| `Makefile` | Build system for CUDA 11.5+ |

## Quick Example

```cpp
#include "ga_kernel.cuh"

// Create a rotor: 90-degree rotation around Z-axis
Vec3 axis(0, 0, 1);
Rotor R = rotor_from_axis_angle(axis, 3.14159f / 2.0f);
R = rotor_normalize(R);

// Rotate a vector
Vec3 v(1, 0, 0);
Vec3 rotated = rotor_apply_vec3(R, v);
// rotated ≈ (0, 1, 0)

// Conformal embedding
Multivector p = conformal_embed(1.0f, 2.0f, 3.0f);
// p is a null vector: p*p ≈ 0
```

## Batch Processing

All operations have batch GPU kernels:

```cpp
// Allocate device arrays
Multivector *d_a, *d_b, *d_out;
cudaMalloc(&d_a, N * sizeof(Multivector));
cudaMalloc(&d_b, N * sizeof(Multivector));
cudaMalloc(&d_out, N * sizeof(Multivector));

// ... copy data to device ...

launch_batch_geometric_product(d_a, d_b, d_out, N);
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
