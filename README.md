# gpu-ga-kernel

**CUDA geometric algebra kernels — Cl(3,1) multivector operations, rotor compositions, and conformal embeddings running on GPU.**

GPU implementation of conformal geometric algebra operations in Cl(3,1). Multivector addition, geometric product, rotor composition, conformal point embedding, and reflection/rotation — all as CUDA kernels. Designed as the computational backend for geometric algebra applications needing GPU throughput.

## What This Gives You

- **Geometric product** — full Cl(3,1) multivector multiplication on GPU
- **Rotor operations** — compose, normalize, apply (sandwich product)
- **Conformal embeddings** — Euclidean 3D points → conformal space
- **Batch processing** — transform millions of points per kernel launch
- **Benchmarking** — throughput measurements for all operations

## Quick Start

```cuda
#include "ga_kernel.cuh"

// Batch embed points
launch_embed_points(d_points, d_multivectors, N, stream);

// Batch apply rotor
launch_apply_rotor(d_rotor, d_multivectors, d_results, N, stream);

// Batch geometric product
launch_geometric_product(d_a, d_b, d_result, N, stream);
```

## Build

```bash
nvcc -O3 -o test_bench test_bench.cu ga_kernel.cu -lcurand
./test_bench
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[ga-core](https://github.com/SuperInstance/ga-core)** — Rust geometric algebra library
- **gpu-ga-kernel** — CUDA geometric algebra kernels (this repo)

## License

MIT
