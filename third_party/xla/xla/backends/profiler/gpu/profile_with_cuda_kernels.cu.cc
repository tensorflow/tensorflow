/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/profiler/gpu/profile_with_cuda_kernels.h"

#include <cstddef>
#include <vector>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

__global__ void VecAdd(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void VecSub(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}

}  // namespace

std::vector<double> SimpleAddSubWithProfiler(int num_elements) {
  std::vector<double> vec_a;
  std::vector<double> vec_b;
  std::vector<double> vec_c;
  {
    // Allocates input/output vectors in host memory.
    vec_a.resize(num_elements, 10.);
    vec_b.resize(num_elements, 20.);
    vec_c.resize(num_elements, -1.);
  }

  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_c = nullptr;
  cudaStream_t stream = nullptr;
  const size_t num_bytes = num_elements * sizeof(double);

  {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // Allocates vectors in device memory.
    cudaMalloc((void**)&d_a, num_bytes);
    cudaMalloc((void**)&d_b, num_bytes);
    cudaMalloc((void**)&d_c, num_bytes);
  }

  {
    {
      // Copies vectors from host to device memory.
      cudaMemcpyAsync(d_a, vec_a.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(d_b, vec_b.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(d_c, vec_c.data(), num_bytes, cudaMemcpyHostToDevice,
                      stream);
    }

    {
      constexpr int kThreadsPerBlock = 256;
      const int blocks_per_grid =
          (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;

      // b1[i] = a[i] + b[i]
      VecAdd<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_a, d_b, d_b,
                                                               num_elements);
      // c1[i] = a[i] - b1[i] = a[i] - (a[i] + b[i]) = -b[i]
      VecSub<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_a, d_b, d_c,
                                                               num_elements);
      // c2[i] = c1[i] + b1[i]  ==> -b[i] + (a[i] + b[i]) = a[i]
      VecAdd<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_c, d_b, d_c,
                                                               num_elements);
      // c3[i] = c2[i] - a[i] = a[i] - a[i] = 0
      VecSub<<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(d_c, d_a, d_c,
                                                               num_elements);
    }

    {
      // Copies vectors from device to host memory.
      cudaMemcpyAsync(vec_c.data(), d_c, num_bytes, cudaMemcpyDeviceToHost,
                      stream);
    }
  }

  {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  return vec_c;
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
