/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA

#include "tensorflow/compiler/xla/experiments/sm_bandwidth_benchmark/sm_bw_kernels.h"

#include "tensorflow/compiler/xla/experiments/sm_bandwidth_benchmark/sm_bw_utils.h"

namespace experiments {
namespace benchmark {
#define DFUNC __forceinline__ __device__
#define HDFUNC DFUNC __host__

template <typename ET, size_t S>
class Vec {
 public:
  using ElementType = ET;
  constexpr static size_t Size = S;

  template <typename... Ts>
  HDFUNC Vec(Ts... elements) : data_() {
    InsertElements(0, elements...);
  }

  HDFUNC ElementType& operator[](size_t idx) { return data_[idx]; }
  HDFUNC const ElementType& operator[](size_t idx) const { return data_[idx]; }

 private:
  template <typename T, typename... Ts>
  HDFUNC void InsertElements(size_t idx, T element, Ts... rest) {
    data_[idx] = element;
    InsertElements(idx + 1, rest...);
  }
  HDFUNC void InsertElements(size_t idx) {}

  ElementType data_[Size];
};

template <typename VectorType, typename T>
DFUNC void Store(VectorType vx, T* __restrict__ x, size_t id) {
  reinterpret_cast<VectorType* __restrict__>(x)[id] = vx;
}
template <>
DFUNC void Store(Vec<float, 4> vx, float* __restrict__ x, size_t id) {
  asm("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(x + 4 * id), "f"(vx[0]), "f"(vx[1]), "f"(vx[2]), "f"(vx[3]));
}

template <typename VectorType, typename T>
DFUNC void LoadNc(VectorType& vx, const T* __restrict__ x, size_t id) {
  vx = reinterpret_cast<const VectorType* __restrict__>(x)[id];
}

template <>
DFUNC void LoadNc(Vec<float, 4>& vx, const float* __restrict__ x, size_t id) {
  asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(vx[0]), "=f"(vx[1]), "=f"(vx[2]), "=f"(vx[3])
      : "l"(x + 4 * id));
}

template <int chunks>
__global__ void BenchmarkDeviceCopyKernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int64_t size) {
  const int64_t lines = size / (blockDim.x * chunks);
  const int64_t start_line = lines * blockIdx.x / gridDim.x;
  const int64_t end_line = lines * (blockIdx.x + 1) / gridDim.x;
  const int64_t start_offset =
      start_line * blockDim.x * chunks + 4 * threadIdx.x;
  const int64_t end_offset = end_line * blockDim.x * chunks;
  Vec<float, 4> buffer[chunks / 4];
  for (int64_t i = start_offset; i < end_offset; i += blockDim.x * chunks) {
#pragma unroll
    for (int j = 0; j < chunks; j += 4) {
      LoadNc(buffer[j / 4], in + i + blockDim.x * j, 0);
    }
#pragma unroll
    for (int j = 0; j < chunks; j += 4) {
      Store(buffer[j / 4], out + i + blockDim.x * j, 0);
    }
  }
}

template <int chunks>
void BenchmarkDeviceCopy(float* in, float* out, int64_t size, int num_blocks,
                         int num_threads) {
  BenchmarkDeviceCopyKernel<chunks><<<num_blocks, num_threads>>>(in, out, size);
  CHECK_CUDA(cudaGetLastError());
}

template void BenchmarkDeviceCopy<1>(float* in, float* out, int64_t size,
                                     int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 1>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 2>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 3>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 4>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 5>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 6>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 7>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 8>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 9>(float* in, float* out, int64_t size,
                                          int num_blocks, int num_threads);
template void BenchmarkDeviceCopy<1 << 10>(float* in, float* out, int64_t size,
                                           int num_blocks, int num_threads);
}  // namespace benchmark
}  // namespace experiments

#endif  // GOOGLE_CUDA
