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
template <int chunks>
__global__ void BenchmarkDeviceCopyKernel(const float* __restrict__ in,
                                          float* __restrict__ out, int64_t size,
                                          int num_blocks) {
  constexpr int num_thr = 64;
  int64_t lines = size / (num_thr * chunks);
  int64_t start_line = lines * blockIdx.x / num_blocks;
  int64_t end_line = lines * (blockIdx.x + 1) / num_blocks;
  int64_t start_offset = start_line * num_thr * chunks + threadIdx.x;
  int64_t end_offset = end_line * num_thr * chunks;
  float buffer[chunks];
  for (int64_t i = start_offset; i < end_offset; i += num_thr * chunks) {
#pragma unroll
    for (int j = 0; j < chunks; j++) {
      buffer[j] = in[i + num_thr * j];
    }
#pragma unroll
    for (int j = 0; j < chunks; j++) {
      out[i + num_thr * j] = buffer[j];
    }
  }
}

template <int chunks>
void BenchmarkDeviceCopy(float* in, float* out, int64_t size, int num_blocks) {
  BenchmarkDeviceCopyKernel<chunks>
      <<<num_blocks, 64>>>(in, out, size, num_blocks);
  CHECK_CUDA(cudaGetLastError());
}

template void BenchmarkDeviceCopy<1>(float* in, float* out, int64_t size,
                                     int num_blocks);
template void BenchmarkDeviceCopy<1 << 1>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 2>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 3>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 4>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 5>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 6>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 7>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 8>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 9>(float* in, float* out, int64_t size,
                                          int num_blocks);
template void BenchmarkDeviceCopy<1 << 10>(float* in, float* out, int64_t size,
                                           int num_blocks);
template void BenchmarkDeviceCopy<1 << 11>(float* in, float* out, int64_t size,
                                           int num_blocks);
template void BenchmarkDeviceCopy<1 << 12>(float* in, float* out, int64_t size,
                                           int num_blocks);
}  // namespace benchmark
}  // namespace experiments

#endif  // GOOGLE_CUDA
