/* Copyright 2022 The OpenXLA Authors.

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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <limits>
namespace stream_executor {
namespace gpu {

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride,
                                          int batches, float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride * 2;
  float* dst4 = dst + dst_stride * 3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
    if (b2) {
      dst2[i] = src[i];
    }
    if (b3) {
      dst3[i] = src[i];
    }
    if (b4) {
      dst4[i] = src[i];
    }
  }
}

void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size);
}

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
__global__ void __xla_MakeBatchPointers(char* base, int stride, int n,
                                        void** ptrs_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}

void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                            void** ptrs_out) {
  const int threads_per_block = 256;
  hipLaunchKernelGGL(
      __xla_MakeBatchPointers,
      dim3((n + threads_per_block - 1) / threads_per_block, 1, 1),
      dim3(threads_per_block, 1, 1), 0, (hipStream_t)stream, base, stride, n,
      ptrs_out);
}

__device__ float sigmoid(float x) {
  if (x > 0)
    return 1. / (1. + __expf(-x));
  else
    return __expf(x) / (__expf(x) + 1.);
}

template <typename T, int act_mode>
__global__ void launchInplaceBiasActivation_kernel(T* c_data,
                                                   const T* bias_data,
                                                   uint64_t m, uint64_t n,
                                                   int64_t ldc, float param) {
  uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= n || y >= m) return;
  float v = static_cast<float>(c_data[x + y * ldc]) +
            static_cast<float>(bias_data[x]);
  if (act_mode == 1)
    v = sigmoid(v);
  else if (act_mode == 2)
    v = v > 0.0f ? v : 0.0f;
  else if (act_mode == 3)
    v = v > 0.0f ? (v > 6.0f ? 6.0f : v) : 0.0f;
  else if (act_mode == 4)
    v = v > 0.0f ? (v > param ? param : v) : 0.0f;
  else if (act_mode == 5)
    v = tanh(v);
  else if (act_mode == 6)
    v = v > -param ? (v > param ? param : v) : -param;
  else if (act_mode == 7)
    v = v > 0.0f ? v : __expf(v) - 1;
  else if (act_mode == 8)
    v = v > 0.0f ? v : param * v;
  else if (act_mode == 9)
    v = 0.5 * v * (1 + erf(v / sqrt(2.0f)));
  c_data[x + y * ldc] = (T)v;
}

template <typename T>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data, int activation_mode,
                                 uint64_t m, uint64_t n, int64_t ldc,
                                 float param) {
  uint64_t bx = min(n, static_cast<uint64_t>(256));
  uint64_t by = min(m, static_cast<uint64_t>(256) / bx);
  uint64_t gx = (n + bx - 1) / bx;
  uint64_t gy = (m + by - 1) / by;
  auto kernel = launchInplaceBiasActivation_kernel<T, 0>;
  if (activation_mode == 1)
    kernel = launchInplaceBiasActivation_kernel<T, 1>;
  else if (activation_mode == 2)
    kernel = launchInplaceBiasActivation_kernel<T, 2>;
  else if (activation_mode == 3)
    kernel = launchInplaceBiasActivation_kernel<T, 3>;
  else if (activation_mode == 4)
    kernel = launchInplaceBiasActivation_kernel<T, 4>;
  else if (activation_mode == 5)
    kernel = launchInplaceBiasActivation_kernel<T, 5>;
  else if (activation_mode == 6)
    kernel = launchInplaceBiasActivation_kernel<T, 6>;
  else if (activation_mode == 7)
    kernel = launchInplaceBiasActivation_kernel<T, 7>;
  else if (activation_mode == 8)
    kernel = launchInplaceBiasActivation_kernel<T, 8>;
  else if (activation_mode == 9)
    kernel = launchInplaceBiasActivation_kernel<T, 9>;

  hipLaunchKernelGGL(kernel, dim3(gx, gy, 1), dim3(bx, by, 1), 0, stream,
                     static_cast<T*>(c_data), static_cast<const T*>(bias_data),
                     m, n, ldc, param);
}

template void launchInplaceBiasActivation<__half>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<hip_bfloat16>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<float>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<double>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

};  // namespace gpu
};  // namespace stream_executor
