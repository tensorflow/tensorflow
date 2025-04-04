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

#include <cmath>
#include <cstdint>

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

__device__ float sigmoid(float x) {
  if (x > 0)
    return 1. / (1. + __expf(-x));
  else
    return __expf(x) / (__expf(x) + 1.);
}

template <typename T, typename Tbias, int act_mode>
__global__ void launchInplaceBiasActivation_kernel(
    T* c_data, const Tbias* bias_data, const T* side_input_data,
    float side_input_scale, uint64_t m, uint64_t n, int64_t ldc, float param,
    int transpose) {
  uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
  uint64_t z = blockIdx.z;
  if (x >= n || y >= m) return;
  float v;
  uint64_t addr = x + y * ldc + z * m * n;
  if (!transpose)
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[x]);
  else
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[y]);
  if (side_input_data != 0)
    v += static_cast<float>(side_input_data[addr]) * side_input_scale;
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
  c_data[addr] = (T)v;
}

template <typename T, typename Tbias>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data,
                                 const void* side_input_data,
                                 float side_input_scale, int activation_mode,
                                 uint64_t batch, uint64_t m, uint64_t n,
                                 int64_t ldc, float param) {
  uint64_t bx = min(n, static_cast<uint64_t>(256));
  uint64_t by = min(m, static_cast<uint64_t>(256) / bx);
  uint64_t gx = (n + bx - 1) / bx;
  uint64_t gy = (m + by - 1) / by;
  int transpose = (activation_mode >= 10);
  activation_mode %= 10;
  auto kernel = launchInplaceBiasActivation_kernel<T, Tbias, 0>;
  if (activation_mode == 1)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 1>;
  else if (activation_mode == 2)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 2>;
  else if (activation_mode == 3)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 3>;
  else if (activation_mode == 4)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 4>;
  else if (activation_mode == 5)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 5>;
  else if (activation_mode == 6)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 6>;
  else if (activation_mode == 7)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 7>;
  else if (activation_mode == 8)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 8>;
  else if (activation_mode == 9)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 9>;

  hipLaunchKernelGGL(kernel, dim3(gx, gy, batch), dim3(bx, by, 1), 0, stream,
                     static_cast<T*>(c_data),
                     static_cast<const Tbias*>(bias_data),
                     static_cast<const T*>(side_input_data), side_input_scale,
                     m, n, ldc, param, transpose);
}

#define INSTANTIATE_BIAS_ACTIVATION(X, Y)                          \
  template void launchInplaceBiasActivation<X, Y>(                 \
      hipStream_t stream, void* c_data, const void* bias_data,     \
      const void* side_input_data, float side_input_scale,         \
      int activation_mode, uint64_t batch, uint64_t m, uint64_t n, \
      int64_t ldc, float param);

INSTANTIATE_BIAS_ACTIVATION(__half, __half)
INSTANTIATE_BIAS_ACTIVATION(__half, float)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, hip_bfloat16)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, float)
INSTANTIATE_BIAS_ACTIVATION(float, float)
INSTANTIATE_BIAS_ACTIVATION(double, double)

};  // namespace gpu
};  // namespace stream_executor
