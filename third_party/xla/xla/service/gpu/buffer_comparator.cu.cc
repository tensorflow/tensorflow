/* Copyright 2018 The OpenXLA Authors.

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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using bfloat16 = __nv_bfloat16;
#define BF16_TO_F32 __bfloat162float

#elif TENSORFLOW_USE_ROCM
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

using bfloat16 = hip_bfloat16;
#define BF16_TO_F32 float

#endif

#include <cstdint>

namespace xla::gpu::buffer_comparator {

// Comparison kernel code: compare two buffers of
// fp8/bf16/fp16/fp32/fp64/int8_t/int32_t of length buffer_length where the
// relative error does not exceed the passed rel_error_threshold. Write the
// number of mismatches into out parameter mismatch_count.

// NaN's are considered equal, and for half's we clamp all numbers to largest
// and smallest numbers representable to avoid miscomparisons due to overflows.
namespace {

__device__ __inline__ float Canonicalize(float input) {
  // All fp16 infinities are treated as 65505 or -65505, in order to avoid
  // differences due to overflows.
  return isnan(input) ? input : max(-65505.0f, min(input, 65505.0f));
}

#if GOOGLE_CUDA
__global__ void xla_fp8_e4m3fn_comparison(__nv_fp8_storage_t* buffer_a,
                                          __nv_fp8_storage_t* buffer_b,
                                          float rel_error_threshold,
                                          uint64_t buffer_length,
                                          int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  // TODO(philipphack): Replace with direct conversion to float when this
  // functionality becomes available.
  float elem_a =
      __half2float(__nv_cvt_fp8_to_halfraw(buffer_a[idx], __NV_E4M3));
  float elem_b =
      __half2float(__nv_cvt_fp8_to_halfraw(buffer_b[idx], __NV_E4M3));
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

__global__ void xla_fp8_e5m2_comparison(__nv_fp8_storage_t* buffer_a,
                                        __nv_fp8_storage_t* buffer_b,
                                        float rel_error_threshold,
                                        uint64_t buffer_length,
                                        int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  // TODO(philipphack): Replace with direct conversion to float when this
  // functionality becomes available.
  float elem_a =
      __half2float(__nv_cvt_fp8_to_halfraw(buffer_a[idx], __NV_E5M2));
  float elem_b =
      __half2float(__nv_cvt_fp8_to_halfraw(buffer_b[idx], __NV_E5M2));
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}
#endif  // GOOGLE_CUDA

__global__ void xla_fp16_comparison(__half* buffer_a, __half* buffer_b,
                                    float rel_error_threshold,
                                    uint64_t buffer_length,
                                    int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float elem_a = __half2float(buffer_a[idx]);
  float elem_b = __half2float(buffer_b[idx]);
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

__global__ void xla_fp32_comparison(float* buffer_a, float* buffer_b,
                                    float rel_error_threshold,
                                    uint64_t buffer_length,
                                    int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float elem_a = buffer_a[idx];
  float elem_b = buffer_b[idx];
  if (isnan(elem_a) && isnan(elem_b)) return;
  if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
    return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);
  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

__global__ void xla_fp64_comparison(double* buffer_a, double* buffer_b,
                                    float rel_error_threshold,
                                    uint64_t buffer_length,
                                    int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;

  double elem_a = buffer_a[idx];
  double elem_b = buffer_b[idx];
  if (isnan(elem_a) && isnan(elem_b)) return;
  if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
    return;
  double rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);
  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

__global__ void xla_bf16_comparison(bfloat16* buffer_a, bfloat16* buffer_b,
                                    float rel_error_threshold,
                                    uint64_t buffer_length,
                                    int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float elem_a = BF16_TO_F32(buffer_a[idx]);
  float elem_b = BF16_TO_F32(buffer_b[idx]);
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

// TODO(b/191520348): The comparison below requires exact equality.
__global__ void xla_int8_comparison(int8_t* buffer_a, int8_t* buffer_b,
                                    float rel_error_threshold,
                                    uint64_t buffer_length,
                                    int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float a = buffer_a[idx];
  float b = buffer_b[idx];
  float rel_error = abs(a - b) / (max(abs(a), abs(b)) + 1);
  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

__global__ void xla_int32_comparison(int* buffer_a, int* buffer_b,
                                     float rel_error_threshold,
                                     uint64_t buffer_length,
                                     int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float elem_a = static_cast<float>(buffer_a[idx]);
  float elem_b = static_cast<float>(buffer_b[idx]);
  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);
  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

}  // namespace

#if GOOGLE_CUDA
void* fp8_e4m3fn_comparison() {
  return reinterpret_cast<void*>(&xla_fp8_e4m3fn_comparison);
}

void* fp8_e5m2_comparison() {
  return reinterpret_cast<void*>(&xla_fp8_e5m2_comparison);
}
#endif

void* fp16_comparison() {
  return reinterpret_cast<void*>(&xla_fp16_comparison);
}

void* bf16_comparison() {
  return reinterpret_cast<void*>(&xla_bf16_comparison);
}

void* fp32_comparison() {
  return reinterpret_cast<void*>(&xla_fp32_comparison);
}

void* fp64_comparison() {
  return reinterpret_cast<void*>(&xla_fp64_comparison);
}

void* int8_comparison() {
  return reinterpret_cast<void*>(&xla_int8_comparison);
}

void* int32_comparison() {
  return reinterpret_cast<void*>(&xla_int32_comparison);
}

}  // namespace xla::gpu::buffer_comparator
