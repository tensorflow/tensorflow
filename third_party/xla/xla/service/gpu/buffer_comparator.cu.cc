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

#include <cmath>
#include <cstdint>
#include <limits>

#include "xla/primitive_util.h"
#include "xla/types.h"

namespace xla::gpu::buffer_comparator {

// Comparison kernel code: compare two buffers of
// fp8/bf16/fp16/fp32/fp64/int8_t/int32_t of length buffer_length where the
// relative error does not exceed the passed rel_error_threshold. Write the
// number of mismatches into out parameter mismatch_count.

namespace {

// NaN's are considered equal, and for half's we clamp all numbers to largest
// and smallest numbers representable to avoid miscomparisons due to overflows.
template <typename T>
__device__ __inline__ auto Canonicalize(T elem) {
  // All fp16 infinities are treated as 65505 or -65505, in order to avoid
  // differences due to overflows.
  if (Eigen::numext::isinf(elem)) {
    return std::copysignf(Eigen::NumTraits<xla::half>::highest(), elem);
  }
  return static_cast<float>(elem);
}

template <>
__device__ __inline__ auto Canonicalize(float elem) {
  return elem;
}

template <>
__device__ __inline__ auto Canonicalize(double elem) {
  return elem;
}

#if TENSORFLOW_USE_ROCM && TF_ROCM_VERSION >= 60200

__global__ void xla_fp8_e4m3fnuz_comparison(__hip_fp8_storage_t* buffer_a,
                                            __hip_fp8_storage_t* buffer_b,
                                            float rel_error_threshold,
                                            uint64_t buffer_length,
                                            int* mismatch_count) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) 
// NOTE: according to amd_hip_fp8.h, GFX1200 and GFX1201 support ocp __hip_fp8_e4m3 
// but not __hip_fp8_e4m3_fnuz

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  __hip_fp8_e4m3_fnuz elem_a_fp8, elem_b_fp8;
  elem_a_fp8.__x = buffer_a[idx];
  elem_b_fp8.__x = buffer_b[idx];
  float elem_a = static_cast<float>(elem_a_fp8);
  float elem_b = static_cast<float>(elem_b_fp8);
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
#else
  // on unsupported architectures, this should not / cannot be used!
  atomicAdd(mismatch_count, 1); 
#endif
}

__global__ void xla_fp8_e5m2fnuz_comparison(__hip_fp8_storage_t* buffer_a,
                                            __hip_fp8_storage_t* buffer_b,
                                            float rel_error_threshold,
                                            uint64_t buffer_length,
                                            int* mismatch_count) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) 
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  __hip_fp8_e5m2_fnuz elem_a_fp8, elem_b_fp8;
  elem_a_fp8.__x = buffer_a[idx];
  elem_b_fp8.__x = buffer_b[idx];
  float elem_a = static_cast<float>(elem_a_fp8);
  float elem_b = static_cast<float>(elem_b_fp8);
  elem_a = Canonicalize(elem_a);
  elem_b = Canonicalize(elem_b);
  if (isnan(elem_a) && isnan(elem_b)) return;

  float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) + 1);

  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
#else
  // on unsupported architectures, this should not / cannot be used!
  atomicAdd(mismatch_count, 1); 
#endif
}

#endif  // TENSORFLOW_USE_ROCM && TF_ROCM_VERSION >= 60200

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
  if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b)) return;

  auto elem_a = Canonicalize(buffer_a[idx]);
  auto elem_b = Canonicalize(buffer_b[idx]);

  // NaN's are considered equal.
  if (Eigen::numext::isnan(elem_a) && Eigen::numext::isnan(elem_b)) {
    return;
  }

  // Two infinities are considered equal. Computing relative error would
  // otherwise result in NaN.
  if (elem_a == elem_b) {
    return;
  }

  float rel_error = Eigen::numext::abs(elem_a - elem_b) /
                    (Eigen::numext::maxi(Eigen::numext::abs(elem_a),
                                         Eigen::numext::abs(elem_b)) +
                     1);

  if (rel_error > rel_error_threshold || Eigen::numext::isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

// TODO(b/191520348): The comparison below requires exact equality.
template <typename T>
__global__ void xla_int_comparison(T* buffer_a, T* buffer_b,
                                   float rel_error_threshold,
                                   uint64_t buffer_length,
                                   int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  float elem_a;
  float elem_b;
  if constexpr (std::numeric_limits<T>::is_signed) {
    elem_a = static_cast<int64_t>(buffer_a[idx]);
    elem_b = static_cast<int64_t>(buffer_b[idx]);
  } else {
    elem_a = static_cast<uint64_t>(buffer_a[idx]);
    elem_b = static_cast<uint64_t>(buffer_b[idx]);
  }
  float rel_error =
      fabs(elem_a - elem_b) / (fmax(fabs(elem_a), fabs(elem_b)) + 1);
  if (rel_error > rel_error_threshold || isnan(rel_error))
    atomicAdd(mismatch_count, 1);
}

}  // namespace

void* comparison_fn(const xla::PrimitiveType type) {
  if (xla::primitive_util::IsFloatingPointType(type)) {
    return primitive_util::FloatingPointTypeSwitch<void*>(
        [](auto cst_type) {
          using native_type = primitive_util::NativeTypeOf<cst_type>;
          return reinterpret_cast<void*>(&xla_fp_comparison<native_type>);
        },
        type);
  }
  if (xla::primitive_util::IsIntegralType(type)) {
    return primitive_util::IntegralTypeSwitch<void*>(
        [](auto cst_type) {
          using native_type = primitive_util::NativeTypeOf<cst_type>;
          return reinterpret_cast<void*>(&xla_int_comparison<native_type>);
        },
        type);
  }
  return nullptr;
}

}  // namespace xla::gpu::buffer_comparator
