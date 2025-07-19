/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_BUFFER_COMPARATOR_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_BUFFER_COMPARATOR_KERNEL_LIB_CU_H_

#include <sys/types.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/gpu/buffer_comparator_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/types.h"

namespace stream_executor::gpu {

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

template <typename T>
__global__ void xla_fp_comparison(T* buffer_a, T* buffer_b,
                                  float rel_error_threshold,
                                  uint64_t buffer_length, int* mismatch_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) {
    return;
  }

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

template <typename NativeT>
void RegisterBufferComparatorKernelParametrized(Platform::Id platform_id) {
  void* kernel_symbol = nullptr;
  constexpr xla::PrimitiveType p_type =
      xla::primitive_util::NativeToPrimitiveType<NativeT>();

  if constexpr (xla::primitive_util::IsIntegralType(p_type)) {
    kernel_symbol = absl::bit_cast<void*>(&xla_int_comparison<NativeT>);
  } else if constexpr (xla::primitive_util::IsFloatingPointType(p_type)) {
    kernel_symbol = absl::bit_cast<void*>(&xla_fp_comparison<NativeT>);
  } else {
    LOG(FATAL) << "Failed to register buffer comparator kernel for type "
               << xla::primitive_util::LowercasePrimitiveTypeName(p_type);
    return;
  }
  std::string kernel_name = absl::StrCat(
      xla::primitive_util::LowercasePrimitiveTypeName(p_type), "_comparison");

  stream_executor::KernelLoaderSpec spec =
      stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          kernel_symbol, kernel_name, 5);

  absl::Status result =
      stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .RegisterKernel<BufferComparatorKernel<NativeT>>(platform_id, spec);

  if (!result.ok()) {
    LOG(FATAL) << "Failed to register buffer comparator kernel for type "
               << xla::primitive_util::LowercasePrimitiveTypeName(p_type)
               << ": " << result;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BUFFER_COMPARATOR_KERNEL_LIB_CU_H_
