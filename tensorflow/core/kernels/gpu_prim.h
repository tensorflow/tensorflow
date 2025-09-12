/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

To in writing unless required by applicable law or agreed,
distributed on an, software distributed under the license is "AS IS"
BASIS, WITHOUT OF ANY KIND WARRANTIES OR CONDITIONS, either express
or implied. For the specific language governing permissions and
limitations under the license, the license you must see.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
#define TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_

#include "tensorflow/core/platform/bfloat16.h"

#if GOOGLE_CUDA
#include "cub/block/block_load.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/block/block_store.cuh"
#include "cub/device/device_histogram.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_segmented_radix_sort.cuh"
#include "cub/device/device_segmented_reduce.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/warp/warp_reduce.cuh"
#include "third_party/gpus/cuda/include/cusparse.h"

namespace gpuprim = ::cub;

// Required for sorting Eigen::half and bfloat16.
namespace cub {

__device__ __forceinline__ void ThreadStoreVolatilePtr(
    Eigen::half* ptr, Eigen::half val, Int2Type<true> /*is_primitive*/) {
  *reinterpret_cast<volatile uint16_t *>(ptr) =
      Eigen::numext::bit_cast<uint16_t>(val);
}

__device__ __forceinline__ Eigen::half ThreadLoadVolatilePointer(
    Eigen::half *ptr, Int2Type<true> /*is_primitive*/) {
  const uint16_t result = *reinterpret_cast<volatile const uint16_t *>(ptr);
  return Eigen::numext::bit_cast<Eigen::half>(result);
}

__device__ __forceinline__ void ThreadStoreVolatilePtr(
    Eigen::bfloat16* ptr, Eigen::bfloat16 val,
    Int2Type<true> /*is_primitive*/) {
  *reinterpret_cast<volatile uint16_t *>(ptr) =
      Eigen::numext::bit_cast<uint16_t>(val);
}

__device__ __forceinline__ Eigen::bfloat16 ThreadLoadVolatilePointer(
    Eigen::bfloat16 *ptr, Int2Type<true> /*is_primitive*/) {
  uint16_t result = *reinterpret_cast<volatile uint16_t *>(ptr);
  return Eigen::numext::bit_cast<Eigen::bfloat16>(result);
}

template <>
struct NumericTraits<Eigen::half>
    : BaseTraits</*_CATEGORY=*/FLOATING_POINT, /*_PRIMITIVE=*/true,
                 /*_NULL_TYPE=*/false, /*_UnsignedBits=*/uint16_t,
                 /*T=*/Eigen::half> {};
template <>
struct NumericTraits<tensorflow::bfloat16>
    : BaseTraits</*_CATEGORY=*/FLOATING_POINT, /*_PRIMITIVE=*/true,
                 /*_NULL_TYPE=*/false, /*_UnsignedBits=*/uint16_t,
                 /*T=*/tensorflow::bfloat16> {};
}  // namespace cub
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hipcub/hipcub.hpp"
#include "rocm/rocm_config.h"
namespace gpuprim = ::hipcub;

// Required for sorting Eigen::half and bfloat16.
namespace rocprim {
namespace detail {
#if (TF_ROCM_VERSION >= 50200)
template <>
struct float_bit_mask<Eigen::half> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7C00;
  static constexpr uint16_t mantissa = 0x03FF;
  using bit_type = uint16_t;
};

template <>
struct float_bit_mask<Eigen::bfloat16> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7F80;
  static constexpr uint16_t mantissa = 0x007F;
  using bit_type = uint16_t;
};
#endif
template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
template <>
struct radix_key_codec_base<tensorflow::bfloat16>
    : radix_key_codec_floating<tensorflow::bfloat16, uint16_t> {};
};  // namespace detail
};  // namespace rocprim

#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
