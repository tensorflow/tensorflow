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

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "third_party/cub/block/block_load.cuh"
#include "third_party/cub/block/block_scan.cuh"
#include "third_party/cub/block/block_store.cuh"
#include "third_party/cub/device/device_histogram.cuh"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_reduce.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/iterator/counting_input_iterator.cuh"
#include "third_party/cub/iterator/transform_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"
#include "third_party/cub/warp/warp_reduce.cuh"
#include "third_party/gpus/cuda/include/cusparse.h"

namespace gpuprim = ::cub;

// Required for sorting Eigen::half
namespace cub {
template <>
struct NumericTraits<Eigen::half>
    : BaseTraits<FLOATING_POINT, true, false, unsigned short, Eigen::half> {};

// Provide overload for CUB to assign to volatile Eigen::half.
template <>
__device__ __forceinline__ void ThreadStoreVolatilePtr<Eigen::half>(
    Eigen::half *ptr, Eigen::half val, Int2Type<true> /*is_primitive*/) {
  reinterpret_cast<volatile unsigned short &>(ptr->x) = val.x;
}

// Provide overload for CUB to load from volatile Eigen::half.
template <>
__device__ __forceinline__ Eigen::half ThreadLoadVolatilePointer<Eigen::half>(
    Eigen::half *ptr, Int2Type<true> /*is_primitive*/) {
  auto x = reinterpret_cast<const volatile unsigned short &>(ptr->x);
  return Eigen::half_impl::raw_uint16_to_half(x);
}

}  // namespace cub

#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hipcub/hipcub.hpp"
namespace gpuprim = ::hipcub;

// Required for sorting Eigen::half
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, unsigned short> {};
};  // namespace detail
};  // namespace rocprim
#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
