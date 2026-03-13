/* Copyright 2026 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

#include "cub/device/device_scan.cuh"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cub_prefix_sum_kernel_cuda.h"
#include "xla/types.h"

namespace stream_executor::cuda {

template <typename KeyT>
cudaError_t CubPrefixSum(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_in, void* d_out, size_t num_items,
                         CUstream stream) {
  return cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_in),
      static_cast<KeyT*>(d_out), num_items, stream);
}

#define XLA_CUB_DEFINE_PREFIX_SUM(type)                                       \
  template cudaError_t CubPrefixSum<type>(void*, size_t&, const void*, void*, \
                                          size_t, CUstream);

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_DEFINE_PREFIX_SUM(__nv_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_DEFINE_PREFIX_SUM(__half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_DEFINE_PREFIX_SUM(float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_DEFINE_PREFIX_SUM(double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_DEFINE_PREFIX_SUM(int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_DEFINE_PREFIX_SUM(int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_DEFINE_PREFIX_SUM(int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_DEFINE_PREFIX_SUM(int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_DEFINE_PREFIX_SUM(uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_DEFINE_PREFIX_SUM(uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_DEFINE_PREFIX_SUM(uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_DEFINE_PREFIX_SUM(uint64_t)
#endif

}  // namespace stream_executor::cuda
