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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_

#include <cstddef>
#include <cstdint>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace stream_executor::cuda {

template <typename KeyT>
cudaError_t CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                        const void* d_keys_in, void* d_keys_out,
                        size_t num_items, bool descending, size_t batch_size,
                        CUstream stream);

template <typename KeyT, typename ValT>
cudaError_t CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         const void* d_values_in, void* d_values_out,
                         size_t num_items, bool descending, size_t batch_size,
                         CUstream stream);

// Helper macros for extern template declarations.
#define XLA_CUB_EXTERN_SORT_KEYS(type)           \
  extern template cudaError_t CubSortKeys<type>( \
      void*, size_t&, const void*, void*, size_t, bool, size_t, CUstream)

#define XLA_CUB_EXTERN_SORT_PAIRS(type1, type2)                             \
  extern template cudaError_t CubSortPairs<type1, type2>(                   \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, CUstream)

// Floating point types.
XLA_CUB_EXTERN_SORT_KEYS(__nv_bfloat16);
XLA_CUB_EXTERN_SORT_KEYS(__half);
XLA_CUB_EXTERN_SORT_KEYS(float);
XLA_CUB_EXTERN_SORT_KEYS(double);

// Signed integer types.
XLA_CUB_EXTERN_SORT_KEYS(int8_t);
XLA_CUB_EXTERN_SORT_KEYS(int16_t);
XLA_CUB_EXTERN_SORT_KEYS(int32_t);
XLA_CUB_EXTERN_SORT_KEYS(int64_t);

// Unsigned integer types.
XLA_CUB_EXTERN_SORT_KEYS(uint8_t);
XLA_CUB_EXTERN_SORT_KEYS(uint16_t);
XLA_CUB_EXTERN_SORT_KEYS(uint32_t);
XLA_CUB_EXTERN_SORT_KEYS(uint64_t);

// Pairs with 8-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint64_t);

// Pairs with 16-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint64_t);

// Pairs with signed 32-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint64_t);

// Pairs with unsigned 32-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint64_t);

// Pairs with 64-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint64_t);

// Pairs with f32 keys.
XLA_CUB_EXTERN_SORT_PAIRS(float, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(float, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(float, uint64_t);

#undef XLA_CUB_EXTERN_SORT_KEYS
#undef XLA_CUB_EXTERN_SORT_PAIRS

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUB_SORT_KERNEL_CUDA_H_
