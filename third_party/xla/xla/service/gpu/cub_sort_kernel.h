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

#ifndef XLA_SERVICE_GPU_CUB_SORT_KERNEL_H_
#define XLA_SERVICE_GPU_CUB_SORT_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace xla {
namespace gpu {

// Returns nullptr if no error, otherwise the error message as a null-terminated
// string (cudaGetErrorString or similar).
#define XLA_CUB_DECLARE_SORT_KEYS(suffix)                                    \
  const char* CubSortKeys_##suffix(void* d_temp_storage, size_t& temp_bytes, \
                                   const void* d_keys_in, void* d_keys_out,  \
                                   size_t num_items, bool descending);

// Returns nullptr if no error, otherwise the error message as a null-terminated
// string (cudaGetErrorString or similar).
#define XLA_CUB_DECLARE_SORT_PAIRS(suffix)                             \
  const char* CubSortPairs_##suffix(                                   \
      void* d_temp_storage, size_t& temp_bytes, const void* d_keys_in, \
      void* d_keys_out, const void* d_values_in, void* d_values_out,   \
      size_t num_items, bool descending);

XLA_CUB_DECLARE_SORT_KEYS(bf16)
XLA_CUB_DECLARE_SORT_KEYS(f16)
XLA_CUB_DECLARE_SORT_KEYS(f32)
XLA_CUB_DECLARE_SORT_KEYS(f64)
XLA_CUB_DECLARE_SORT_KEYS(s8)
XLA_CUB_DECLARE_SORT_KEYS(s16)
XLA_CUB_DECLARE_SORT_KEYS(s32)
XLA_CUB_DECLARE_SORT_KEYS(s64)
XLA_CUB_DECLARE_SORT_KEYS(u8)
XLA_CUB_DECLARE_SORT_KEYS(u16)
XLA_CUB_DECLARE_SORT_KEYS(u32)
XLA_CUB_DECLARE_SORT_KEYS(u64)

XLA_CUB_DECLARE_SORT_PAIRS(u16_b16)
XLA_CUB_DECLARE_SORT_PAIRS(u16_b32)
XLA_CUB_DECLARE_SORT_PAIRS(u16_b64)
XLA_CUB_DECLARE_SORT_PAIRS(u32_b16)
XLA_CUB_DECLARE_SORT_PAIRS(u32_b32)
XLA_CUB_DECLARE_SORT_PAIRS(u32_b64)
XLA_CUB_DECLARE_SORT_PAIRS(u64_b16)
XLA_CUB_DECLARE_SORT_PAIRS(u64_b32)
XLA_CUB_DECLARE_SORT_PAIRS(u64_b64)

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUB_SORT_KERNEL_H_
