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

#ifndef XLA_STREAM_EXECUTOR_SYCL_CUB_SORT_KERNEL_SYCL_H_
#define XLA_STREAM_EXECUTOR_SYCL_CUB_SORT_KERNEL_SYCL_H_

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "absl/status/status.h"

namespace stream_executor::sycl {

// Sorts `num_items` keys, split into `batch_size` equally sized contiguous
// segments. A null `d_keys_in` and `d_keys_out` is a request for the required
// scratch size, returned in `temp_bytes`.
template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         ::sycl::queue* stream);

// Sorts `num_items` key/value pairs, split into `batch_size` equally sized
// contiguous segments. A null `d_keys_in` and `d_keys_out` is a request for the
// required scratch size, returned in `temp_bytes`.
template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          ::sycl::queue* stream);

#define XLA_CUB_EXTERN_SORT_KEYS(type)                                        \
  extern template absl::Status CubSortKeys<type>(void*, size_t&, const void*, \
                                                 void*, size_t, bool, size_t, \
                                                 ::sycl::queue*)

#define XLA_CUB_EXTERN_SORT_PAIRS(key_type, val_type)                       \
  extern template absl::Status CubSortPairs<key_type, val_type>(            \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, ::sycl::queue*)

// Floating point types.
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
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, float);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, double);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint8_t, uint64_t);

// Pairs with 16-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, float);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, double);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint16_t, uint64_t);

// Pairs with signed 32-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, float);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, double);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(int32_t, uint64_t);

// Pairs with unsigned 32-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, float);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, double);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint32_t, uint64_t);

// Pairs with 64-bit key.
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, float);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, double);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(uint64_t, uint64_t);

// Pairs with f32 key.
XLA_CUB_EXTERN_SORT_PAIRS(float, float);
XLA_CUB_EXTERN_SORT_PAIRS(float, double);
XLA_CUB_EXTERN_SORT_PAIRS(float, uint16_t);
XLA_CUB_EXTERN_SORT_PAIRS(float, uint32_t);
XLA_CUB_EXTERN_SORT_PAIRS(float, uint64_t);

#undef XLA_CUB_EXTERN_SORT_KEYS
#undef XLA_CUB_EXTERN_SORT_PAIRS

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_CUB_SORT_KERNEL_SYCL_H_
