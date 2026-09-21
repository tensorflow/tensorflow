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
#include <cstdint>
#include <oneapi/dpl/experimental/kernel_templates>
#include <sycl/sycl.hpp>

#include "absl/status/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/sycl/cub_sort_kernel_sycl.h"

namespace stream_executor::sycl {

template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         ::sycl::queue* stream) {
  // The compile-time scratch-size query passes null buffers. oneDPL allocates
  // its own temporaries, so no caller scratch is needed.
  if (d_keys_in == nullptr && d_keys_out == nullptr) {
    temp_bytes = 0;
    return absl::OkStatus();
  }

  TF_RET_CHECK(stream != nullptr) << "SYCL queue cannot be null";
  TF_RET_CHECK(num_items > 0) << "num_items must be > 0";
  TF_RET_CHECK(batch_size > 0) << "batch_size must be > 0";
  TF_RET_CHECK(num_items % batch_size == 0)
      << "num_items (" << num_items << ") must be divisible by batch_size ("
      << batch_size << ")";

  // TODO(intel-tf): sort the segments with oneDPL's radix sort.
  return absl::UnimplementedError(
      "CubSortKeys is not implemented for the SYCL backend");
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          ::sycl::queue* stream) {
  // The compile-time scratch-size query passes null buffers. oneDPL allocates
  // its own temporaries, so no caller scratch is needed.
  if (d_keys_in == nullptr && d_keys_out == nullptr) {
    temp_bytes = 0;
    return absl::OkStatus();
  }

  TF_RET_CHECK(stream != nullptr) << "SYCL queue cannot be null";
  TF_RET_CHECK(num_items > 0) << "num_items must be > 0";
  TF_RET_CHECK(batch_size > 0) << "batch_size must be > 0";
  TF_RET_CHECK(num_items % batch_size == 0)
      << "num_items (" << num_items << ") must be divisible by batch_size ("
      << batch_size << ")";

  // TODO(intel-tf): sort the segments with oneDPL's radix sort by key.
  return absl::UnimplementedError(
      "CubSortPairs is not implemented for the SYCL backend");
}

#define XLA_CUB_INSTANTIATE_SORT_KEYS(type)                                   \
  template absl::Status CubSortKeys<type>(void*, size_t&, const void*, void*, \
                                          size_t, bool, size_t,               \
                                          ::sycl::queue*)

#define XLA_CUB_INSTANTIATE_SORT_PAIRS(key_type, val_type)                  \
  template absl::Status CubSortPairs<key_type, val_type>(                   \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, ::sycl::queue*)

// Floating point types.
XLA_CUB_INSTANTIATE_SORT_KEYS(float);
XLA_CUB_INSTANTIATE_SORT_KEYS(double);

// Signed integer types.
XLA_CUB_INSTANTIATE_SORT_KEYS(int8_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(int16_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(int32_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(int64_t);

// Unsigned integer types.
XLA_CUB_INSTANTIATE_SORT_KEYS(uint8_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(uint16_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(uint32_t);
XLA_CUB_INSTANTIATE_SORT_KEYS(uint64_t);

// Pairs with 8-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint64_t);

// Pairs with 16-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint64_t);

// Pairs with signed 32-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint64_t);

// Pairs with unsigned 32-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint64_t);

// Pairs with 64-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint64_t);

// Pairs with f32 key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, float);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, double);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint64_t);

#undef XLA_CUB_INSTANTIATE_SORT_KEYS
#undef XLA_CUB_INSTANTIATE_SORT_PAIRS

}  // namespace stream_executor::sycl
