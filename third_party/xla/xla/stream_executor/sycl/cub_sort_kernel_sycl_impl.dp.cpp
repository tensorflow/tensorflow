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
#include <exception>
#include <oneapi/dpl/experimental/kernel_templates>
#include <sycl/sycl.hpp>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/sycl/cub_sort_kernel_sycl.h"

namespace stream_executor::sycl {
namespace {

namespace kt = oneapi::dpl::experimental::kt;

// Data per work-item and work-group size recommended by oneDPL. The work-group
// size must be 512 or 1024.
using RadixSortParams = kt::kernel_param<3, 512>;

// oneDPL's radix sort only handles inputs smaller than 2^30 elements.
constexpr size_t kMaxRadixSortItems = size_t{1} << 30;

// Runs `enqueue`, converting the exceptions oneDPL and SYCL use to report
// errors (e.g. a failed scratch allocation) into a status.
template <typename F>
absl::Status EnqueueRadixSort(F&& enqueue) {
  try {
    enqueue();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to enqueue the oneDPL radix sort: ", e.what()));
  } catch (...) {
    return absl::InternalError(
        "Failed to enqueue the oneDPL radix sort: unknown exception");
  }
  return absl::OkStatus();
}

// Enqueues the sort of one segment out-of-place, from `keys_in` into
// `keys_out`. Asynchronous: returns once the sort is submitted to `stream`,
// so the buffers must stay valid until the stream's work completes.
template <typename KeyT>
absl::Status RadixSortKeys(const KeyT* keys_in, KeyT* keys_out,
                           size_t num_items, bool descending,
                           ::sycl::queue* stream) {
  return EnqueueRadixSort([&] {
    if (descending) {
      kt::gpu::radix_sort</*is_ascending=*/false, /*radix_bits=*/8>(
          *stream, keys_in, keys_in + num_items, keys_out, RadixSortParams{});
    } else {
      kt::gpu::radix_sort</*is_ascending=*/true, /*radix_bits=*/8>(
          *stream, keys_in, keys_in + num_items, keys_out, RadixSortParams{});
    }
  });
}

// Enqueues the sort of one segment out-of-place, carrying the values along with
// the keys. Asynchronous, see RadixSortKeys.
template <typename KeyT, typename ValT>
absl::Status RadixSortPairs(const KeyT* keys_in, KeyT* keys_out,
                            const ValT* values_in, ValT* values_out,
                            size_t num_items, bool descending,
                            ::sycl::queue* stream) {
  return EnqueueRadixSort([&] {
    if (descending) {
      kt::gpu::radix_sort_by_key</*is_ascending=*/false, /*radix_bits=*/8>(
          *stream, keys_in, keys_in + num_items, values_in, keys_out,
          values_out, RadixSortParams{});
    } else {
      kt::gpu::radix_sort_by_key</*is_ascending=*/true, /*radix_bits=*/8>(
          *stream, keys_in, keys_in + num_items, values_in, keys_out,
          values_out, RadixSortParams{});
    }
  });
}

}  // namespace

template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         ::sycl::queue* stream) {
  TF_RET_CHECK(num_items > 0) << "num_items must be > 0";
  TF_RET_CHECK(batch_size > 0) << "batch_size must be > 0";
  TF_RET_CHECK(num_items % batch_size == 0)
      << "num_items (" << num_items << ") must be divisible by batch_size ("
      << batch_size << ")";
  const size_t segment_size = num_items / batch_size;
  TF_RET_CHECK(segment_size < kMaxRadixSortItems)
      << "Cannot sort " << segment_size << " keys, the limit is "
      << kMaxRadixSortItems;

  // The compile-time scratch-size query passes null buffers. oneDPL allocates
  // its own temporaries, so no caller scratch is needed.
  if (d_keys_in == nullptr && d_keys_out == nullptr) {
    temp_bytes = 0;
    return absl::OkStatus();
  }

  TF_RET_CHECK(stream != nullptr) << "SYCL queue cannot be null";

  // oneDPL's sort entry points all take a single flat range, so sort each
  // segment with a separate call.
  // TODO(intel-tf): revisit when oneDPL provides a segmented sort. Each call
  // allocates its own device temporaries (hidden from XLA by temp_bytes = 0),
  // so a large batch_size may OOM; bound in-flight segments if needed.
  const KeyT* keys_in = static_cast<const KeyT*>(d_keys_in);
  KeyT* keys_out = static_cast<KeyT*>(d_keys_out);
  for (size_t i = 0; i < batch_size; ++i) {
    ABSL_RETURN_IF_ERROR(RadixSortKeys<KeyT>(keys_in + i * segment_size,
                                        keys_out + i * segment_size,
                                        segment_size, descending, stream));
  }
  return absl::OkStatus();
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          ::sycl::queue* stream) {
  TF_RET_CHECK(num_items > 0) << "num_items must be > 0";
  TF_RET_CHECK(batch_size > 0) << "batch_size must be > 0";
  TF_RET_CHECK(num_items % batch_size == 0)
      << "num_items (" << num_items << ") must be divisible by batch_size ("
      << batch_size << ")";
  const size_t segment_size = num_items / batch_size;
  TF_RET_CHECK(segment_size < kMaxRadixSortItems)
      << "Cannot sort " << segment_size << " keys, the limit is "
      << kMaxRadixSortItems;

  // The compile-time scratch-size query passes null buffers. oneDPL allocates
  // its own temporaries, so no caller scratch is needed.
  if (d_keys_in == nullptr && d_keys_out == nullptr) {
    temp_bytes = 0;
    return absl::OkStatus();
  }

  TF_RET_CHECK(stream != nullptr) << "SYCL queue cannot be null";

  // oneDPL's sort entry points all take a single flat range, so sort each
  // segment with a separate call.
  // TODO(intel-tf): revisit when oneDPL provides a segmented sort. Each call
  // allocates its own device temporaries (hidden from XLA by temp_bytes = 0),
  // so a large batch_size may OOM; bound in-flight segments if needed.
  const KeyT* keys_in = static_cast<const KeyT*>(d_keys_in);
  KeyT* keys_out = static_cast<KeyT*>(d_keys_out);
  const ValT* values_in = static_cast<const ValT*>(d_values_in);
  ValT* values_out = static_cast<ValT*>(d_values_out);
  for (size_t i = 0; i < batch_size; ++i) {
    ABSL_RETURN_IF_ERROR(RadixSortPairs(
        keys_in + i * segment_size, keys_out + i * segment_size,
        values_in + i * segment_size, values_out + i * segment_size,
        segment_size, descending, stream));
  }
  return absl::OkStatus();
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
XLA_CUB_INSTANTIATE_SORT_KEYS(::sycl::ext::oneapi::bfloat16);
XLA_CUB_INSTANTIATE_SORT_KEYS(::sycl::half);
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
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint8_t, uint64_t);

// Pairs with 16-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint16_t, uint64_t);

// Pairs with signed 32-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(int32_t, uint64_t);

// Pairs with unsigned 32-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint32_t, uint64_t);

// Pairs with 64-bit key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(uint64_t, uint64_t);

// Pairs with f32 key.
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint16_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint32_t);
XLA_CUB_INSTANTIATE_SORT_PAIRS(float, uint64_t);

#undef XLA_CUB_INSTANTIATE_SORT_KEYS
#undef XLA_CUB_INSTANTIATE_SORT_PAIRS

}  // namespace stream_executor::sycl
