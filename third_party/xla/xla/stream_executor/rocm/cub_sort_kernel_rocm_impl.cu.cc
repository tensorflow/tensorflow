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

#include "absl/status/status.h"
#include "Eigen/Core"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hipcub/backend/rocprim/device/device_radix_sort.hpp"
#include "rocm/include/hipcub/backend/rocprim/device/device_segmented_radix_sort.hpp"
#include "rocm/include/rocprim/thread/radix_key_codec.hpp"
#include "rocm/include/rocprim/type_traits.hpp"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/cub_sort_kernel_rocm.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/bfloat16.h"

// Required for sorting Eigen::half and bfloat16.
namespace rocprim {

#if (TF_ROCM_VERSION >= 50200 && TF_ROCM_VERSION < 70000)
namespace detail {
template <>
struct float_bit_mask<Eigen::half> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7C00;
  static constexpr uint16_t mantissa = 0x03FF;
  using bit_type = uint16_t;
};

template <>
struct float_bit_mask<tsl::bfloat16> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7F80;
  static constexpr uint16_t mantissa = 0x007F;
  using bit_type = uint16_t;
};

template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
template <>
struct radix_key_codec_base<tsl::bfloat16>
    : radix_key_codec_floating<tsl::bfloat16, uint16_t> {};
}  // namespace detail
#else   // TF_ROCM_VERSION >= 70000
namespace traits {

template <>
struct define<Eigen::half> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7C00, 0x03FF>;
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      traits::number_format::kind::floating_point_type>;
};

template <>
struct define<tsl::bfloat16> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
  using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
  using number_format = rocprim::traits::number_format::values<
      traits::number_format::kind::floating_point_type>;
};

}  // namespace traits
#endif  // TF_ROCM_VERSION >= 50200 && TF_ROCM_VERSION < 70000

};  // namespace rocprim

namespace stream_executor {
namespace rocm {
namespace {

template <typename KeyT>
absl::Status CubSortKeysImpl(void* d_temp_storage, size_t& temp_bytes,
                             const void* d_keys_in, void* d_keys_out,
                             size_t num_items, bool descending,
                             hipStream_t stream) {
  auto err =
      descending
          ? hipcub::DeviceRadixSort::SortKeysDescending<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceRadixSort::SortKeys<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT>
absl::Status CubSortKeysImpl(void* d_temp_storage, size_t& temp_bytes,
                             const void* d_keys_in, void* d_keys_out,
                             size_t num_items, bool descending,
                             size_t batch_size, hipStream_t stream) {
  if (batch_size == 1) {
    return CubSortKeysImpl<KeyT>(d_temp_storage, temp_bytes, d_keys_in,
                                 d_keys_out, num_items, descending, stream);
  }
  void* d_offsets = static_cast<char*>(d_temp_storage) + temp_bytes;
  int* start_offsets =
      d_temp_storage != nullptr ? static_cast<int*>(d_offsets) : nullptr;
  int* end_offsets = start_offsets != nullptr ? start_offsets + 1 : nullptr;
  auto err =
      descending
          ? hipcub::DeviceSegmentedRadixSort::SortKeysDescending<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                start_offsets, end_offsets, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceSegmentedRadixSort::SortKeys<KeyT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                start_offsets, end_offsets, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsImpl(void* d_temp_storage, size_t& temp_bytes,
                              const void* d_keys_in, void* d_keys_out,
                              const void* d_values_in, void* d_values_out,
                              size_t num_items, bool descending,
                              hipStream_t stream) {
  auto err =
      descending
          ? hipcub::DeviceRadixSort::SortPairsDescending<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceRadixSort::SortPairs<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsImpl(void* d_temp_storage, size_t& temp_bytes,
                              const void* d_keys_in, void* d_keys_out,
                              const void* d_values_in, void* d_values_out,
                              size_t num_items, bool descending,
                              size_t batch_size, hipStream_t stream) {
  if (batch_size == 1) {
    return CubSortPairsImpl<KeyT, ValT>(d_temp_storage, temp_bytes, d_keys_in,
                                        d_keys_out, d_values_in, d_values_out,
                                        num_items, descending, stream);
  }
  void* d_offsets = static_cast<char*>(d_temp_storage) + temp_bytes;
  int* start_offsets =
      d_temp_storage != nullptr ? static_cast<int*>(d_offsets) : nullptr;
  int* end_offsets = start_offsets != nullptr ? start_offsets + 1 : nullptr;
  auto err =
      descending
          ? hipcub::DeviceSegmentedRadixSort::SortPairsDescending<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, batch_size,
                start_offsets, end_offsets, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream)
          : hipcub::DeviceSegmentedRadixSort::SortPairs<KeyT, ValT>(
                d_temp_storage, temp_bytes, static_cast<const KeyT*>(d_keys_in),
                static_cast<KeyT*>(d_keys_out),
                static_cast<const ValT*>(d_values_in),
                static_cast<ValT*>(d_values_out), num_items, batch_size,
                start_offsets, end_offsets, /*begin_bit=*/0,
                /*end_bit=*/sizeof(KeyT) * 8, stream);
  return stream_executor::gpu::ToStatus(err);
}

}  // namespace

// Template instantiations for CubSortKeys
template <typename KeyT>
absl::Status CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         size_t num_items, bool descending, size_t batch_size,
                         hipStream_t stream) {
  return CubSortKeysImpl<KeyT>(d_temp_storage, temp_bytes, d_keys_in,
                               d_keys_out, num_items, descending, batch_size,
                               stream);
}

// Template instantiations for CubSortPairs
template <typename KeyT, typename ValT>
absl::Status CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                          const void* d_keys_in, void* d_keys_out,
                          const void* d_values_in, void* d_values_out,
                          size_t num_items, bool descending, size_t batch_size,
                          hipStream_t stream) {
  return CubSortPairsImpl<KeyT, ValT>(
      d_temp_storage, temp_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, descending, batch_size, stream);
}

#define XLA_CUB_DEFINE_SORT_KEYS(type)                                        \
  template absl::Status CubSortKeys<type>(void*, size_t&, const void*, void*, \
                                          size_t, bool, size_t, hipStream_t);

#define XLA_CUB_DEFINE_SORT_PAIRS(type1, type2)                             \
  template absl::Status CubSortPairs<type1, type2>(                         \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, hipStream_t);

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_DEFINE_SORT_KEYS(hip_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_DEFINE_SORT_KEYS(__half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_DEFINE_SORT_KEYS(float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_DEFINE_SORT_KEYS(double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_DEFINE_SORT_KEYS(int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_DEFINE_SORT_KEYS(int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_DEFINE_SORT_KEYS(int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_DEFINE_SORT_KEYS(int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_DEFINE_SORT_KEYS(uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_DEFINE_SORT_KEYS(uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_DEFINE_SORT_KEYS(uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_DEFINE_SORT_KEYS(uint64_t)
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint16_t)
#endif
#ifdef CUB_TYPE_U8_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint32_t)
#endif
#ifdef CUB_TYPE_U8_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint8_t, uint64_t)
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint16_t)
#endif
#ifdef CUB_TYPE_U16_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint32_t)
#endif
#ifdef CUB_TYPE_U16_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint16_t, uint64_t)
#endif

// Pairs with 32-bit key.
#ifdef CUB_TYPE_S32_B32
XLA_CUB_DEFINE_SORT_PAIRS(int32_t, uint32_t)
#endif
#ifdef CUB_TYPE_U32_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint16_t)
#endif
#ifdef CUB_TYPE_U32_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint32_t)
#endif
#ifdef CUB_TYPE_U32_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint32_t, uint64_t)
#endif
#ifdef CUB_TYPE_F32_B16
XLA_CUB_DEFINE_SORT_PAIRS(float, uint16_t)
#endif
#ifdef CUB_TYPE_F32_B32
XLA_CUB_DEFINE_SORT_PAIRS(float, uint32_t)
#endif
#ifdef CUB_TYPE_F32_B64
XLA_CUB_DEFINE_SORT_PAIRS(float, uint64_t)
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint16_t)
#endif
#ifdef CUB_TYPE_U64_B32
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint32_t)
#endif
#ifdef CUB_TYPE_U64_B64
XLA_CUB_DEFINE_SORT_PAIRS(uint64_t, uint64_t)
#endif

}  // namespace rocm
}  // namespace stream_executor
