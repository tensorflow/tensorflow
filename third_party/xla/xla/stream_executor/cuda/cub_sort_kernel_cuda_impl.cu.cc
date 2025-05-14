/* Copyright 2023 The OpenXLA Authors.

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

#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_segmented_radix_sort.cuh"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/stream_executor/cuda/cub_sort_kernel_cuda.h"

namespace stream_executor {
namespace cuda {

template <typename KeyT>
cudaError_t CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                        const void* d_keys_in, void* d_keys_out,
                        size_t num_items, bool descending, CUstream stream) {
  return descending
             ? cub::DeviceRadixSort::SortKeysDescending<KeyT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream)
             : cub::DeviceRadixSort::SortKeys<KeyT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out), num_items, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream);
}

template <typename KeyT>
cudaError_t CubSortKeys(void* d_temp_storage, size_t& temp_bytes,
                        const void* d_keys_in, void* d_keys_out,
                        size_t num_items, bool descending, size_t batch_size,
                        CUstream stream) {
  if (batch_size == 1) {
    return CubSortKeys<KeyT>(d_temp_storage, temp_bytes, d_keys_in, d_keys_out,
                             num_items, descending, stream);
  }
  void* d_offsets = static_cast<char*>(d_temp_storage) + temp_bytes;
  int* start_offsets =
      d_temp_storage != nullptr ? static_cast<int*>(d_offsets) : nullptr;
  int* end_offsets = start_offsets != nullptr ? start_offsets + 1 : nullptr;
  return descending ? cub::DeviceSegmentedRadixSort::SortKeysDescending<KeyT>(
                          d_temp_storage, temp_bytes,
                          static_cast<const KeyT*>(d_keys_in),
                          static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                          start_offsets, end_offsets, /*begin_bit=*/0,
                          /*end_bit=*/sizeof(KeyT) * 8, stream)
                    : cub::DeviceSegmentedRadixSort::SortKeys<KeyT>(
                          d_temp_storage, temp_bytes,
                          static_cast<const KeyT*>(d_keys_in),
                          static_cast<KeyT*>(d_keys_out), num_items, batch_size,
                          start_offsets, end_offsets, /*begin_bit=*/0,
                          /*end_bit=*/sizeof(KeyT) * 8, stream);
}

template <typename KeyT, typename ValT>
cudaError_t CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         const void* d_values_in, void* d_values_out,
                         size_t num_items, bool descending, CUstream stream) {
  return descending
             ? cub::DeviceRadixSort::SortPairsDescending<KeyT, ValT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out),
                   static_cast<const ValT*>(d_values_in),
                   static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream)
             : cub::DeviceRadixSort::SortPairs<KeyT, ValT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out),
                   static_cast<const ValT*>(d_values_in),
                   static_cast<ValT*>(d_values_out), num_items, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream);
}

template <typename KeyT, typename ValT>
cudaError_t CubSortPairs(void* d_temp_storage, size_t& temp_bytes,
                         const void* d_keys_in, void* d_keys_out,
                         const void* d_values_in, void* d_values_out,
                         size_t num_items, bool descending, size_t batch_size,
                         CUstream stream) {
  if (batch_size == 1) {
    return CubSortPairs<KeyT, ValT>(d_temp_storage, temp_bytes, d_keys_in,
                                    d_keys_out, d_values_in, d_values_out,
                                    num_items, descending, stream);
  }
  void* d_offsets = static_cast<char*>(d_temp_storage) + temp_bytes;
  int* start_offsets =
      d_temp_storage != nullptr ? static_cast<int*>(d_offsets) : nullptr;
  int* end_offsets = start_offsets != nullptr ? start_offsets + 1 : nullptr;
  return descending
             ? cub::DeviceSegmentedRadixSort::SortPairsDescending<KeyT, ValT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out),
                   static_cast<const ValT*>(d_values_in),
                   static_cast<ValT*>(d_values_out), num_items, batch_size,
                   start_offsets, end_offsets, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream)
             : cub::DeviceSegmentedRadixSort::SortPairs<KeyT, ValT>(
                   d_temp_storage, temp_bytes,
                   static_cast<const KeyT*>(d_keys_in),
                   static_cast<KeyT*>(d_keys_out),
                   static_cast<const ValT*>(d_values_in),
                   static_cast<ValT*>(d_values_out), num_items, batch_size,
                   start_offsets, end_offsets, /*begin_bit=*/0,
                   /*end_bit=*/sizeof(KeyT) * 8, stream);
}

#define XLA_CUB_DEFINE_SORT_KEYS(suffix, type)                               \
  template cudaError_t CubSortKeys<type>(void*, size_t&, const void*, void*, \
                                         size_t, bool, size_t, CUstream);

#define XLA_CUB_DEFINE_SORT_PAIRS(suffix, type1, type2)                     \
  template cudaError_t CubSortPairs<type1, type2>(                          \
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool, \
      size_t, CUstream);

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_DEFINE_SORT_KEYS(bf16, __nv_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_DEFINE_SORT_KEYS(f16, __half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_DEFINE_SORT_KEYS(f32, float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_DEFINE_SORT_KEYS(f64, double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_DEFINE_SORT_KEYS(s8, int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_DEFINE_SORT_KEYS(s16, int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_DEFINE_SORT_KEYS(s32, int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_DEFINE_SORT_KEYS(s64, int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_DEFINE_SORT_KEYS(u8, uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_DEFINE_SORT_KEYS(u16, uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_DEFINE_SORT_KEYS(u32, uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_DEFINE_SORT_KEYS(u64, uint64_t)
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
XLA_CUB_DEFINE_SORT_PAIRS(u8_b16, uint8_t, uint16_t)
#endif
#ifdef CUB_TYPE_U8_B32
XLA_CUB_DEFINE_SORT_PAIRS(u8_b32, uint8_t, uint32_t)
#endif
#ifdef CUB_TYPE_U8_B64
XLA_CUB_DEFINE_SORT_PAIRS(u8_b64, uint8_t, uint64_t)
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
XLA_CUB_DEFINE_SORT_PAIRS(u16_b16, uint16_t, uint16_t)
#endif
#ifdef CUB_TYPE_U16_B32
XLA_CUB_DEFINE_SORT_PAIRS(u16_b32, uint16_t, uint32_t)
#endif
#ifdef CUB_TYPE_U16_B64
XLA_CUB_DEFINE_SORT_PAIRS(u16_b64, uint16_t, uint64_t)
#endif

// Pairs with 32-bit key.
#ifdef CUB_TYPE_U32_B16
XLA_CUB_DEFINE_SORT_PAIRS(u32_b16, uint32_t, uint16_t)
#endif
#ifdef CUB_TYPE_U32_B32
XLA_CUB_DEFINE_SORT_PAIRS(u32_b32, uint32_t, uint32_t)
#endif
#ifdef CUB_TYPE_U32_B64
XLA_CUB_DEFINE_SORT_PAIRS(u32_b64, uint32_t, uint64_t)
#endif
#ifdef CUB_TYPE_F32_B16
XLA_CUB_DEFINE_SORT_PAIRS(f32_b16, float, uint16_t)
#endif
#ifdef CUB_TYPE_F32_B32
XLA_CUB_DEFINE_SORT_PAIRS(f32_b32, float, uint32_t)
#endif
#ifdef CUB_TYPE_F32_B64
XLA_CUB_DEFINE_SORT_PAIRS(f32_b64, float, uint64_t)
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
XLA_CUB_DEFINE_SORT_PAIRS(u64_b16, uint64_t, uint16_t)
#endif
#ifdef CUB_TYPE_U64_B32
XLA_CUB_DEFINE_SORT_PAIRS(u64_b32, uint64_t, uint32_t)
#endif
#ifdef CUB_TYPE_U64_B64
XLA_CUB_DEFINE_SORT_PAIRS(u64_b64, uint64_t, uint64_t)
#endif

}  // namespace cuda
}  // namespace stream_executor
