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

#include "xla/stream_executor/cuda/cub_sort_kernel_cuda.h"

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"  // IWYU pragma: keep
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cuda_status.h"

namespace stream_executor {
namespace cuda {
namespace {

template <typename KeyT>
absl::Status CubSortKeysExecute(
    xla::ffi::AnyBuffer d_temp_storage, xla::ffi::AnyBuffer d_keys_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out, size_t num_items,
    bool descending, size_t batch_size, CUstream stream) {
  size_t temp_bytes = d_temp_storage.size_bytes();
  return ToStatus(CubSortKeys<KeyT>(
      d_temp_storage.untyped_data(), temp_bytes, d_keys_in.untyped_data(),
      d_keys_out->untyped_data(), num_items, descending, batch_size, stream));
}

template <typename KeyT>
absl::Status CubSortKeysGetScratchSize(size_t* temp_bytes, size_t num_items,
                                       size_t batch_size) {
  return ToStatus(CubSortKeys<KeyT>(nullptr, *temp_bytes, nullptr, nullptr,
                                    num_items, false, batch_size, nullptr));
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsExecute(
    xla::ffi::AnyBuffer d_temp_storage, xla::ffi::AnyBuffer d_keys_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out,
    xla::ffi::AnyBuffer d_values_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_values_out, size_t num_items,
    bool descending, size_t batch_size, CUstream stream) {
  size_t temp_bytes = d_temp_storage.size_bytes();
  return ToStatus(CubSortPairs<KeyT, ValT>(
      d_temp_storage.untyped_data(), temp_bytes, d_keys_in.untyped_data(),
      d_keys_out->untyped_data(), d_values_in.untyped_data(),
      d_values_out->untyped_data(), num_items, descending, batch_size, stream));
}

template <typename KeyT, typename ValT>
absl::Status CubSortPairsGetScratchSize(size_t* temp_bytes, size_t num_items,
                                        size_t batch_size) {
  return ToStatus(CubSortPairs<KeyT, ValT>(nullptr, *temp_bytes, nullptr,
                                           nullptr, nullptr, nullptr, num_items,
                                           false, batch_size, nullptr));
}

}  // namespace

#define XLA_CUB_DEFINE_SORT_KEYS(suffix, type)                                \
  XLA_FFI_DEFINE_HANDLER(kCubSortKeysExecute_##suffix,                        \
                         CubSortKeysExecute<type>,                            \
                         xla::ffi::Ffi::Bind()                                \
                             .Arg<xla::ffi::AnyBuffer>()                      \
                             .Arg<xla::ffi::AnyBuffer>()                      \
                             .Ret<xla::ffi::AnyBuffer>()                      \
                             .Attr<size_t>("num_items")                       \
                             .Attr<bool>("descending")                        \
                             .Attr<size_t>("batch_size")                      \
                             .Ctx<xla::ffi::PlatformStream<CUstream>>());     \
  XLA_FFI_DEFINE_HANDLER(                                                     \
      kCubSortKeysInitialize_##suffix, CubSortKeysGetScratchSize<type>,       \
      xla::ffi::Ffi::Bind<xla::ffi::ExecutionStage::kInitialize>()            \
          .Attr<xla::ffi::Pointer<size_t>>("temp_bytes")                      \
          .Attr<size_t>("num_items")                                          \
          .Attr<size_t>("batch_size"));                                       \
  XLA_FFI_REGISTER_HANDLER(                                                   \
      xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.cub_sort_keys_" #suffix, "CUDA", \
      {/* .instantiate = */ nullptr, /* .prepare = */ nullptr,                \
       /* .initialize = */ kCubSortKeysInitialize_##suffix,                   \
       /* .execute = */ kCubSortKeysExecute_##suffix});

#define XLA_CUB_DEFINE_SORT_PAIRS(suffix, type1, type2)                        \
  XLA_FFI_DEFINE_HANDLER(kCubSortPairsExecute_##suffix,                        \
                         (CubSortPairsExecute<type1, type2>),                  \
                         xla::ffi::Ffi::Bind()                                 \
                             .Arg<xla::ffi::AnyBuffer>()                       \
                             .Arg<xla::ffi::AnyBuffer>()                       \
                             .Ret<xla::ffi::AnyBuffer>()                       \
                             .Arg<xla::ffi::AnyBuffer>()                       \
                             .Ret<xla::ffi::AnyBuffer>()                       \
                             .Attr<size_t>("num_items")                        \
                             .Attr<bool>("descending")                         \
                             .Attr<size_t>("batch_size")                       \
                             .Ctx<xla::ffi::PlatformStream<CUstream>>());      \
  XLA_FFI_DEFINE_HANDLER(                                                      \
      kCubSortPairsInitialize_##suffix,                                        \
      (CubSortPairsGetScratchSize<type1, type2>),                              \
      xla::ffi::Ffi::Bind<xla::ffi::ExecutionStage::kInitialize>()             \
          .Attr<xla::ffi::Pointer<size_t>>("temp_bytes")                       \
          .Attr<size_t>("num_items")                                           \
          .Attr<size_t>("batch_size"));                                        \
  XLA_FFI_REGISTER_HANDLER(                                                    \
      xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.cub_sort_pairs_" #suffix, "CUDA", \
      {/* .instantiate = */ nullptr, /* .prepare = */ nullptr,                 \
       /* .initialize = */ kCubSortPairsInitialize_##suffix,                   \
       /* .execute = */ kCubSortPairsExecute_##suffix});

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
