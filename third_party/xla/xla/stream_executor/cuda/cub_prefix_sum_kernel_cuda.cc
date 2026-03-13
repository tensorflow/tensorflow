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

#include "xla/stream_executor/cuda/cub_prefix_sum_kernel_cuda.h"

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"  // IWYU pragma: keep
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cuda_status.h"

namespace stream_executor::cuda {
namespace {

template <typename KeyT>
absl::Status CubPrefixSumExecute(xla::ffi::AnyBuffer d_temp_storage,
                                 xla::ffi::AnyBuffer d_in,
                                 xla::ffi::Result<xla::ffi::AnyBuffer> d_out,
                                 size_t num_items, CUstream stream) {
  size_t temp_bytes = d_temp_storage.size_bytes();
  return ToStatus(CubPrefixSum<KeyT>(d_temp_storage.untyped_data(), temp_bytes,
                                     d_in.untyped_data(), d_out->untyped_data(),
                                     num_items, stream));
}

template <typename KeyT>
absl::Status CubPrefixSumGetScratchSize(size_t* temp_bytes, size_t num_items) {
  return ToStatus(CubPrefixSum<KeyT>(nullptr, *temp_bytes, nullptr, nullptr,
                                     num_items, nullptr));
}

}  // namespace

#define XLA_CUB_DEFINE_PREFIX_SUM(suffix, type)                                \
  XLA_FFI_DEFINE_HANDLER(kCubPrefixSumExecute_##suffix,                        \
                         CubPrefixSumExecute<type>,                            \
                         xla::ffi::Ffi::Bind()                                 \
                             .Arg<xla::ffi::AnyBuffer>()                       \
                             .Arg<xla::ffi::AnyBuffer>()                       \
                             .Ret<xla::ffi::AnyBuffer>()                       \
                             .Attr<size_t>("num_items")                        \
                             .Ctx<xla::ffi::PlatformStream<CUstream>>());      \
  XLA_FFI_DEFINE_HANDLER(                                                      \
      kCubPrefixSumInitialize_##suffix, CubPrefixSumGetScratchSize<type>,      \
      xla::ffi::Ffi::Bind<xla::ffi::ExecutionStage::kInitialize>()             \
          .Attr<xla::ffi::Pointer<size_t>>("temp_bytes")                       \
          .Attr<size_t>("num_items"));                                         \
  XLA_FFI_REGISTER_HANDLER(                                                    \
      xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.cub_prefix_sum_" #suffix, "CUDA", \
      {/* .instantiate = */ nullptr, /* .prepare = */ nullptr,                 \
       /* .initialize = */ kCubPrefixSumInitialize_##suffix,                   \
       /* .execute = */ kCubPrefixSumExecute_##suffix});

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_DEFINE_PREFIX_SUM(bf16, __nv_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_DEFINE_PREFIX_SUM(f16, __half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_DEFINE_PREFIX_SUM(f32, float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_DEFINE_PREFIX_SUM(f64, double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_DEFINE_PREFIX_SUM(s8, int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_DEFINE_PREFIX_SUM(s16, int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_DEFINE_PREFIX_SUM(s32, int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_DEFINE_PREFIX_SUM(s64, int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_DEFINE_PREFIX_SUM(u8, uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_DEFINE_PREFIX_SUM(u16, uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_DEFINE_PREFIX_SUM(u32, uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_DEFINE_PREFIX_SUM(u64, uint64_t)
#endif

}  // namespace stream_executor::cuda
