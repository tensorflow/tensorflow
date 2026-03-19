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
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"  // IWYU pragma: keep
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor {
namespace cuda {
namespace {

using SortKeysFn = cudaError_t (*)(void* d_temp_storage, size_t& temp_bytes,
                                   const void* d_keys_in, void* d_keys_out,
                                   size_t num_items, bool descending,
                                   size_t batch_size, CUstream stream);

using SortPairsFn = cudaError_t (*)(void* d_temp_storage, size_t& temp_bytes,
                                    const void* d_keys_in, void* d_keys_out,
                                    const void* d_values_in, void* d_values_out,
                                    size_t num_items, bool descending,
                                    size_t batch_size, CUstream stream);

absl::StatusOr<SortKeysFn> GetSortKeysFn(xla::PrimitiveType key_type) {
  switch (key_type) {
    case xla::BF16:
      return CubSortKeys<__nv_bfloat16>;
    case xla::F16:
      return CubSortKeys<__half>;
    case xla::F32:
      return CubSortKeys<float>;
    case xla::F64:
      return CubSortKeys<double>;
    case xla::S8:
      return CubSortKeys<int8_t>;
    case xla::S16:
      return CubSortKeys<int16_t>;
    case xla::S32:
      return CubSortKeys<int32_t>;
    case xla::S64:
      return CubSortKeys<int64_t>;
    case xla::U8:
      return CubSortKeys<uint8_t>;
    case xla::U16:
      return CubSortKeys<uint16_t>;
    case xla::U32:
      return CubSortKeys<uint32_t>;
    case xla::U64:
      return CubSortKeys<uint64_t>;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported key type for CUB sort: ",
          xla::primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

template <typename KeyT>
absl::StatusOr<SortPairsFn> GetSortPairsFnForValueWidth(int value_bit_width) {
  switch (value_bit_width) {
    case 16:
      return CubSortPairs<KeyT, uint16_t>;
    case 32:
      return CubSortPairs<KeyT, uint32_t>;
    case 64:
      return CubSortPairs<KeyT, uint64_t>;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported value bit width for CUB sort: ", value_bit_width));
  }
}

absl::StatusOr<SortPairsFn> GetSortPairsFn(xla::PrimitiveType key_type,
                                           int value_bit_width) {
  switch (key_type) {
    case xla::U8:
      return GetSortPairsFnForValueWidth<uint8_t>(value_bit_width);
    case xla::U16:
      return GetSortPairsFnForValueWidth<uint16_t>(value_bit_width);
    case xla::S32:
      return GetSortPairsFnForValueWidth<int32_t>(value_bit_width);
    case xla::U32:
      return GetSortPairsFnForValueWidth<uint32_t>(value_bit_width);
    case xla::F32:
      return GetSortPairsFnForValueWidth<float>(value_bit_width);
    case xla::U64:
      return GetSortPairsFnForValueWidth<uint64_t>(value_bit_width);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported key type for CUB sort pairs: ",
          xla::primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

// Computes the scratch buffer size needed for CUB sort, including batch
// offsets if batch_size > 1.
absl::StatusOr<int64_t> ComputeScratchSize(SortKeysFn fn, int64_t num_items,
                                           int64_t batch_size) {
  size_t temp_bytes = 0;
  RETURN_IF_ERROR(ToStatus(fn(nullptr, temp_bytes, nullptr, nullptr, num_items,
                              false, batch_size, nullptr)));
  int64_t scratch_size = temp_bytes;
  if (batch_size > 1) {
    scratch_size += sizeof(int32_t) - scratch_size % sizeof(int32_t);
    scratch_size += (batch_size + 1) * sizeof(int32_t);
  }
  return scratch_size;
}

absl::StatusOr<int64_t> ComputeScratchSize(SortPairsFn fn, int64_t num_items,
                                           int64_t batch_size) {
  size_t temp_bytes = 0;
  RETURN_IF_ERROR(ToStatus(fn(nullptr, temp_bytes, nullptr, nullptr, nullptr,
                              nullptr, num_items, false, batch_size, nullptr)));
  int64_t scratch_size = temp_bytes;
  if (batch_size > 1) {
    scratch_size += sizeof(int32_t) - scratch_size % sizeof(int32_t);
    scratch_size += (batch_size + 1) * sizeof(int32_t);
  }
  return scratch_size;
}

// N pairs of [start_offset, end_offset) require (N+1) storage.
int64_t GetOffsetsSize(int64_t batch_size) {
  return (batch_size + 1) * sizeof(int32_t);
}

// Copies segment offsets [0, segment_size, 2*segment_size, ...] to device
// memory at the end of the scratch buffer for batched segmented sort.
absl::Status CopyOffsets(void* scratch, size_t scratch_bytes,
                         int64_t batch_size, int64_t segment_size,
                         CUstream stream) {
  int64_t offsets_size = GetOffsetsSize(batch_size);
  char* offsets_buffer =
      static_cast<char*>(scratch) + scratch_bytes - offsets_size;
  std::vector<int32_t> h_offsets(batch_size + 1);
  for (int32_t i = 0; i <= batch_size; ++i) {
    h_offsets[i] = i * segment_size;
  }
  return ToStatus(cudaMemcpyAsync(offsets_buffer, h_offsets.data(),
                                  offsets_size, cudaMemcpyHostToDevice,
                                  stream));
}

//===----------------------------------------------------------------------===//
// CubSortKeys: instantiate + execute
//===----------------------------------------------------------------------===//

// HLO custom call layout:
//   operands: [keys_in]
//   results:  [keys_out, scratch]

absl::StatusOr<std::unique_ptr<int64_t>> CubSortKeysInstantiate(
    xla::ffi::AnyBuffer d_keys_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_temp_storage, bool descending,
    int64_t batch_size) {
  ASSIGN_OR_RETURN(auto fn, GetSortKeysFn(d_keys_in.element_type()));
  int64_t num_items = d_keys_in.element_count();
  ASSIGN_OR_RETURN(int64_t scratch_size,
                   ComputeScratchSize(fn, num_items, batch_size));
  return std::make_unique<int64_t>(scratch_size);
}

absl::Status CubSortKeysExecute(
    xla::ffi::AnyBuffer d_keys_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_temp_storage, bool descending,
    int64_t batch_size, CUstream stream) {
  ASSIGN_OR_RETURN(auto fn, GetSortKeysFn(d_keys_in.element_type()));
  size_t num_items = d_keys_in.element_count();
  size_t temp_bytes = d_temp_storage->size_bytes();
  if (batch_size > 1) {
    RETURN_IF_ERROR(CopyOffsets(d_temp_storage->untyped_data(), temp_bytes,
                                batch_size, num_items / batch_size, stream));
    temp_bytes -= GetOffsetsSize(batch_size);
  }
  return ToStatus(fn(d_temp_storage->untyped_data(), temp_bytes,
                     d_keys_in.untyped_data(), d_keys_out->untyped_data(),
                     num_items, descending, batch_size, stream));
}

XLA_FFI_DEFINE_HANDLER(kCubSortKeysInstantiate, CubSortKeysInstantiate,
                       xla::ffi::Ffi::BindInstantiate()
                           .Arg<xla::ffi::AnyBuffer>()  // d_keys_in
                           .Ret<xla::ffi::AnyBuffer>()  // d_keys_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortKeysExecute, CubSortKeysExecute,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // d_keys_in
                           .Ret<xla::ffi::AnyBuffer>()  // d_keys_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size")
                           .Ctx<xla::ffi::PlatformStream<CUstream>>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         xla::gpu::kCubDeviceRadixSortKeysTarget.data(), "CUDA",
                         {/* .instantiate = */ kCubSortKeysInstantiate,
                          /* .prepare = */ nullptr,
                          /* .initialize = */ nullptr,
                          /* .execute = */ kCubSortKeysExecute});

//===----------------------------------------------------------------------===//
// CubSortPairs: instantiate + execute
//===----------------------------------------------------------------------===//

// HLO custom call layout:
//   operands: [keys_in, values_in]
//   results:  [keys_out, values_out, scratch]

absl::StatusOr<std::unique_ptr<int64_t>> CubSortPairsInstantiate(
    xla::ffi::AnyBuffer d_keys_in, xla::ffi::AnyBuffer d_values_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_values_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_temp_storage, bool descending,
    int64_t batch_size) {
  ASSIGN_OR_RETURN(auto fn, GetSortPairsFn(d_keys_in.element_type(),
                                           xla::primitive_util::BitWidth(
                                               d_values_in.element_type())));
  int64_t num_items = d_keys_in.element_count();
  ASSIGN_OR_RETURN(int64_t scratch_size,
                   ComputeScratchSize(fn, num_items, batch_size));
  return std::make_unique<int64_t>(scratch_size);
}

absl::Status CubSortPairsExecute(
    xla::ffi::AnyBuffer d_keys_in, xla::ffi::AnyBuffer d_values_in,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_keys_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_values_out,
    xla::ffi::Result<xla::ffi::AnyBuffer> d_temp_storage, bool descending,
    int64_t batch_size, CUstream stream) {
  ASSIGN_OR_RETURN(auto fn, GetSortPairsFn(d_keys_in.element_type(),
                                           xla::primitive_util::BitWidth(
                                               d_values_in.element_type())));
  size_t num_items = d_keys_in.element_count();
  size_t temp_bytes = d_temp_storage->size_bytes();
  if (batch_size > 1) {
    RETURN_IF_ERROR(CopyOffsets(d_temp_storage->untyped_data(), temp_bytes,
                                batch_size, num_items / batch_size, stream));
    temp_bytes -= GetOffsetsSize(batch_size);
  }
  return ToStatus(fn(d_temp_storage->untyped_data(), temp_bytes,
                     d_keys_in.untyped_data(), d_keys_out->untyped_data(),
                     d_values_in.untyped_data(), d_values_out->untyped_data(),
                     num_items, descending, batch_size, stream));
}

XLA_FFI_DEFINE_HANDLER(kCubSortPairsInstantiate, CubSortPairsInstantiate,
                       xla::ffi::Ffi::BindInstantiate()
                           .Arg<xla::ffi::AnyBuffer>()  // d_keys_in
                           .Arg<xla::ffi::AnyBuffer>()  // d_values_in
                           .Ret<xla::ffi::AnyBuffer>()  // d_keys_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_values_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortPairsExecute, CubSortPairsExecute,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // d_keys_in
                           .Arg<xla::ffi::AnyBuffer>()  // d_values_in
                           .Ret<xla::ffi::AnyBuffer>()  // d_keys_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_values_out
                           .Ret<xla::ffi::AnyBuffer>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size")
                           .Ctx<xla::ffi::PlatformStream<CUstream>>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         xla::gpu::kCubDeviceRadixSortPairsTarget.data(),
                         "CUDA",
                         {/* .instantiate = */ kCubSortPairsInstantiate,
                          /* .prepare = */ nullptr,
                          /* .initialize = */ nullptr,
                          /* .execute = */ kCubSortPairsExecute});

}  // namespace
}  // namespace cuda
}  // namespace stream_executor
