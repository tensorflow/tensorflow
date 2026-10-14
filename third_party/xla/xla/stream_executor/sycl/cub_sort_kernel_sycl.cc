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

#include "xla/stream_executor/sycl/cub_sort_kernel_sycl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"  // IWYU pragma: keep
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::sycl {
namespace {

namespace ffi = ::xla::ffi;

using SortKeysFn = absl::Status (*)(void* d_temp_storage, size_t& temp_bytes,
                                    const void* d_keys_in, void* d_keys_out,
                                    size_t num_items, bool descending,
                                    size_t batch_size, ::sycl::queue* stream);

absl::StatusOr<SortKeysFn> GetSortKeysFn(xla::PrimitiveType key_type) {
  switch (key_type) {
    case xla::BF16:
      return CubSortKeys<::sycl::ext::oneapi::bfloat16>;
    case xla::F16:
      return CubSortKeys<::sycl::half>;
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

using SortPairsFn = absl::Status (*)(void* d_temp_storage, size_t& temp_bytes,
                                     const void* d_keys_in, void* d_keys_out,
                                     const void* d_values_in,
                                     void* d_values_out, size_t num_items,
                                     bool descending, size_t batch_size,
                                     ::sycl::queue* stream);

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

absl::Status VerifySortKeysBuffers(ffi::AnyBuffer keys,
                                   ffi::Result<ffi::AnyBuffer> keys_out) {
  return ffi::Verify("keys output", *keys_out, ffi::match::Buffer().Like(keys));
}

absl::Status VerifySortPairsBuffers(ffi::AnyBuffer keys, ffi::AnyBuffer values,
                                    ffi::Result<ffi::AnyBuffer> keys_out,
                                    ffi::Result<ffi::AnyBuffer> values_out) {
  ABSL_RETURN_IF_ERROR(ffi::Verify("values input", values,
                              ffi::match::Buffer().WithShapeOf(keys)));
  ABSL_RETURN_IF_ERROR(
      ffi::Verify("keys output", *keys_out, ffi::match::Buffer().Like(keys)));
  return ffi::Verify("values output", *values_out,
                     ffi::match::Buffer().Like(values));
}

//===----------------------------------------------------------------------===//
// CubSortKeys: instantiate + execute
//===----------------------------------------------------------------------===//

// HLO custom call layout:
//   operands: [keys_in]
//   results:  [keys_out, scratch]

absl::StatusOr<std::unique_ptr<int64_t>> CubSortKeysInstantiate(
    ffi::AnyBuffer d_keys_in, ffi::Result<ffi::AnyBuffer> d_keys_out,
    ffi::Result<ffi::BufferR1<xla::U8>> d_temp_storage, bool descending,
    int64_t batch_size) {
  ABSL_RETURN_IF_ERROR(VerifySortKeysBuffers(d_keys_in, d_keys_out));
  if (batch_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be > 0, got ", batch_size));
  }

  ABSL_ASSIGN_OR_RETURN(auto fn, GetSortKeysFn(d_keys_in.element_type()));
  size_t temp_bytes = 0;
  ABSL_RETURN_IF_ERROR(fn(/*d_temp_storage=*/nullptr, temp_bytes,
                     /*d_keys_in=*/nullptr, /*d_keys_out=*/nullptr,
                     d_keys_in.element_count(), /*descending=*/false,
                     batch_size, /*stream=*/nullptr));
  return std::make_unique<int64_t>(static_cast<int64_t>(temp_bytes));
}

absl::Status CubSortKeysExecute(
    ffi::AnyBuffer d_keys_in, ffi::Result<ffi::AnyBuffer> d_keys_out,
    ffi::Result<ffi::BufferR1<xla::U8>> d_temp_storage, bool descending,
    int64_t batch_size, ::sycl::queue* stream) {
  ABSL_RETURN_IF_ERROR(VerifySortKeysBuffers(d_keys_in, d_keys_out));
  if (batch_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be > 0, got ", batch_size));
  }

  ABSL_ASSIGN_OR_RETURN(auto fn, GetSortKeysFn(d_keys_in.element_type()));
  size_t temp_bytes = d_temp_storage->size_bytes();
  return fn(d_temp_storage->untyped_data(), temp_bytes,
            d_keys_in.untyped_data(), d_keys_out->untyped_data(),
            d_keys_in.element_count(), descending, batch_size, stream);
}

XLA_FFI_DEFINE_HANDLER(kCubSortKeysInstantiate, CubSortKeysInstantiate,
                       ffi::Ffi::BindInstantiate()
                           .Arg<ffi::AnyBuffer>()          // d_keys_in
                           .Ret<ffi::AnyBuffer>()          // d_keys_out
                           .Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortKeysExecute, CubSortKeysExecute,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()          // d_keys_in
                           .Ret<ffi::AnyBuffer>()          // d_keys_out
                           .Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size")
                           .Ctx<ffi::PlatformStream<::sycl::queue*>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         xla::gpu::kCubDeviceRadixSortKeysTarget.data(), "SYCL",
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
    ffi::AnyBuffer d_keys_in, ffi::AnyBuffer d_values_in,
    ffi::Result<ffi::AnyBuffer> d_keys_out,
    ffi::Result<ffi::AnyBuffer> d_values_out,
    ffi::Result<ffi::BufferR1<xla::U8>> d_temp_storage, bool descending,
    int64_t batch_size) {
  ABSL_RETURN_IF_ERROR(
      VerifySortPairsBuffers(d_keys_in, d_values_in, d_keys_out, d_values_out));
  if (batch_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be > 0, got ", batch_size));
  }

  ABSL_ASSIGN_OR_RETURN(auto fn, GetSortPairsFn(d_keys_in.element_type(),
                                           xla::primitive_util::BitWidth(
                                               d_values_in.element_type())));
  size_t temp_bytes = 0;
  ABSL_RETURN_IF_ERROR(fn(/*d_temp_storage=*/nullptr, temp_bytes,
                     /*d_keys_in=*/nullptr, /*d_keys_out=*/nullptr,
                     /*d_values_in=*/nullptr, /*d_values_out=*/nullptr,
                     d_keys_in.element_count(), /*descending=*/false,
                     batch_size, /*stream=*/nullptr));
  return std::make_unique<int64_t>(static_cast<int64_t>(temp_bytes));
}

absl::Status CubSortPairsExecute(
    ffi::AnyBuffer d_keys_in, ffi::AnyBuffer d_values_in,
    ffi::Result<ffi::AnyBuffer> d_keys_out,
    ffi::Result<ffi::AnyBuffer> d_values_out,
    ffi::Result<ffi::BufferR1<xla::U8>> d_temp_storage, bool descending,
    int64_t batch_size, ::sycl::queue* stream) {
  ABSL_RETURN_IF_ERROR(
      VerifySortPairsBuffers(d_keys_in, d_values_in, d_keys_out, d_values_out));
  if (batch_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be > 0, got ", batch_size));
  }

  ABSL_ASSIGN_OR_RETURN(auto fn, GetSortPairsFn(d_keys_in.element_type(),
                                           xla::primitive_util::BitWidth(
                                               d_values_in.element_type())));
  size_t temp_bytes = d_temp_storage->size_bytes();
  return fn(d_temp_storage->untyped_data(), temp_bytes,
            d_keys_in.untyped_data(), d_keys_out->untyped_data(),
            d_values_in.untyped_data(), d_values_out->untyped_data(),
            d_keys_in.element_count(), descending, batch_size, stream);
}

XLA_FFI_DEFINE_HANDLER(kCubSortPairsInstantiate, CubSortPairsInstantiate,
                       ffi::Ffi::BindInstantiate()
                           .Arg<ffi::AnyBuffer>()          // d_keys_in
                           .Arg<ffi::AnyBuffer>()          // d_values_in
                           .Ret<ffi::AnyBuffer>()          // d_keys_out
                           .Ret<ffi::AnyBuffer>()          // d_values_out
                           .Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortPairsExecute, CubSortPairsExecute,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()          // d_keys_in
                           .Arg<ffi::AnyBuffer>()          // d_values_in
                           .Ret<ffi::AnyBuffer>()          // d_keys_out
                           .Ret<ffi::AnyBuffer>()          // d_values_out
                           .Ret<ffi::BufferR1<xla::U8>>()  // d_temp_storage
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size")
                           .Ctx<ffi::PlatformStream<::sycl::queue*>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         xla::gpu::kCubDeviceRadixSortPairsTarget.data(),
                         "SYCL",
                         {/* .instantiate = */ kCubSortPairsInstantiate,
                          /* .prepare = */ nullptr,
                          /* .initialize = */ nullptr,
                          /* .execute = */ kCubSortPairsExecute});

}  // namespace
}  // namespace stream_executor::sycl
