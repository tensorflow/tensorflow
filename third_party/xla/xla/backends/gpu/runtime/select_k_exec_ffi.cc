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

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/ffi/ffi.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

absl::Status SelectKExecuteFfiHandler(
    int32_t device_ordinal, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, ffi::AnyBuffer d_in, ffi::Result<ffi::AnyBuffer> d_out,
    ffi::Result<ffi::Buffer<xla::S32>> indices_out,
    ffi::Result<ffi::BufferR1<xla::U8>> scratch_buffer) {
  ffi::AnyBuffer::Dimensions in_dims = d_in.dimensions();
  ffi::AnyBuffer::Dimensions out_dims = d_out->dimensions();
  if (in_dims.size() != 1 && in_dims.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelectK expects 1D or 2D input buffer, got ", in_dims.size(), "D"));
  }
  if (out_dims.size() != in_dims.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("SelectK expects output rank to match input rank (",
                     in_dims.size(), " vs ", out_dims.size(), ")"));
  }

  const bool has_batch = in_dims.size() == 2;
  const std::uint32_t batch_size =
      has_batch ? static_cast<std::uint32_t>(in_dims[0]) : 1;
  const std::uint32_t num_elements =
      static_cast<std::uint32_t>(in_dims[has_batch ? 1 : 0]);
  const std::uint32_t k =
      static_cast<std::uint32_t>(out_dims[has_batch ? 1 : 0]);
  const xla::PrimitiveType dtype = d_in.element_type();

  VLOG(3) << "SelectKExecuteFfiHandler: batch_size=" << batch_size
          << ", num_elements=" << num_elements << ", k=" << k
          << ", dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype);

  switch (dtype) {
    case PrimitiveType::F32:
      return select_k_exec<float>(
          device_ordinal, allocator, stream, d_in.device_memory(),
          d_out->device_memory(), indices_out->device_memory(), batch_size,
          num_elements, k, scratch_buffer->device_memory());
    case PrimitiveType::BF16:
      return select_k_exec<::xla::bfloat16>(
          device_ordinal, allocator, stream, d_in.device_memory(),
          d_out->device_memory(), indices_out->device_memory(), batch_size,
          num_elements, k, scratch_buffer->device_memory());
    case PrimitiveType::U64:
      return select_k_exec<std::uint64_t>(
          device_ordinal, allocator, stream, d_in.device_memory(),
          d_out->device_memory(), indices_out->device_memory(), batch_size,
          num_elements, k, scratch_buffer->device_memory());
    default:
      return absl::UnimplementedError(
          absl::StrCat("SelectK: Unsupported dtype: ",
                       primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

XLA_FFI_DEFINE_HANDLER(kSelectKExecute, SelectKExecuteFfiHandler,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::DeviceOrdinal>()
                           .Ctx<ffi::Allocator>()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()           // d_in
                           .Ret<ffi::AnyBuffer>()           // d_out
                           .Ret<ffi::Buffer<xla::S32>>()    // indices_out
                           .Ret<ffi::BufferR1<xla::U8>>(),  // scratch_buffer
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kTopKCustomCallTarget, "CUDA",
                         kSelectKExecute);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kTopKCustomCallTarget, "ROCM",
                         kSelectKExecute);

}  // namespace
}  // namespace xla::gpu
