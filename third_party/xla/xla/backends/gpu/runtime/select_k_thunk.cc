/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/select_k_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// SelectKThunk
//===----------------------------------------------------------------------===//

SelectKThunk::SelectKThunk(ThunkInfo thunk_info, std::uint32_t batch_size,
                           std::uint32_t num_elements, std::uint32_t k,
                           xla::PrimitiveType dtype,
                           const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kSelectK, thunk_info),
      batch_size_(batch_size),
      num_elements_(num_elements),
      k_(k),
      dtype_(dtype),
      args_(kernel_arguments.GetArgumentBufferSlices()) {
  CHECK_EQ(args_.size(), 3)
      << "SelectKThunk expects exactly 3 buffer arguments "
         "(input_data, output_data, output_indices)";
}

std::string SelectKThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  return absl::StrCat(indent_str, "SelectKThunk(batch_size=", batch_size_,
                      ", num_elements=", num_elements_, ", k=", k_,
                      ", dtype=", dtype_, ")");
}

// Execute the TopK operation on the GPU stream.
// Maps kernel arguments to device memory and dispatches the appropriate
// select_k_exec implementation based on the platform and data type.
absl::Status SelectKThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Launching " << ToString(0);

  // Map buffer slices to device memory.
  absl::InlinedVector<se::DeviceAddressBase, 3> buffer_args;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceAddressBase buf =
        params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  int device_ordinal = params.buffer_allocations->device_ordinal();
  se::DeviceAddressAllocator* allocator =
      params.buffer_allocations->memory_allocator();
  se::Stream* stream = params.stream;

  // Dispatch to the correct typed implementation based on dtype.
  switch (dtype_) {
    case PrimitiveType::F32:
      return select_k_exec<float>(
          device_ordinal, allocator, stream, buffer_args[0], buffer_args[1],
          buffer_args[2], batch_size_, num_elements_, k_);
    case PrimitiveType::BF16:
      return select_k_exec<::xla::bfloat16>(
          device_ordinal, allocator, stream, buffer_args[0], buffer_args[1],
          buffer_args[2], batch_size_, num_elements_, k_);
    default:
      return absl::UnimplementedError(
          absl::StrCat("SelectKThunk: Unsupported dtype: ",
                       primitive_util::LowercasePrimitiveTypeName(dtype_)));
  }
}

absl::StatusOr<ThunkProto> SelectKThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  SelectKThunkProto* select_k_proto = proto.mutable_select_k_thunk();

  select_k_proto->set_batch_size(batch_size_);
  select_k_proto->set_num_elements(num_elements_);
  select_k_proto->set_k(k_);
  select_k_proto->set_dtype(dtype_);

  for (const BufferAllocation::Slice& arg : args_) {
    TF_ASSIGN_OR_RETURN(*select_k_proto->add_args(), arg.ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<SelectKThunk>> SelectKThunk::FromProto(
    ThunkInfo thunk_info, const SelectKThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<emitters::KernelArgument> arguments;
  arguments.reserve(proto.args().size());
  for (const xla::buffer_assignment::BufferAllocationSliceProto& arg :
       proto.args()) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        BufferAllocation::Slice::FromProto(arg, buffer_allocations));
    emitters::KernelArgument argument{Shape{}, slice};
    arguments.push_back(std::move(argument));
  }
  return std::make_unique<SelectKThunk>(
      thunk_info, proto.batch_size(), proto.num_elements(), proto.k(),
      proto.dtype(), emitters::KernelArguments(std::move(arguments)));
}

}  // namespace xla::gpu
