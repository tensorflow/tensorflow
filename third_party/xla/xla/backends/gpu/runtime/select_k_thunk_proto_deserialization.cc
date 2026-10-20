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

#include "xla/backends/gpu/runtime/select_k_thunk_proto_deserialization.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<CustomCallThunk>> DeserializeSelectKThunkProto(
    Thunk::ThunkInfo thunk_info, const SelectKThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    absl::string_view platform_name,
    const stream_executor::GpuComputeCapability& gpu_compute_capability) {
  if (proto.args_size() != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SelectKThunkProto expects exactly 4 buffer arguments, got %d",
        proto.args_size()));
  }
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice_in,
      BufferAllocation::Slice::FromProto(proto.args(0), buffer_allocations));
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice_out_val,
      BufferAllocation::Slice::FromProto(proto.args(1), buffer_allocations));
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice_out_idx,
      BufferAllocation::Slice::FromProto(proto.args(2), buffer_allocations));
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice_scratch,
      BufferAllocation::Slice::FromProto(proto.args(3), buffer_allocations));

  std::vector<NullableShapedSlice> operands = {
      ShapedSlice{slice_in, ShapeUtil::MakeShape(
                                proto.dtype(),
                                {static_cast<int64_t>(proto.batch_size()),
                                 static_cast<int64_t>(proto.num_elements())})},
  };
  std::vector<NullableShapedSlice> results = {
      ShapedSlice{slice_out_val,
                  ShapeUtil::MakeShape(
                      proto.dtype(), {static_cast<int64_t>(proto.batch_size()),
                                      static_cast<int64_t>(proto.k())})},
      ShapedSlice{
          slice_out_idx,
          ShapeUtil::MakeShape(S32, {static_cast<int64_t>(proto.batch_size()),
                                     static_cast<int64_t>(proto.k())})},
      ShapedSlice{slice_scratch,
                  ShapeUtil::MakeShape(U8, {slice_scratch.size()})},
  };
  return CustomCallThunk::Create(
      std::move(thunk_info), std::string(kTopKCustomCallTarget),
      std::move(operands), std::move(results), /*attributes=*/{},
      /*called_computation=*/nullptr, platform_name, gpu_compute_capability);
}

}  // namespace xla::gpu
