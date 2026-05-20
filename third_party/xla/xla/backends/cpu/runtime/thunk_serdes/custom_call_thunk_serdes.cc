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

#include "xla/backends/cpu/runtime/thunk_serdes/custom_call_thunk_serdes.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

absl::Status CustomCallThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& custom_call_thunk =
      absl::down_cast<const CustomCallThunk&>(thunk);
  CustomCallThunkProto* custom_call_proto = proto.mutable_custom_call_thunk();

  custom_call_proto->set_target_name(custom_call_thunk.target_name());
  custom_call_proto->set_backend_config(custom_call_thunk.backend_config());
  custom_call_proto->set_api_version(custom_call_thunk.api_version());

  for (size_t i = 0;
       i < custom_call_thunk.op_buffers().arguments_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        custom_call_thunk.op_buffers().arguments_buffers[i],
        custom_call_thunk.op_buffers().arguments_shapes[i],
        custom_call_proto->mutable_op_buffers()->add_arguments_shapes()));
  }

  for (size_t i = 0; i < custom_call_thunk.op_buffers().results_buffers.size();
       ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        custom_call_thunk.op_buffers().results_buffers[i],
        custom_call_thunk.op_buffers().results_shapes[i],
        custom_call_proto->mutable_op_buffers()->add_results_shapes()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> CustomCallThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  CustomCallThunk::OpBuffers op_buffers;
  for (const ShapeBufferAllocationSliceProto& arg_buff_shape :
       proto.custom_call_thunk().op_buffers().arguments_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto args_slice_shape,
        DeserializeSliceShapeFromProto(arg_buff_shape, buffer_allocations));

    const auto& [args_buffer, args_shape] = args_slice_shape;
    op_buffers.arguments_buffers.push_back(args_buffer);
    op_buffers.arguments_shapes.push_back(args_shape);
  }

  for (const ShapeBufferAllocationSliceProto& res_buff_shape :
       proto.custom_call_thunk().op_buffers().results_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto res_slice_shape,
        DeserializeSliceShapeFromProto(res_buff_shape, buffer_allocations));

    const auto& [res_buffer, res_shape] = res_slice_shape;
    op_buffers.results_buffers.push_back(res_buffer);
    op_buffers.results_shapes.push_back(res_shape);
  }

  return CustomCallThunk::Create(
      std::move(info), proto.custom_call_thunk().target_name(),
      std::move(op_buffers), proto.custom_call_thunk().backend_config(),
      proto.custom_call_thunk().api_version());
}

}  // namespace

void RegisterCustomCallThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(Thunk::Kind::kCustomCall,
                                               CustomCallThunkToProto,
                                               CustomCallThunkFromProto));
}

// Statically registers the CustomCallThunk serialization/deserialization logic.
static bool custom_call_thunk_serdes_registered = [] {
  RegisterCustomCallThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
