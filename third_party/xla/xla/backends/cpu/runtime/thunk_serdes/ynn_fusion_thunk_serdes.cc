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

#include "xla/backends/cpu/runtime/thunk_serdes/ynn_fusion_thunk_serdes.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/ynn_emitter.h"
#include "xla/backends/cpu/ynn_fusion_options.pb.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

absl::Status YnnFusionThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& ynn_fusion_thunk = absl::down_cast<const YnnFusionThunk&>(thunk);
  YnnFusionThunkProto* ynn_fusion_proto = proto.mutable_ynn_fusion_thunk();
  ynn_fusion_proto->mutable_options()->set_use_threadpool(
      ynn_fusion_thunk.options().use_threadpool);
  ynn_fusion_proto->set_instruction_id(ynn_fusion_thunk.hlo()->unique_id());

  for (const YnnFusionThunk::Argument& argument :
       ynn_fusion_thunk.arguments()) {
    TF_RETURN_IF_ERROR(
        SerializeSliceShapeIntoProto(argument.slice, argument.shape,
                                     ynn_fusion_proto->add_arguments_shapes()));
  }

  for (const YnnFusionThunk::Result& result : ynn_fusion_thunk.results()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        result.slice, result.shape, ynn_fusion_proto->add_results_shapes()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> YnnFusionThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  const YnnFusionThunkProto& ynn_fusion_proto = proto.ynn_fusion_thunk();

  YnnFusionThunk::Options options = {
      ynn_fusion_proto.options().use_threadpool(),
  };

  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  if (hlo_module == nullptr) {
    return Internal(
        "HLO module is required for YnnFusionThunk deserialization");
  }

  const HloInstruction* hlo = std::invoke([&]() -> const HloInstruction* {
    for (const HloComputation* computation : hlo_module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->unique_id() == ynn_fusion_proto.instruction_id()) {
          return instruction;
        }
      }
    }
    return nullptr;
  });

  if (hlo == nullptr) {
    return Internal(
        "HLO instruction with unique id %d not found in the HLO module",
        ynn_fusion_proto.instruction_id());
  }

  std::vector<YnnFusionThunk::Argument> arguments;
  for (auto& argument_shape_proto : ynn_fusion_proto.arguments_shapes()) {
    TF_ASSIGN_OR_RETURN(auto argument_shape,
                        DeserializeSliceShapeFromProto(argument_shape_proto,
                                                       buffer_allocations));
    arguments.push_back(
        YnnFusionThunk::Argument{argument_shape.first, argument_shape.second});
  }

  std::vector<YnnFusionThunk::Result> results;
  for (auto& result_shape_proto : ynn_fusion_proto.results_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto result_shape,
        DeserializeSliceShapeFromProto(result_shape_proto, buffer_allocations));
    results.push_back(
        YnnFusionThunk::Result{result_shape.first, result_shape.second});
  }

  absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
      absl::Span<const se::DeviceAddressBase> arguments_buffers)>
      builder;
  auto* fusion = Cast<HloFusionInstruction>(hlo);
  const HloComputation* computation = fusion->fused_instructions_computation();

  std::vector<int64_t> captured_arguments_ids;
  captured_arguments_ids.reserve(computation->num_parameters());
  for (const HloInstruction* param : computation->parameter_instructions()) {
    const HloInstruction* operand = fusion->operand(param->parameter_number());
    if (IsConstant(operand)) {
      captured_arguments_ids.push_back(param->parameter_number());
    }
  }

  // Construct YNNPACK subgraph builder from the fusion computation.
  TF_ASSIGN_OR_RETURN(
      builder, EmitYnnFusionBuilder(computation, captured_arguments_ids));

  return YnnFusionThunk::Create(
      std::move(options), std::move(info), hlo, std::move(arguments),
      std::move(results),
      [b = std::move(builder)](auto, auto, auto arg_buffers) mutable {
        return b(arg_buffers);
      },
      captured_arguments_ids);
}

}  // namespace

void RegisterYnnFusionThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(
      Thunk::Kind::kYnnFusion, YnnFusionThunkToProto, YnnFusionThunkFromProto));
}

static bool ynn_fusion_thunk_serdes_registered = [] {
  RegisterYnnFusionThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
