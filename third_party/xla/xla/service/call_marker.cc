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

#include "xla/service/call_marker.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

HloInstruction* InsertCallMarkerBefore(HloInstruction* instruction) {
  // Gather operand shapes of the original call.
  std::vector<Shape> operands_shapes;
  operands_shapes.reserve(instruction->operand_count());
  for (HloInstruction* operand : instruction->operands()) {
    operands_shapes.push_back(operand->shape());
  }
  Shape tuple_shape_of_operands = ShapeUtil::MakeTupleShape(operands_shapes);

  // Create a custom call before the call with the tuple shape of the operands.
  std::unique_ptr<HloInstruction> call_before_ptr =
      HloInstruction::CreateCustomCall(
          tuple_shape_of_operands, instruction->operands(),
          kCallMarkerBeforeTarget, "",
          CustomCallApiVersion::API_VERSION_ORIGINAL);
  Cast<HloCustomCallInstruction>(call_before_ptr.get())
      ->set_custom_call_has_side_effect(true);

  return instruction->parent()->AddInstruction(std::move(call_before_ptr));
}

HloInstruction* InsertCallMarkerAfter(HloInstruction* instruction) {
  std::unique_ptr<HloInstruction> call_after_ptr =
      HloInstruction::CreateCustomCall(
          instruction->to_apply()->root_instruction()->shape(), {instruction},
          kCallMarkerAfterTarget, "",
          CustomCallApiVersion::API_VERSION_ORIGINAL);
  Cast<HloCustomCallInstruction>(call_after_ptr.get())
      ->set_custom_call_has_side_effect(true);

  return instruction->parent()->AddInstruction(std::move(call_after_ptr));
}

absl::Status PopulateMetadataAndDependencies(HloInstruction* call,
                                             HloInstruction* call_before,
                                             HloInstruction* call_after) {
  // Copy sharding to the 'after' marker, which represents the call output.
  if (call->has_sharding()) {
    call_after->set_sharding(call->sharding());
  }

  // Copy HLO metadata to the 'after' marker.
  call_after->set_metadata(call->metadata());

  // Copy backend config to the 'after' marker.
  if (call->has_backend_config()) {
    call_after->set_raw_backend_config_string(
        call->raw_backend_config_string());
  }

  // Copy original value to the 'after' marker.
  if (call->original_value()) {
    call_after->set_original_value(call->original_value());
  }

  // Set frontend attributes on the 'before' marker.
  FrontendAttributes before_attributes;
  (*before_attributes.mutable_map())[kCallMarkedComputationAttribute.data()] =
      call->to_apply()->name();
  // Call markers declare output-to-operand aliasing to pass values through.
  call_before->set_frontend_attributes(before_attributes);

  // Set frontend attributes on the 'after' marker (merge original call's
  // attributes and the call-marker computation attribute).
  FrontendAttributes after_attributes = call->frontend_attributes();
  (*after_attributes.mutable_map())[kCallMarkedComputationAttribute.data()] =
      call->to_apply()->name();
  (*after_attributes
        .mutable_map())[kCallMarkedInstructionNameAttribute.data()] =
      call->name();
  call_after->set_frontend_attributes(after_attributes);

  // Move control predecessors of the call to the 'before' marker so they
  // execute before the outlined block starts.
  std::vector<HloInstruction*> control_predecessors =
      call->control_predecessors();
  for (HloInstruction* predecessor : control_predecessors) {
    RETURN_IF_ERROR(predecessor->AddControlDependencyTo(call_before));
    RETURN_IF_ERROR(predecessor->RemoveControlDependencyTo(call));
  }

  // Move control successors of the call to the 'after' marker so they
  // execute after the outlined block ends.
  std::vector<HloInstruction*> control_successors = call->control_successors();
  for (HloInstruction* successor : control_successors) {
    RETURN_IF_ERROR(call_after->AddControlDependencyTo(successor));
    RETURN_IF_ERROR(call->RemoveControlDependencyTo(successor));
  }

  return absl::OkStatus();
}

absl::Status WrapCallWithCustomCall(HloInstruction* instruction) {
  HloInstruction* call_before = InsertCallMarkerBefore(instruction);

  // Replace the operands of the original call with the get-tuple-element
  // instructions from the custom call.
  for (int i = 0; i < call_before->shape().tuple_shapes().size(); ++i) {
    HloInstruction* gte = instruction->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            call_before->shape().tuple_shapes(i), call_before, i));
    RETURN_IF_ERROR(instruction->ReplaceOperandWith(i, gte));
  }

  // Replace the original call with the custom call.
  HloInstruction* call_after = InsertCallMarkerAfter(instruction);
  RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(call_after));

  RETURN_IF_ERROR(
      PopulateMetadataAndDependencies(instruction, call_before, call_after));

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> CallMarker::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  std::vector<HloInstruction*> inlineable_calls;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // Don't mark calls that are non-inlineable or have no operands.
      // Instructions that don't have operands can't be reliably marked since
      // we can't introduce data dependencies between them and the wrapping
      // custom calls.
      if (instruction->operand_count() == 0) {
        continue;
      }
      if (inliner_.ShouldInline(*call_graph, instruction)) {
        inlineable_calls.push_back(instruction);
      }
    }
  }
  if (inlineable_calls.empty()) {
    return false;
  }

  for (HloInstruction* instruction : inlineable_calls) {
    RETURN_IF_ERROR(WrapCallWithCustomCall(instruction));
  }

  return true;
}
}  // namespace xla
