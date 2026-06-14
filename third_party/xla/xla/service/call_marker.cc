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

#include "xla/service/call_marker.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

HloInstruction* InsertCallMarkerBefore(HloInstruction* instruction,
                                       const FrontendAttributes& attributes) {
  // Gather operand shapes to original call
  std::vector<Shape> operands_shapes;
  operands_shapes.reserve(instruction->operand_count());
  for (HloInstruction* operand : instruction->operands()) {
    operands_shapes.push_back(operand->shape());
  }
  Shape tuple_shape_of_operands = ShapeUtil::MakeTupleShape(operands_shapes);

  // create a custom call before the call with the tuple shape of the operands
  std::unique_ptr<HloInstruction> call_before_ptr =
      HloInstruction::CreateCustomCall(
          tuple_shape_of_operands, instruction->operands(),
          kCallMarkerBeforeTarget, "",
          CustomCallApiVersion::API_VERSION_ORIGINAL);
  call_before_ptr->set_frontend_attributes(attributes);

  return instruction->parent()->AddInstruction(std::move(call_before_ptr));
}

HloInstruction* InsertCallMarkerAfter(HloInstruction* instruction,
                                      const FrontendAttributes& attributes) {
  std::unique_ptr<HloInstruction> call_after_ptr =
      HloInstruction::CreateCustomCall(
          instruction->to_apply()->root_instruction()->shape(), {instruction},
          kCallMarkerAfterTarget, "",
          CustomCallApiVersion::API_VERSION_ORIGINAL);
  call_after_ptr->set_frontend_attributes(attributes);

  return instruction->parent()->AddInstruction(std::move(call_after_ptr));
}

absl::Status WrapCallWithCustomCall(HloInstruction* instruction) {
  const HloComputation* called_computation = instruction->to_apply();

  FrontendAttributes attributes;
  (*attributes.mutable_map())[kCallMarkedComputationAttribute.data()] =
      called_computation->name();

  HloInstruction* call_before = InsertCallMarkerBefore(instruction, attributes);

  // replace the operands of the original call with the get-tuple-element
  // instructions from the custom call
  for (int i = 0; i < call_before->shape().tuple_shapes_size(); ++i) {
    HloInstruction* gte = instruction->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            call_before->shape().tuple_shapes(i), call_before, i));
    RETURN_IF_ERROR(instruction->ReplaceOperandWith(i, gte));
  }

  // replace the original call with the custom call
  HloInstruction* call_after = InsertCallMarkerAfter(instruction, attributes);
  RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(call_after));

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
