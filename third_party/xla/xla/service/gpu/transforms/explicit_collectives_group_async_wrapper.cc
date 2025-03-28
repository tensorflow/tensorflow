/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/explicit_collectives_group_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<bool> CreateCollectivesGroupAsyncPair(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCall ||
      !instr->frontend_attributes().map().contains(kCollectivesGroupAttr)) {
    return false;
  }
  HloComputation* computation = instr->parent();
  auto new_computation = instr->GetModule()->AddEmbeddedComputation(
      instr->to_apply()->Clone("collectives_group"));
  // Get the shapes for the original instruction.
  std::vector<const Shape*> parameter_shapes(instr->operand_count());
  for (int i = 0; i < instr->operand_count(); ++i) {
    parameter_shapes[i] = &instr->operand(i)->shape();
  }
  std::vector<Shape> start_shapes = {
      ShapeUtil::MakeTupleShapeWithPtrs(parameter_shapes), instr->shape()};
  HloInstruction* async_start =
      computation->AddInstruction(HloInstruction::CreateAsyncStart(
          ShapeUtil::MakeTupleShape(start_shapes), instr->operands(),
          new_computation, "explicit"));
  HloInstruction* async_done = computation->AddInstruction(
      HloInstruction::CreateAsyncDone(instr->shape(), async_start));
  // Forward frontend attributes to both async instructions.
  async_start->set_frontend_attributes(instr->frontend_attributes());
  async_done->set_frontend_attributes(instr->frontend_attributes());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instr, async_done));
  return true;
}
}  // namespace

absl::StatusOr<bool> ExplicitCollectivesGroupAsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool result, CreateCollectivesGroupAsyncPair(instr));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
