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

#include "xla/service/cpu/small_while_loop_hoisting_pass.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

// The call thunk emitter compiles the loop with runtime calls disabled, so dots
// and convolutions that would otherwise call Eigen fall back to a naive loop
// nest. IrEmitter::HandleDot and HandleConvolution also reject operand types
// that the thunk emitter supports (e.g. BF16).
static bool OperandTypesUnsupported(const HloInstruction* instr) {
  constexpr PrimitiveType kSupportedTypes[] = {
      PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128};
  PrimitiveType lhs_type = instr->operand(0)->shape().element_type();
  return lhs_type != instr->operand(1)->shape().element_type() ||
         !absl::c_linear_search(kSupportedTypes, lhs_type);
}

static bool InstructionIsUnavailable(
    const HloInstruction* instr,
    const TargetMachineFeatures& target_machine_features) {
  switch (instr->opcode()) {
    // The following instructions are not currently supported by the call thunk
    // emitter due to how the legacy & thunk emitters interact; specifically,
    // how the run options are passed.
    case HloOpcode::kCustomCall:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFft:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
      return true;

    // Legacy call emitter does not support custom fusions.
    case HloOpcode::kFusion:
      return instr->fusion_kind() == HloInstruction::FusionKind::kCustom;

    case HloOpcode::kDot:
      return OperandTypesUnsupported(instr) ||
             GetDotImplementationStrategy(instr->GetModule()->config(), *instr,
                                          target_machine_features,
                                          /*allow_runtime_calls=*/true) ==
                 DotImplementationStrategy::kEigen;

    case HloOpcode::kConvolution:
      return OperandTypesUnsupported(instr) ||
             PotentiallyImplementedAsEigenConvolution(*instr,
                                                      target_machine_features);

    default:
      return IsCollective(instr);
  }
}

// Simple DFS check to see if a computation contains any instructions that are
// "unavailable" for the call thunk emitter.
static bool ContainsUnavailableInstruction(
    const HloInstruction* instr,
    const TargetMachineFeatures& target_machine_features,
    absl::flat_hash_map<const HloInstruction*, bool>& has_unavailable_instr) {
  if (const auto itr = has_unavailable_instr.find(instr);
      itr != has_unavailable_instr.end()) {
    return itr->second;
  }

  if (InstructionIsUnavailable(instr, target_machine_features)) {
    return has_unavailable_instr.insert({instr, true}).first->second;
  }

  for (const HloComputation* called_comp : instr->called_computations()) {
    for (const HloInstruction* sub_instr : called_comp->instructions()) {
      if (ContainsUnavailableInstruction(sub_instr, target_machine_features,
                                         has_unavailable_instr)) {
        return has_unavailable_instr.insert({instr, true}).first->second;
      }
    }
  }

  return has_unavailable_instr.insert({instr, false}).first->second;
}

SmallWhileLoopHoistingPass::SmallWhileLoopHoistingPass(
    int64_t small_buffer_access_size,
    const TargetMachineFeatures* target_machine_features)
    : small_buffer_access_size_(small_buffer_access_size),
      target_machine_features_(target_machine_features) {}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }

  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, bool> has_unavailable_instr;
  for (HloInstruction* while_instr : while_instrs) {
    if (ContainsUnavailableInstruction(while_instr, *target_machine_features_,
                                       has_unavailable_instr)) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(bool is_small_call_site, IsSmall(while_instr));
    if (!is_small_call_site) {
      continue;
    }

    HloComputation::Builder builder(
        absl::StrCat(while_instr->name(), "_computation"));
    std::vector<HloInstruction*> parameters;
    parameters.reserve(while_instr->operand_count());
    for (HloInstruction* operand : while_instr->operands()) {
      ABSL_ASSIGN_OR_RETURN(HloInstruction * parameter,
                       builder.AddParameter(HloInstruction::CreateParameter(
                           while_instr->operand_index(operand),
                           operand->shape(), operand->name())));
      parameters.push_back(parameter);
    }
    builder.AddInstruction(
        while_instr->CloneWithNewOperands(while_instr->shape(), parameters));

    HloInstruction* call_instruction =
        while_instr->AddInstruction(HloInstruction::CreateCall(
            while_instr->shape(), while_instr->operands(),
            module->AddEmbeddedComputation(builder.Build())));
    call_instruction->add_frontend_attribute("xla_cpu_small_call", "true");

    ABSL_RETURN_IF_ERROR(while_instr->ReplaceAllUsesWith(call_instruction));
    ABSL_RETURN_IF_ERROR(while_instr->SafelyDropAllControlDependencies());
    ABSL_RETURN_IF_ERROR(while_instr->parent()->RemoveInstruction(while_instr));

    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::IsSmall(
    const HloInstruction* instr) {
  HloCostAnalysis cost_analysis(&CpuExecutable::ShapeSizeBytes);
  ABSL_RETURN_IF_ERROR(cost_analysis.RevisitInstruction(instr));
  return cost_analysis.bytes_accessed(*instr) < small_buffer_access_size_;
}

}  // namespace xla::cpu
