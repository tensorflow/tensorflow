/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"

#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/pattern_matcher.h"

namespace xla {

// Finds and returns the non-constant operand in instr.
//
// CHECK-fails if instr doesn't have exactly one unique non-constant operand.
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr) {
        CHECK_EQ(result, operand);
      }
      result = operand;
    }
  }
  CHECK_NE(result, nullptr);
  return result;
}

// Finds the step (k) for while instruction, if the loop is of the form:
//
// while(cond) {
//   ind_var = ind_var + k
// }
//
// If this pattern is not found, it returns std::nullopt.
std::optional<int64_t> GetStep(HloInstruction* while_inst) {
  // Get the update operation
  std::optional<int64_t> indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(while_inst);
  if (!indvar_tuple_idx) {
    return std::nullopt;
  };
  auto* while_body_indvar_update =
      while_inst->while_body()->root_instruction()->mutable_operand(
          *indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);

  HloInstruction* trip_count_increase_step_instr = nullptr;
  if (!Match(while_body_indvar_update,
             match::AddAnyOrder(match::Op().Is(while_body_indvar),
                                match::Op(&trip_count_increase_step_instr)))) {
    return std::nullopt;
  }
  return LiteralUtil::LiteralAsScalarInt64(
      trip_count_increase_step_instr->literal());
}

absl::StatusOr<bool> CollectivePermuteValidIterationAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* inst : comp->instructions()) {
      if (HloPredicateIsNotOp<HloOpcode::kCollectivePermute>(inst)) {
        continue;
      }

      if (inst->frontend_attributes().map().find(kSendRecvValidationAttr) !=
          inst->frontend_attributes().map().end()) {
        continue;
      }
      auto sourceTargetPairs = inst->source_target_pairs();
      if (GetCycleTypeAndIndices(sourceTargetPairs).first ==
          CycleType::kUnknown) {
        continue;
      }

      VLOG(2) << "Collective permute with cycle: " << inst->ToString();

      int64_t max_device_num = -1;
      for (auto [source, target] : sourceTargetPairs) {
        max_device_num = std::max(std::max(source, target), max_device_num);
      }
      int64_t num_devices = max_device_num + 1;

      HloInstruction* whileOp = inst->parent()->WhileCallInstruction();
      if (whileOp == nullptr) {
        VLOG(2) << "No surrounding while op found. Ignoring " << inst->name();
        continue;
      }
      if (!whileOp->frontend_attributes().map().contains(
              "is_pipelined_while_loop"))
        continue;
      TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                          whileOp->backend_config<WhileLoopBackendConfig>());
      if (!config.has_known_trip_count()) {
        VLOG(2) << "Trip count for while loop (" << whileOp->name()
                << "): unknown";
        continue;
      }

      int64_t trip_count = config.known_trip_count().n();
      std::optional<int64_t> step = GetStep(whileOp);
      VLOG(2) << "Trip count for while loop (" << whileOp->name()
              << "): " << trip_count;
      if (!step) {
        VLOG(2) << "Could not find step for while operation";
        continue;
      }
      VLOG(2) << "Step for while loop (" << whileOp->name() << "): " << *step;
      if (*step != 1) {
        VLOG(2) << "Step is not 1. Skipping...";
        continue;
      }

      // For each source i, the send/recv iteration instances are {i, i+offset}
      // where offset is `number of microbatches * CR - 1`. We know that
      // `trip_count = number_of_microbatches * CR + num_devices - 1` So, offset
      // = number_of_microbatches * CR - 1 = trip_count - num_devices.
      int64_t offset = trip_count - num_devices;

      std::vector<std::pair<int64_t, int64_t>> sendRecvValidation(
          sourceTargetPairs.size());

      for (size_t currIdx = 0; currIdx < sourceTargetPairs.size(); currIdx++) {
        sendRecvValidation[currIdx] = {currIdx, currIdx + offset};
      }

      if (GetCycleTypeAndIndices(sourceTargetPairs).first ==
          CycleType::kBackward) {
        std::reverse(sendRecvValidation.begin(), sendRecvValidation.end());
      }

      xla::FrontendAttributes attributes;
      std::string iteration_instances =
          "{" +
          absl::StrJoin(sendRecvValidation, ",",
                        [](std::string* out, std::pair<int64_t, int64_t> item) {
                          absl::StrAppend(out, "{", item.first, ",",
                                          item.second, "}");
                        }) +
          "}";
      (*attributes.mutable_map())[kSendRecvValidationAttr] =
          iteration_instances;

      inst->add_frontend_attributes(attributes);
      VLOG(1) << "Adding " << kSendRecvValidationAttr << " to " << inst->name()
              << ": " << iteration_instances;
      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
