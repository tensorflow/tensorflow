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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/source_target_pairs.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
using CycleType = SourceTargetPairs::CycleType;
// Finds and returns the non-constant operand in instr.
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

      if (inst->frontend_attributes().map().contains(kSendRecvValidationAttr)) {
        continue;
      }
      SourceTargetPairs::CycleType cycleType =
          GetCycleTypeAndIndices(inst->source_target_pairs()).first;

      if (cycleType == CycleType::kUnknown) {
        continue;
      }

      HloInstruction* whileOp = inst->parent()->WhileCallInstruction();
      if (whileOp == nullptr) {
        VLOG(2) << "No surrounding while op found. Ignoring " << inst->name();
        continue;
      }
      if (!whileOp->frontend_attributes().map().contains(
              "is_pipelined_while_loop")) {
        continue;
      }

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
      SourceTargetPairs sourceTargetPairs(inst->source_target_pairs());
      int64_t num_devices = sourceTargetPairs.GetMaxDeviceNum() + 1;
      int64_t offset = trip_count - num_devices;
      SourceTargetPairs sendRecvValidation;
      for (int64_t currIdx = 0; currIdx < sourceTargetPairs.size(); currIdx++) {
        sendRecvValidation.emplace_back(currIdx, currIdx + offset);
      }

      if (cycleType == CycleType::kBackward) {
        std::reverse(sendRecvValidation.data().begin(),
                     sendRecvValidation.data().end());
      }

      inst->set_frontend_attribute(kSendRecvValidationAttr,
                                   sendRecvValidation.ToString());
      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
