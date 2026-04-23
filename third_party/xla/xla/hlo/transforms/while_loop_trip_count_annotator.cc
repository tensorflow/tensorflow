/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// For a while loop with known init, step, and trip count, replace all
// get-tuple-element instructions that extract the induction variable from the
// while result with a constant equal to (init + step * trip_count).
static bool ForwardInductionVarToConstants(HloInstruction* while_instr,
                                           int64_t indvar_index,
                                           int64_t final_value) {
  bool changed = false;

  // Collect GTEs first to avoid modifying the user list while iterating.
  std::vector<HloInstruction*> indvar_gtes;
  for (HloInstruction* user : while_instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == indvar_index) {
      indvar_gtes.push_back(user);
    }
  }

  for (HloInstruction* gte : indvar_gtes) {
    HloComputation* comp = gte->parent();
    Literal literal;
    if (gte->shape().element_type() == S32) {
      literal =
          LiteralUtil::CreateR0<int32_t>(static_cast<int32_t>(final_value));
    } else {
      literal = LiteralUtil::CreateR0<int64_t>(final_value);
    }
    HloInstruction* constant = comp->AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));
    CHECK_OK(gte->ReplaceAllUsesWith(constant));
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> WhileLoopTripCountAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }

      if (auto induction_variable_index = GetLoopInductionVarTupleIdx(instr)) {
        // The following analyses all need the induction variable index.
        WhileLoopBackendConfig config;

        // Preserve existing backend config data
        if (auto existing_config =
                instr->backend_config<WhileLoopBackendConfig>();
            existing_config.ok()) {
          config = *existing_config;
        }

        config.mutable_known_induction_variable()->set_tuple_index(
            *induction_variable_index);
        if (auto range = MatchTrivialLoopRange(instr);
            range.has_value() && range->IsBounded() && range->IsStepKnown() &&
            // We store the values in signed integers, so we need to verify
            // they fit.
            range->max()->GetSignedValue() >= 0 &&
            range->min().GetSignedValue() >= 0 &&
            range->step()->GetSignedValue() > 0) {
          int64_t max = range->max()->GetUnsignedValue();
          int64_t min = range->min().GetUnsignedValue();
          int64_t step = range->step()->GetSignedValue();
          int64_t trip_count = (max - min) / step + 1;

          config.mutable_known_trip_count()->set_n(trip_count);
          config.mutable_known_init_step()->set_init(min);
          config.mutable_known_init_step()->set_step(step);

          // The induction variable's value after the loop exits is
          // init + step * trip_count.
          int64_t final_value = min + step * trip_count;
          changed |= ForwardInductionVarToConstants(
              instr, *induction_variable_index, final_value);

        } else if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
          // If this is not a trivial loop, it might still be possible to brute
          // force the trip count.
          config.mutable_known_trip_count()->set_n(*trip_count);
        }

        RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
