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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> WhileLoopTripCountAnnotator::Run(
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
        } else if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
          // If this is not a trivial loop, it might still be possible to brute
          // force the trip count.
          config.mutable_known_trip_count()->set_n(*trip_count);
        }

        TF_RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
