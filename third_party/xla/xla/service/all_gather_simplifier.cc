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

#include "xla/service/all_gather_simplifier.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<bool> AllGatherSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllGather) {
        continue;
      }
      if (ShapeUtil::Compatible(inst->shape(), inst->operand(0)->shape())) {
        changed = true;
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstruction(inst, inst->mutable_operand(0)));
      } else {
        HloAllGatherInstruction* all_gather =
            Cast<HloAllGatherInstruction>(inst);
        const HloModuleConfig& config = module->config();
        std::optional<ReduceScatterSpec> spec =
            AllGatherDynamicSliceCancellation(
                all_gather, config.num_partitions(), config.replica_count(),
                /*allow_multiple_split_dims=*/false,
                /*allow_intervening_reshape=*/false, /*min_rank=*/1,
                HloPredicateIsOp<HloOpcode::kPartitionId>,
                HloPredicateIsOp<HloOpcode::kReplicaId>);
        if (spec.has_value() &&
            spec->split_dim == all_gather->all_gather_dimension()) {
          changed = true;
          CHECK_EQ(all_gather->users().size(), 1);
          HloInstruction* ds = all_gather->users().front();
          TF_RETURN_IF_ERROR(
              ds->ReplaceAllUsesWith(all_gather->mutable_operand(0)));
          TF_RETURN_IF_ERROR(
              computation->RemoveInstructionAndUnusedOperands(ds));
        }
      }
    }
  }

  return changed;
}

}  // namespace xla
