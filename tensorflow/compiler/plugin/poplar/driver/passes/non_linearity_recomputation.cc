/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/non_linearity_recomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

NonLinearityRecomputaion::NonLinearityRecomputaion(
    bool recompute_non_linearities)
    : recompute_non_linearities_(recompute_non_linearities) {}

StatusOr<bool> NonLinearityRecomputaion::Run(HloModule* module) {
  if (!recompute_non_linearities_) {
    return false;
  }

  bool changed = false;

  // Find all the Non Linearities (NLs) which are suitable for recomputation.
  absl::flat_hash_set<HloInstruction*> non_linearities;
  for (HloComputation* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsNonLinearity(inst)) {
        // We currently only consider NLs which directly follow a Norm Training
        // op.
        if (!(IsGTEIndex0(inst->operand(0)) &&
              IsNormTraining(inst->operand(0)->operand(0)))) {
          continue;
        }

        non_linearities.insert(inst);
      }
    }
  }

  for (HloInstruction* nl : non_linearities) {
    HloComputation* comp = nl->parent();
    auto nl_users = nl->users();

    // For every NL user, we clone a unique NL and replace all the uses of NL
    // with the clone.
    // We also add control dependencies to make sure the NL clone is executed
    // as late as possible.
    for (HloInstruction* nl_user : nl_users) {
      HloInstruction* nl_clone = comp->AddInstruction(nl->Clone());
      std::unique_ptr<HloReachabilityMap> reachability_map =
          HloReachabilityMap::Build(comp);
      for (int64 op_idx = 0; op_idx < nl_user->operand_count(); op_idx++) {
        HloInstruction* operand = nl_user->mutable_operand(op_idx);
        if (operand == nl) {
          // Replace any uses of nl with nl_clone.
          nl_user->ReplaceOperandWith(op_idx, nl_clone);
          reachability_map->UpdateReachabilityThroughInstruction(nl_clone);
        } else if (!reachability_map->IsReachable(nl_clone, operand)) {
          operand->AddControlDependencyTo(nl_clone);
          reachability_map->UpdateReachabilityThroughInstruction(nl_clone);
        }
      }
    }

    changed = true;
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
