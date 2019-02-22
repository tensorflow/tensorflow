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

        // We expect three users: nl user (fwd pass), nl user grad (bwd pass)
        // and nl grad (bwd pass).
        if (inst->users().size() != 3) {
          continue;
        }

        non_linearities.insert(inst);
      }
    }
  }

  for (HloInstruction* nl : non_linearities) {
    HloComputation* comp = nl->parent();
    auto nl_users = nl->users();

    // We construct the graph of the computation - this is used for measuring
    // distances between nl_users.
    const auto get_operands = [](HloInstruction* inst) {
      return inst->operands();
    };
    const auto g =
        MetaGraph<HloInstruction*>(comp->root_instruction(), get_operands);

    // We identify the nl user fwd pass op by finding a user of nl such that:
    // * It's connected to the other users, and
    // * The total distance to the other users is the highest.
    HloInstruction* nl_fwd_user = nullptr;
    int64 total_distance = 0;

    // We also keep track of the nearest neigbour (other op using nl) of
    // nl_fwd_user and use that to insert dependencies.
    HloInstruction* nearest_neigbour = nullptr;
    for (int64 idx_a = 0; idx_a < nl_users.size(); idx_a++) {
      bool connected = true;
      HloInstruction* user_a = nl_users[idx_a];
      int64 current_total_distance = 0;
      int64 nearest_neigbour_distance = std::numeric_limits<int64>::max();
      for (int64 idx_b = 0; idx_b < nl_users.size(); idx_b++) {
        if (idx_a == idx_b) {
          continue;
        }
        HloInstruction* user_b = nl_users[idx_b];
        // Check that there exists a path from user a to user b.
        auto optional_a_to_b = g.ShortestPathDistance(user_a, user_b);
        if (optional_a_to_b) {
          int64 a_to_b = *optional_a_to_b;
          current_total_distance += a_to_b;
          if (a_to_b < nearest_neigbour_distance) {
            nearest_neigbour_distance = a_to_b;
            nearest_neigbour = user_b;
          }
        } else {
          connected = false;
          break;
        }
      }
      if (connected) {
        if (current_total_distance > total_distance) {
          total_distance = current_total_distance;
          nl_fwd_user = user_a;
        }
      }
    }

    if (nl_fwd_user && nearest_neigbour) {
      // Clone the NL and then replace all the uses of it in the backwards pass
      // and add control dependencies to make sure it gets executed as late as
      // possible.
      HloInstruction* nl_clone = comp->AddInstruction(nl->Clone());
      std::unique_ptr<HloReachabilityMap> reachability_map =
          HloReachabilityMap::Build(comp);

      // Replace operands - make sure to update reachability.
      for (HloInstruction* user : nl_users) {
        if (user == nl_fwd_user) {
          continue;
        }
        for (int64 op_idx = 0; op_idx < user->operand_count(); op_idx++) {
          HloInstruction* operand = user->mutable_operand(op_idx);
          if (operand == nl) {
            // Replace any uses of nl with nl_clone.
            user->ReplaceOperandWith(op_idx, nl_clone);
            reachability_map->UpdateReachabilityThroughInstruction(nl_clone);
          }
        }
      }

      // Add control dependencies to the inputs of the nearest neigbour making
      // sure we don't create cycles.
      for (HloInstruction* operand : nearest_neigbour->operands()) {
        if (operand != nl_clone &&
            !reachability_map->IsReachable(nl_clone, operand)) {
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
