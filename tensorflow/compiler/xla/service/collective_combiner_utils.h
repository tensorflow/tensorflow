/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_

#include <functional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Combines instructions with matching keys together.
//
// Instructions are combined in topological post-order.
//
// `key_fn` should return equal keys for two instructions that might be combined
// together. Instructions will be combined until the threshold for output byte
// size or instruction count is reached.
template <typename K>
StatusOr<bool> CombineInstructionsByKey(
    HloComputation* computation,
    const std::function<absl::optional<K>(const HloInstruction*)>& key_fn,
    const std::function<Status(absl::Span<HloInstruction* const>)>& combine_fn,
    int64_t combine_threshold_bytes, int64_t combine_threshold_count) {
  // Cache keys for each instruction and build sets of instructions with the
  // same key that might be combined together.
  absl::flat_hash_map<HloInstruction*, K> keys;
  absl::flat_hash_map<K, absl::flat_hash_set<HloInstruction*>> groups;

  for (HloInstruction* instruction : computation->instructions()) {
    absl::optional<K> key = key_fn(instruction);
    if (key) {
      keys.insert({instruction, *key});
      groups[*key].insert(instruction);
    }
  }

  bool changed = false;

  // Keys are removed after the instruction is combined (or never will be).
  while (!keys.empty()) {
    std::vector<HloInstruction*> to_combine;
    int64_t to_combine_bytes = 0;
    absl::flat_hash_set<HloInstruction*>* group = nullptr;

    // Recompute reachability after every combine group because we can't
    // maintain a cross group topological order to be able to rely on the
    // transitive dependencies to detect cycles.
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);

    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      auto it = keys.find(instruction);
      if (it == keys.end()) continue;

      // If this is the first instruction, set the active group.
      if (to_combine.empty()) {
        group = &groups.find(it->second)->second;
      }

      // Check instruction is in the active group.
      if (group->find(instruction) == group->end()) {
        continue;
      }

      VLOG(1) << "Considering HLO " << instruction->ToString()
              << " with current set size of " << to_combine_bytes
              << " and current operand count of " << to_combine.size();

      // We do not handle ops that have more than one operand since that is
      // simpler and this pass is the only way to generate such ops.
      if (instruction->operands().size() != 1) {
        VLOG(1) << "Skipping due to " << instruction->operands().size()
                << " operands";
        keys.erase(it);
        continue;
      }

      TF_RET_CHECK(instruction->shape().IsArray());
      int64_t instruction_bytes = ShapeUtil::ByteSizeOf(instruction->shape());

      // If the instruction is greater than the threshold, then we can never
      // combine it with anything.
      if (instruction_bytes > combine_threshold_bytes) {
        VLOG(1) << "Size " << instruction_bytes << " above threshold.";
        keys.erase(it);
        continue;
      }

      if (to_combine_bytes + instruction_bytes > combine_threshold_bytes) {
        VLOG(1) << "Combined size threshold exceeded.";
        break;
      }

      // We can't combine dependent instructions.
      bool is_reachable =
          absl::c_any_of(to_combine, [&](HloInstruction* to_combine_inst) {
            return reachability->IsReachable(to_combine_inst, instruction);
          });
      if (is_reachable) {
        VLOG(1) << "Instruction is reachable.";
        break;
      }

      VLOG(1) << "Adding instruction to set.";
      to_combine.push_back(instruction);
      to_combine_bytes += instruction_bytes;
      keys.erase(it);

      if (to_combine.size() >= combine_threshold_count) {
        VLOG(1) << "Combined count threshold reached.";
        break;
      }
    }

    if (to_combine.size() > 1) {
      TF_RETURN_IF_ERROR(combine_fn(to_combine));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_COMBINER_UTILS_H_
