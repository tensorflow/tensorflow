/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/async_collective_replacer.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool ShouldBeReplaced(const AsyncCollectiveReplacer::Config& config,
                      const HloInstruction* instruction) {
  HloOpcode op = instruction->opcode();
  std::optional<HloOpcode> wrapped_op =
      (op == HloOpcode::kAsyncStart)
          ? std::make_optional(instruction->async_wrapped_opcode())
          : std::nullopt;

  if (op == HloOpcode::kAllReduceStart || wrapped_op == HloOpcode::kAllReduce) {
    return config.convert_all_reduce(instruction);
  }
  if (op == HloOpcode::kAllGatherStart || wrapped_op == HloOpcode::kAllGather) {
    return config.convert_all_gather(instruction);
  }
  if (op == HloOpcode::kCollectivePermuteStart ||
      wrapped_op == HloOpcode::kCollectivePermute) {
    return config.convert_collective_permute(instruction);
  }
  if (wrapped_op == HloOpcode::kCollectiveBroadcast) {
    return config.convert_collective_broadcast(instruction);
  }
  if (wrapped_op == HloOpcode::kAllToAll) {
    return config.convert_all_to_all(instruction);
  }
  if (wrapped_op == HloOpcode::kReduceScatter) {
    return config.convert_reduce_scatter(instruction);
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> AsyncCollectiveReplacer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // The list of all async pairs that need to be replaced.
    std::vector<std::pair<HloInstruction*, HloInstruction*>> async_pairs;

    // The set of all "control_dep" custom calls that need to be erased.
    // Consider the following HLO.
    //
    //     start = async-start(...)
    //     math = dot_general(...)
    //     done = async-done(start)
    //     custom_call(start, math) target="control_dep"
    //     custom_call(math, done) target="control_dep"
    //
    // We are replacing start and done with a single synchronous collective, so
    // the "control_dep" custom calls cannot be preserved.
    std::vector<HloInstruction*> control_deps_to_remove;

    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (!ShouldBeReplaced(config_, inst)) {
        continue;
      }

      // Mark the async pair for replacement.
      HloInstruction* done = hlo_instruction_utils::async::FindAsyncDone(inst);
      if (done == nullptr) {
        return Internal(
            "Expected to find matching async done op for %s, but found none",
            inst->name());
      }
      async_pairs.push_back({inst, done});
      changed = true;

      // Find any "control_dep" custom calls we need to delete.
      //
      // TODO(mwhittaker): We can keep some of the control_dep calls. For now,
      // we delete all of them.
      for (HloInstruction* user : inst->users()) {
        if (user->IsCustomCall("control_dep")) {
          control_deps_to_remove.push_back(user);
        }
      }
      for (HloInstruction* user : done->users()) {
        if (user->IsCustomCall("control_dep")) {
          control_deps_to_remove.push_back(user);
        }
      }
    }

    // Remove the "control_dep" custom calls.
    absl::flat_hash_set<HloInstruction*> removed;
    for (HloInstruction* control_dep : control_deps_to_remove) {
      if (!removed.contains(control_dep)) {
        ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(control_dep));
        removed.insert(control_dep);
      }
    }

    // Replace the async collectives.
    if (VLOG_IS_ON(1)) {
      for (auto& [inst, done] : async_pairs) {
        VLOG(1) << "Replacing async start/done ops with synchronous op.";
        VLOG(1) << "async start = " << inst->ToString();
        VLOG(1) << "async done = " << done->ToString();
      }
    }
    ABSL_RETURN_IF_ERROR(
        ConvertAsyncCollectivesToSync::ReplaceAsyncInstructionsWithSync(
            computation, async_pairs));
  }
  return changed;
}

}  // namespace xla
