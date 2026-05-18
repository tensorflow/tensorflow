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

#include "xla/hlo/transforms/simplifiers/alias_anti_dependency_inserter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

bool IsMutatingWriter(const HloInstruction* instruction) {
  HloOpcode opcode = instruction->opcode();
  // Filter out non-writing/non-updating/forwarding instructions.
  if (opcode == HloOpcode::kParameter || opcode == HloOpcode::kConstant ||
      opcode == HloOpcode::kBitcast || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kGetTupleElement || opcode == HloOpcode::kTuple) {
    return false;
  }
  // Mutating in-place operators like DynamicUpdateSlice and Scatter are
  // writers.
  return IsDefaultInPlaceOperation(instruction);
}

}  // namespace

absl::StatusOr<bool> AliasAntiDependencyInserter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  AliasInfo alias_info;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, &alias_info));

  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    if (comp->IsFusionComputation()) {
      continue;
    }

    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(comp);

    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      absl::flat_hash_set<HloInstruction*> writers;
      absl::flat_hash_set<HloInstruction*> readers;

      for (const HloValue* value : buffer.values()) {
        HloInstruction* def = value->defining_instruction();
        if (def->parent() == comp && IsMutatingWriter(def)) {
          writers.insert(def);
        }
        for (const HloUse& use : value->GetUses()) {
          if (use.instruction->parent() == comp) {
            readers.insert(use.instruction);
          }
        }
      }

      for (HloInstruction* read : readers) {
        for (HloInstruction* write : writers) {
          if (read == write) {
            continue;
          }
          if (reachability->IsReachable(read, write) ||
              reachability->IsReachable(write, read)) {
            continue;
          }
          TF_RETURN_IF_ERROR(read->AddControlDependencyTo(write));
          added_dependencies_.push_back({read, write});

          std::vector<HloReachabilityMap::Index> predecessor_indices;
          std::vector<HloReachabilityMap::Index> successor_indices;
          for (HloInstruction* inst : comp->instructions()) {
            if (reachability->IsReachable(inst, read)) {
              predecessor_indices.push_back(reachability->GetIndex(inst));
            }
            if (reachability->IsReachable(write, inst)) {
              successor_indices.push_back(reachability->GetIndex(inst));
            }
          }

          for (auto succ_idx : successor_indices) {
            for (auto pred_idx : predecessor_indices) {
              reachability->SetReachable(pred_idx, succ_idx);
            }
          }
          changed = true;
        }
      }
    }
  }

  return changed;
}

absl::Status AliasAntiDependencyInserter::RemoveAddedControlDependencies() {
  for (auto& pair : added_dependencies_) {
    TF_RETURN_IF_ERROR(pair.first->RemoveControlDependencyTo(pair.second));
  }
  added_dependencies_.clear();
  return absl::OkStatus();
}

}  // namespace xla
