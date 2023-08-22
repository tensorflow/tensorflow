/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/spmd/whole_graph_manual_pass.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

// Condition for which we should clear the annotation of the instruction.
bool ShouldClearInstruction(HloInstruction* inst) {
  return inst->opcode() != HloOpcode::kParameter &&
         inst != inst->parent()->root_instruction() &&
         inst->opcode() != HloOpcode::kPartitionId &&
         DynCast<HloCollectiveInstruction>(inst) == nullptr &&
         !inst->HasSideEffectNoRecurse();
}

StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* inst : computation->instructions()) {
    if (ShouldClearInstruction(inst)) {
      inst->clear_sharding();
      changed = true;
      continue;
    }
    if (inst->shape().IsTuple()) {
      inst->set_sharding(
          HloSharding::SingleTuple(inst->shape(), HloSharding::Manual()));
      changed = true;
    } else {
      inst->set_sharding(HloSharding::Manual());
      changed = true;
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> WholeGraphManualPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
