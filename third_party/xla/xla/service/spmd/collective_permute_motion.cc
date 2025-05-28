/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/spmd/collective_permute_motion.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::flat_hash_set<HloInstruction*> FindLoopConsts(HloComputation* body) {
  HloInstruction* root = body->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  absl::flat_hash_set<HloInstruction*> loop_consts;
  // Find pass-through inputs.
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    HloInstruction* output = root->mutable_operand(i);
    while (output->opcode() == HloOpcode::kReshape ||
           output->opcode() == HloOpcode::kCopy) {
      output = output->mutable_operand(0);
    }
    if (output->opcode() == HloOpcode::kGetTupleElement &&
        output->tuple_index() == i &&
        output->operand(0) == body->parameter_instruction(0)) {
      loop_consts.insert(output);
    }
  }
  // Find instructions that depend on only loop consts.
  for (HloInstruction* inst : body->MakeInstructionPostOrder()) {
    if (inst->IsConstant() || inst->opcode() == HloOpcode::kIota ||
        inst->opcode() == HloOpcode::kReplicaId ||
        inst->opcode() == HloOpcode::kPartitionId) {
      loop_consts.insert(inst);
      continue;
    }
    if (!inst->IsElementwise() && inst->opcode() != HloOpcode::kBroadcast &&
        inst->opcode() != HloOpcode::kReduce &&
        inst->opcode() != HloOpcode::kReshape &&
        inst->opcode() != HloOpcode::kDynamicSlice &&
        inst->opcode() != HloOpcode::kTranspose) {
      continue;
    }
    if (inst->HasSideEffectNoRecurse()) {
      continue;
    }
    if (absl::c_all_of(inst->operands(), [&](const HloInstruction* operand) {
          return loop_consts.contains(operand);
        })) {
      loop_consts.insert(inst);
    }
  }
  return loop_consts;
}

constexpr int64_t kMaxMovableClusterSize = 8;

// A collective permute may need to be moved with some ops after it. We only
// consider elementwise ops between this collective-permute and loop constants.
struct MovableCluster {
  int64_t root_tuple_index;
  // Last one must be collective-permute.
  std::vector<HloInstruction*> reverse_order_instructions;
  HloInstruction* collective_permute = nullptr;
};

std::optional<MovableCluster> FindMovableClusterAtBodyRoot(
    HloComputation* body, int64_t root_tuple_index,
    const absl::flat_hash_set<HloInstruction*>& loop_consts) {
  HloInstruction* root = body->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  MovableCluster cluster;
  cluster.root_tuple_index = root_tuple_index;
  std::deque<HloInstruction*> queue;
  queue.push_back(root->mutable_operand(root_tuple_index));
  while (!queue.empty()) {
    HloInstruction* visiting = queue.front();
    queue.pop_front();
    if (cluster.reverse_order_instructions.size() >= kMaxMovableClusterSize) {
      VLOG(2) << "Cannot move: too many instructions to move";
      return std::nullopt;
    }
    if (visiting->user_count() > 1) {
      // Let's support only single-use.
      VLOG(2) << "Cannot move: " << visiting->name() << " used multiple times";
      return std::nullopt;
    }
    cluster.reverse_order_instructions.push_back(visiting);
    if (visiting->opcode() == HloOpcode::kCollectivePermute) {
      if (cluster.collective_permute != nullptr) {
        VLOG(2) << "Cannot move: " << visiting->name()
                << " multiple collective permutes";
        return std::nullopt;
      }
      cluster.collective_permute = visiting;
      continue;
    }
    if (!visiting->IsElementwise() || visiting->HasSideEffectNoRecurse()) {
      VLOG(2) << "Cannot move: " << visiting->name() << " unsupported op";
      return std::nullopt;
    }
    for (HloInstruction* operand : visiting->mutable_operands()) {
      if (!loop_consts.contains(operand)) {
        queue.push_back(operand);
      }
    }
  }
  if (cluster.collective_permute == nullptr) {
    return std::nullopt;
  }
  return cluster;
}

absl::flat_hash_set<int64_t> FindIndicesUnusedAfterLoop(HloInstruction* loop) {
  absl::flat_hash_set<int64_t> indices;
  int64_t count = loop->shape().tuple_shapes().size();
  for (int64_t i = 0; i < count; ++i) {
    indices.insert(i);
  }
  for (HloInstruction* user : loop->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      indices.clear();
      break;
    }
    indices.erase(user->tuple_index());
  }
  return indices;
}

absl::StatusOr<bool> MoveCollectivePermutes(HloComputation* computation,
                                            HloInstruction* loop) {
  HloComputation* body = loop->while_body();
  HloInstruction* root = body->root_instruction();
  if (root->opcode() != HloOpcode::kTuple ||
      loop->operand(0)->opcode() != HloOpcode::kTuple) {
    return false;
  }
  auto maybe_induction_var_idx = GetLoopInductionVarTupleIdx(loop);
  if (!maybe_induction_var_idx.has_value()) {
    VLOG(2) << "Skip " << loop->name() << ", no induction var";
    return false;
  }
  absl::flat_hash_map<const HloInstruction*, int64_t> output_appear_counts;
  for (const HloInstruction* operand : root->operands()) {
    auto res = output_appear_counts.emplace(operand, 1);
    if (!res.second) {
      res.first->second++;
    }
  }
  // We require the loop output is unused, so that we don't need to add a final
  // collective-permute after the loop to fix the missing iteration.
  absl::flat_hash_set<int64_t> unused_indices_after_loop =
      FindIndicesUnusedAfterLoop(loop);
  const absl::flat_hash_set<HloInstruction*> loop_consts = FindLoopConsts(body);
  int64_t induction_var_idx = *maybe_induction_var_idx;
  std::vector<HloInstruction*> input_gtes(root->operand_count(), nullptr);
  absl::flat_hash_set<int64_t> multi_use_indices;
  for (HloInstruction* user : body->parameter_instruction(0)->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      VLOG(2) << "Skip " << loop->name() << ", non-GTE input use";
      return false;
    }
    if (multi_use_indices.contains(user->tuple_index())) {
      continue;
    }
    if (input_gtes[user->tuple_index()] != nullptr) {
      multi_use_indices.insert(user->tuple_index());
      input_gtes[user->tuple_index()] = nullptr;
    } else {
      input_gtes[user->tuple_index()] = user;
    }
  }
  HloInstruction* ind_var = input_gtes[induction_var_idx];
  if (ind_var == nullptr || ind_var->shape().dimensions().size() > 0) {
    VLOG(2) << "Skip " << loop->name() << ", non-scalar induction var";
    return false;
  }
  if (root->operand(induction_var_idx)->opcode() != HloOpcode::kAdd &&
      root->operand(induction_var_idx)->opcode() != HloOpcode::kSubtract) {
    VLOG(2) << "Skip " << loop->name() << ", non-add/sub induction var";
    return false;
  }
  if (root->operand(induction_var_idx)->operand(0) == ind_var) {
    if (!root->operand(induction_var_idx)->operand(1)->IsConstant()) {
      VLOG(2) << "Skip " << loop->name() << ", non-add/sub const induction var";
      return false;
    }
  } else if (root->operand(induction_var_idx)->operand(1) == ind_var) {
    if (!root->operand(induction_var_idx)->operand(0)->IsConstant()) {
      VLOG(2) << "Skip " << loop->name() << ", non-add/sub const induction var";
      return false;
    }
  } else {
    return false;
  }
  HloInstruction* ind_var_orig =
      loop->mutable_operand(0)->mutable_operand(induction_var_idx);
  if (!ind_var_orig->IsConstant()) {
    VLOG(2) << "Skip " << loop->name()
            << ", non-constant initial induction var";
    return false;
  }

  bool changed = false;
  std::vector<MovableCluster> movable_outputs;
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    if (output_appear_counts[root->operand(i)] > 1) {
      VLOG(2) << "Skip " << loop->name() << " index " << i
              << " appears multiple times in output.";
      continue;
    }
    if (!unused_indices_after_loop.contains(i)) {
      VLOG(2) << "Skip " << loop->name() << " index " << i
              << " used after loop.";
      continue;
    }
    auto cluster = FindMovableClusterAtBodyRoot(body, i, loop_consts);
    if (!cluster.has_value()) {
      VLOG(2) << "Skip " << loop->name() << " index " << i
              << " did not find a movable cluster.";
      continue;
    }
    HloInstruction* input = input_gtes[cluster->root_tuple_index];
    HloInstruction* cp = cluster->collective_permute;
    if (input == nullptr || cp->operand(0) == input) {
      VLOG(2) << "Skip " << loop->name() << " index " << i
              << " collective-permute already at top.";
      continue;
    }
    const std::vector<HloInstruction*> original_input_users = input->users();
    absl::flat_hash_map<const HloInstruction*, HloInstruction*> replacement;
    replacement[cp->operand(0)] = input;
    for (auto it = cluster->reverse_order_instructions.rbegin();
         it != cluster->reverse_order_instructions.rend(); ++it) {
      HloInstruction* inst = *it;
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* operand : inst->mutable_operands()) {
        auto rit = replacement.find(operand);
        if (rit != replacement.end()) {
          new_operands.push_back(rit->second);
        } else {
          new_operands.push_back(operand);
        }
      }
      HloInstruction* clone = body->AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), new_operands));
      replacement[inst] = clone;
    }
    HloInstruction* new_input =
        replacement[cluster->reverse_order_instructions[0]];
    if (ind_var_orig->parent() != body) {
      ind_var_orig = body->AddInstruction(ind_var_orig->Clone());
    }
    HloInstruction* is_first_iter =
        body->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::ChangeElementType(new_input->shape(), PRED),
            body->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::MakeValidatedScalarShape(PRED).value(), ind_var,
                ind_var_orig, Comparison::Direction::kEq)),
            {}));
    new_input = body->AddInstruction(
        HloInstruction::CreateTernary(new_input->shape(), HloOpcode::kSelect,
                                      is_first_iter, input, new_input));
    for (HloInstruction* user : original_input_users) {
      TF_RETURN_IF_ERROR(input->ReplaceUseWith(user, new_input));
    }
    TF_RETURN_IF_ERROR(root->ReplaceOperandWith(cluster->root_tuple_index,
                                                cp->mutable_operand(0)));
    TF_RETURN_IF_ERROR(body->RemoveInstructionAndUnusedOperands(
        cluster->reverse_order_instructions[0]));
    VLOG(2) << "Moved " << loop->name() << " index " << i;
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> CollectivePermuteMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(bool moved,
                            MoveCollectivePermutes(computation, instr));
        changed |= moved;
      }
    }
  }
  return changed;
}

}  // namespace xla
