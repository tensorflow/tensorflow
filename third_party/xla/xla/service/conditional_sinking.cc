/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/conditional_sinking.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {
bool HasPathFromCondToDUS(HloInstruction* cond, HloInstruction* dus) {
  HloInstruction* update = dus->mutable_operand(1);
  std::vector<HloInstruction*> worklist;
  worklist.push_back(cond);
  absl::flat_hash_set<HloInstruction*> visited;
  while (!worklist.empty()) {
    HloInstruction* inst = worklist.back();
    worklist.pop_back();
    // Path to the update is expected to exist and would not create a cycle.
    if (inst == update) {
      continue;
    }
    if (inst == dus) {
      return true;
    }
    for (HloInstruction* user : inst->users()) {
      if (visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      worklist.push_back(user);
    }
  }
  return false;
}

absl::InlinedVector<HloInstruction*, 2> GetUpdateChain(HloInstruction* user,
                                                       HloInstruction* gte) {
  absl::InlinedVector<HloInstruction*, 2> chain;
  while (user->opcode() == HloOpcode::kBitcast && user->user_count() == 1 &&
         !user->IsRoot()) {
    chain.push_back(user);
    gte = user;
    user = user->users()[0];
  }
  if (user->opcode() != HloOpcode::kDynamicUpdateSlice ||
      user->operand(1) != gte) {
    chain.clear();
    return chain;
  }
  chain.push_back(user);
  return chain;
}

HloInstruction* MoveIntoBranch(HloInstruction* inst, HloComputation* branch,
                               HloInstruction* cond_operand) {
  if (inst->opcode() == HloOpcode::kConstant) {
    return branch->AddInstruction(inst->Clone());
  }
  auto cond_indices = cond_operand->OperandIndices(inst);
  if (!cond_indices.empty()) {
    for (auto u : branch->parameter_instruction(0)->users()) {
      if (u->opcode() != HloOpcode::kGetTupleElement) {
        continue;
      }
      if (absl::c_linear_search(cond_indices, u->tuple_index())) {
        return u;
      }
    }
    return branch->parameter_instruction(0)->AddInstruction(
        HloInstruction::CreateGetTupleElement(branch->parameter_instruction(0),
                                              cond_indices[0]));
  }
  auto index = cond_operand->shape().tuple_shapes().size();
  cond_operand->AppendOperand(inst);
  cond_operand->mutable_shape()->mutable_tuple_shapes()->push_back(
      inst->shape());
  *(branch->parameter_instruction(0)->mutable_shape()) = cond_operand->shape();
  return branch->parameter_instruction(0)->AddInstruction(
      HloInstruction::CreateGetTupleElement(branch->parameter_instruction(0),
                                            index));
}

absl::InlinedVector<HloInstruction*, 2> GetSliceChain(HloInstruction* operand) {
  absl::InlinedVector<HloInstruction*, 2> chain;
  while (operand->opcode() == HloOpcode::kBitcast) {
    chain.push_back(operand);
    operand = operand->mutable_operand(0);
  }
  bool added_slice = false;
  while (operand->opcode() == HloOpcode::kSlice ||
         operand->opcode() == HloOpcode::kDynamicSlice) {
    added_slice = true;
    chain.push_back(operand);
    operand = operand->mutable_operand(0);
  }
  if (!added_slice) {
    chain.clear();
  }
  return chain;
}
}  // namespace

absl::StatusOr<bool> ConditionalSinking::HoistUpdates(HloInstruction* cond) {
  if (!cond->shape().IsTuple() || cond->IsRoot() ||
      !absl::c_all_of(cond->branch_computations(),
                      [](HloComputation* c) {
                        return c->root_instruction()->opcode() ==
                               HloOpcode::kTuple;
                      }) ||
      !absl::c_all_of(cond->users(),
                      HloPredicateIsOp<HloOpcode::kGetTupleElement>)) {
    return false;
  }
  bool changed = false;
  for (HloInstruction* gte : cond->users()) {
    CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    if (gte->user_count() != 1 || gte->IsRoot()) {
      continue;
    }
    HloInstruction* user = gte->users()[0];
    auto chain = GetUpdateChain(user, gte);
    if (chain.empty() || HasPathFromCondToDUS(cond, chain.back())) {
      continue;
    }
    auto index = gte->tuple_index();
    changed = true;
    // add the chain into the conditional.
    for (int64_t i = 0; i < cond->branch_count(); ++i) {
      // Before adding instructions to the branch computation, we need to clone
      // it in case other callers have different context
      if (call_counts_[cond->branch_computation(i)] > 1) {
        call_counts_[cond->branch_computation(i)]--;
        cond->set_branch_computation(i,
                                     cond->GetModule()->AddEmbeddedComputation(
                                         cond->branch_computation(i)->Clone()));
        call_counts_[cond->branch_computation(i)]++;
      }
      HloComputation* branch = cond->branch_computation(i);
      HloInstruction* cond_operand = cond->mutable_operand(i + 1);
      HloInstruction* root = branch->root_instruction();
      HloInstruction* inside_operand = root->mutable_operand(index);
      for (HloInstruction* inst : chain) {
        std::vector<HloInstruction*> operands;
        if (inst->opcode() == HloOpcode::kBitcast) {
          operands.push_back(inside_operand);
        } else {
          CHECK_EQ(inst->opcode(), HloOpcode::kDynamicUpdateSlice);
          operands.push_back(
              MoveIntoBranch(inst->mutable_operand(0), branch, cond_operand));
          operands.push_back(inside_operand);
          for (int64_t id = 2; id < inst->operand_count(); ++id) {
            operands.push_back(MoveIntoBranch(inst->mutable_operand(id), branch,
                                              cond_operand));
          }
        }
        inside_operand = branch->AddInstruction(
            inst->CloneWithNewOperands(inst->shape(), operands));
        TF_RETURN_IF_ERROR(
            root->ReplaceOperandWithDifferentShape(index, inside_operand));
        root->mutable_shape()->mutable_tuple_shapes()->at(index) =
            inst->shape();
      }
    }
    for (HloInstruction* inst : chain) {
      cond->mutable_shape()->mutable_tuple_shapes()->at(index) = inst->shape();
      *(gte->mutable_shape()) = inst->shape();
      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(gte));
      TF_RETURN_IF_ERROR(
          inst->parent()->RemoveInstructionAndUnusedOperands(inst));
    }
  }
  return changed;
}

absl::StatusOr<bool> ConditionalSinking::SinkSlices(HloInstruction* cond) {
  bool changed = false;
  for (int64_t i = 0; i < cond->branch_count(); ++i) {
    HloInstruction* cond_operand = cond->mutable_operand(i + 1);
    if (cond_operand->opcode() != HloOpcode::kTuple) {
      continue;
    }
    auto operand_count = cond_operand->operand_count();
    for (int64_t j = 0; j < operand_count; ++j) {
      auto chain = GetSliceChain(cond_operand->mutable_operand(j));
      if (chain.empty()) {
        continue;
      }
      changed = true;
      // Before adding instructions to the branch computation, we need to
      // clone it in case other callers have different context
      if (call_counts_[cond->branch_computation(i)] > 1) {
        call_counts_[cond->branch_computation(i)]--;
        cond->set_branch_computation(i,
                                     cond->GetModule()->AddEmbeddedComputation(
                                         cond->branch_computation(i)->Clone()));
        call_counts_[cond->branch_computation(i)]++;
      }
      if (cond_operand->user_count() > 1) {
        TF_RETURN_IF_ERROR(cond->ReplaceOperandWith(
            i + 1, cond_operand->AddInstruction(cond_operand->Clone())));
        cond_operand = cond->mutable_operand(i + 1);
      }
      HloComputation* branch = cond->branch_computation(i);
      HloInstruction* parameter = branch->parameter_instruction(0);
      HloInstruction* inside_gte = nullptr;
      for (HloInstruction* u : parameter->users()) {
        if (u->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        if (u->tuple_index() == j) {
          inside_gte = u;
          break;
        }
      }
      CHECK_NE(inside_gte, nullptr);
      auto original_users = inside_gte->users();
      for (HloInstruction* inst : chain) {
        std::vector<HloInstruction*> inside_operands;
        if (inst->opcode() == HloOpcode::kSlice ||
            inst->opcode() == HloOpcode::kBitcast) {
          inside_operands.push_back(inside_gte);
        } else {
          CHECK_EQ(inst->opcode(), HloOpcode::kDynamicSlice);
          inside_operands.push_back(inside_gte);
          for (int64_t id = 1; id < inst->operand_count(); ++id) {
            inside_operands.push_back(MoveIntoBranch(inst->mutable_operand(id),
                                                     branch, cond_operand));
          }
        }
        cond_operand->mutable_shape()->mutable_tuple_shapes()->at(j) =
            inst->mutable_operand(0)->shape();
        TF_RETURN_IF_ERROR(cond_operand->ReplaceOperandWithDifferentShape(
            j, inst->mutable_operand(0)));
        *parameter->mutable_shape() = cond_operand->shape();
        *inside_gte->mutable_shape() = inst->mutable_operand(0)->shape();
        HloInstruction* inside_inst = branch->AddInstruction(
            inst->CloneWithNewOperands(inst->shape(), inside_operands));
        TF_RETURN_IF_ERROR(inside_gte->ReplaceAllUsesWithDifferentShape(
            original_users, inside_inst));
        original_users.clear();
        original_users.push_back(inside_inst);
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> ConditionalSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  call_counts_.clear();
  bool changed = false;
  std::vector<HloInstruction*> conds;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(conds),
                    HloPredicateIsOp<HloOpcode::kConditional>);
  }

  for (HloInstruction* cond : conds) {
    for (auto* c : cond->branch_computations()) {
      call_counts_[c]++;
    }
  }

  for (HloInstruction* cond : conds) {
    TF_ASSIGN_OR_RETURN(bool result, SinkSlices(cond));
    changed |= result;
    TF_ASSIGN_OR_RETURN(result, HoistUpdates(cond));
    changed |= result;
  }
  return changed;
}

}  // namespace xla
