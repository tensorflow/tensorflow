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

#include "tensorflow/compiler/xla/service/sort_simplifier.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace {

// If the sort instruction has a tuple shape then looks for unused output
// values and removes them from the sort instruction. Returns true if the
// graph has been modified.
StatusOr<bool> RemoveUnusedOperandFromSort(HloInstruction* sort) {
  if (!sort->shape().IsTuple()) {
    return false;
  }

  HloComputation* computation = sort->parent();

  if (computation->root_instruction() == sort) {
    // Can't analyse users of the root instruction.
    return false;
  }

  absl::flat_hash_set<int64> used_indices;
  for (const HloInstruction* user : sort->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      // Can't analyse users other then get-tuple-element.
      return false;
    }
    used_indices.insert(user->tuple_index());
  }

  // Also note which parameters are used by the comparator computation.
  auto comparator = sort->to_apply();
  for (int64 i = 0; i < sort->operand_count() * 2; ++i) {
    if (comparator->parameter_instruction(i)->user_count() > 0) {
      // operand i corresponds to parameters 2 * i and 2 * i + 1 of the
      // computation.
      used_indices.insert(i / 2);
    }
  }

  if (used_indices.size() == sort->operand_count()) {
    // All operands are used.
    return false;
  }

  std::vector<HloInstruction*> operands;
  std::vector<Shape> new_shapes;
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    if (used_indices.contains(i)) {
      operands.push_back(sort->mutable_operand(i));
      new_shapes.push_back(sort->operand(i)->shape());
    }
  }

  Shape new_sort_shape = new_shapes.size() == 1
                             ? new_shapes[0]
                             : ShapeUtil::MakeTupleShape(new_shapes);
  HloInstruction* new_sort = computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, operands));
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  int64 parameter_number = 0;
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    auto* old_lhs_parameter = comparator->parameter_instruction(i * 2);
    auto* old_rhs_parameter = comparator->parameter_instruction(i * 2 + 1);
    if (used_indices.contains(i)) {
      Shape scalar_shape =
          ShapeUtil::MakeShape(sort->operand(i)->shape().element_type(), {});
      replacements[old_lhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".lhs"));
      ++parameter_number;
      replacements[old_rhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".rhs"));
      ++parameter_number;
    } else {
      replacements[old_lhs_parameter] = nullptr;
      replacements[old_rhs_parameter] = nullptr;
    }
  }
  HloModule* module = sort->GetModule();
  HloComputation* new_compare = module->AddEmbeddedComputation(
      comparator->CloneWithReplacements(std::move(replacements)));
  new_sort->set_to_apply(new_compare);

  // Map from original get-tuple-element tuple index to new HLO instruction
  absl::flat_hash_map<int64, HloInstruction*> result_map;
  if (new_sort->shape().IsTuple()) {
    // Old sort key maps to new sort key.
    int64 new_index = 0;
    for (int64 i = 0; i < sort->operand_count(); ++i) {
      if (used_indices.count(i)) {
        result_map[i] =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                new_shapes[new_index], new_sort, new_index));
        ++new_index;
      }
    }
  } else {
    CHECK_EQ(used_indices.size(), 1);
    result_map[*used_indices.begin()] = new_sort;
  }
  std::vector<HloInstruction*> users(sort->users().begin(),
                                     sort->users().end());
  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(
        user->ReplaceAllUsesWith(result_map.at(user->tuple_index())));
    TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(user));
  }
  return true;
}
}  // namespace

StatusOr<bool> SortSimplifier::Run(HloModule* module) {
  VLOG(2) << "HLO module before SortSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> sort_instrs;
  for (auto* comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(sort_instrs),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kSort;
                    });
  }

  for (HloInstruction* sort_instr : sort_instrs) {
    TF_ASSIGN_OR_RETURN(bool result, RemoveUnusedOperandFromSort(sort_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(2) << "HLO module after SortSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after SortSimplifier";
  }

  return changed;
}
}  // namespace xla
