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

#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// If the sort instruction has a tuple shape then looks for unused output
// values and removes them from the sort instruction. Returns true if the
// graph has been modified.
absl::StatusOr<bool> RemoveUnusedOperandFromSort(HloInstruction* sort) {
  if (!sort->shape().IsTuple()) {
    return false;
  }

  HloComputation* computation = sort->parent();

  if (computation->root_instruction() == sort) {
    // Can't analyse users of the root instruction.
    return false;
  }

  absl::flat_hash_set<int64_t> used_indices;
  for (const HloInstruction* user : sort->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      // Can't analyse users other then get-tuple-element.
      return false;
    }
    used_indices.insert(user->tuple_index());
  }

  // Also note which parameters are used by the comparator computation.
  auto comparator = sort->to_apply();
  for (int64_t i = 0; i < sort->operand_count() * 2; ++i) {
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
  std::vector<const Shape*> new_shapes;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (used_indices.contains(i)) {
      operands.push_back(sort->mutable_operand(i));
      new_shapes.push_back(&sort->operand(i)->shape());
    }
  }

  Shape new_sort_shape = new_shapes.size() == 1
                             ? *new_shapes[0]
                             : ShapeUtil::MakeTupleShapeWithPtrs(new_shapes);
  HloInstruction* new_sort = computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, operands));
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  int64_t parameter_number = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
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
      comparator->CloneWithReplacements(&replacements));
  new_sort->set_to_apply(new_compare);

  // Map from original get-tuple-element tuple index to new HLO instruction
  absl::flat_hash_map<int64_t, HloInstruction*> result_map;
  if (new_sort->shape().IsTuple()) {
    // Old sort key maps to new sort key.
    int64_t new_index = 0;
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      if (used_indices.count(i)) {
        result_map[i] =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                *new_shapes[new_index], new_sort, new_index));
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
    RETURN_IF_ERROR(
        user->ReplaceAllUsesWith(result_map.at(user->tuple_index())));
    RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(user));
  }
  return true;
}

bool IsConstantOperand(const HloInstruction* operand) {
  if (operand->IsConstant()) {
    return operand->literal().IsAllFirst();
  }
  if (operand->opcode() == HloOpcode::kBroadcast &&
      operand->operand(0)->IsConstant()) {
    return operand->operand(0)->literal().IsAllFirst();
  }
  return false;
}

Literal GetConstantScalarLiteral(const HloInstruction* operand) {
  if (operand->IsConstant()) {
    return LiteralUtil::GetFirstScalarLiteral(operand->literal());
  }
  CHECK_EQ(operand->opcode(), HloOpcode::kBroadcast);
  return LiteralUtil::GetFirstScalarLiteral(operand->operand(0)->literal());
}

absl::StatusOr<bool> RemoveConstantOperandFromSort(
    HloInstruction* sort,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (sort->operand_count() <= 1) {
    return false;
  }
  HloComputation* computation = sort->parent();
  if (computation->root_instruction() == sort) {
    return false;
  }
  for (const HloInstruction* user : sort->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
  }

  absl::flat_hash_set<int64_t> constant_indices;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (IsConstantOperand(sort->operand(i))) {
      constant_indices.insert(i);
    }
  }
  if (constant_indices.size() == sort->operand_count()) {
    constant_indices.erase(0);
  }
  if (constant_indices.empty()) {
    return false;
  }

  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> new_shapes;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (!constant_indices.contains(i)) {
      operands.push_back(sort->mutable_operand(i));
      new_shapes.push_back(&sort->operand(i)->shape());
    }
  }

  Shape new_sort_shape = new_shapes.size() == 1
                             ? *new_shapes[0]
                             : ShapeUtil::MakeTupleShapeWithPtrs(new_shapes);
  HloInstruction* new_sort = computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, operands));

  auto comparator = sort->to_apply();
  HloModule* module = sort->GetModule();
  HloComputation* new_compare =
      module->AddEmbeddedComputation(comparator->Clone());
  std::vector<int64_t> const_indices_desc(constant_indices.begin(),
                                          constant_indices.end());
  absl::c_sort(const_indices_desc, std::greater<int64_t>());
  for (int64_t i : const_indices_desc) {
    Literal scalar_literal = GetConstantScalarLiteral(sort->operand(i));
    HloInstruction* constant = new_compare->AddInstruction(
        HloInstruction::CreateConstant(scalar_literal.Clone()));
    RETURN_IF_ERROR(
        new_compare->parameter_instruction(i * 2 + 1)->ReplaceAllUsesWith(
            constant));
    RETURN_IF_ERROR(
        new_compare->parameter_instruction(i * 2)->ReplaceAllUsesWith(
            constant));
    RETURN_IF_ERROR(new_compare->RemoveParameter(i * 2 + 1));
    RETURN_IF_ERROR(new_compare->RemoveParameter(i * 2));
  }
  new_sort->set_to_apply(new_compare);

  absl::flat_hash_map<int64_t, HloInstruction*> result_map;
  int64_t new_index = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (!constant_indices.contains(i)) {
      if (new_sort->shape().IsTuple()) {
        result_map[i] =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                *new_shapes[new_index], new_sort, new_index));
      } else {
        result_map[i] = new_sort;
      }
      ++new_index;
    } else {
      result_map[i] = sort->mutable_operand(i);
    }
  }

  std::vector<HloInstruction*> users(sort->users().begin(),
                                     sort->users().end());
  for (HloInstruction* user : users) {
    RETURN_IF_ERROR(
        user->ReplaceAllUsesWith(result_map.at(user->tuple_index())));
    RETURN_IF_ERROR(computation->RemoveInstruction(user));
  }
  RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(sort));

  return true;
}

}  // namespace

absl::StatusOr<bool> SortSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before SortSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  bool changed_in_iteration = true;
  while (changed_in_iteration) {
    changed_in_iteration = false;
    std::vector<HloInstruction*> sort_instrs;
    for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
      absl::c_copy_if(comp->instructions(), std::back_inserter(sort_instrs),
                      HloPredicateIsOp<HloOpcode::kSort>);
    }
    for (HloInstruction* sort_instr : sort_instrs) {
      ASSIGN_OR_RETURN(bool const_result, RemoveConstantOperandFromSort(
                                              sort_instr, execution_threads));
      if (const_result) {
        changed_in_iteration = true;
        break;
      }
      ASSIGN_OR_RETURN(bool unused_result,
                       RemoveUnusedOperandFromSort(sort_instr));
      if (unused_result) {
        changed_in_iteration = true;
        break;
      }
    }
    changed |= changed_in_iteration;
  }

  if (changed) {
    AlgebraicSimplifierOptions options;
    AlgebraicSimplifier alg_simp(options);
    RETURN_IF_ERROR(alg_simp.Run(module, execution_threads).status());
    HloDCE dce;
    RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
    VLOG(2) << "HLO module after SortSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after SortSimplifier";
  }

  return changed;
}
}  // namespace xla
