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

#include "tensorflow/compiler/xla/service/stable_sort_expander.h"

#include <limits>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Looks for a iota operand that can be used as tie breaker in the computation.
// If no matching iota operand is found, a iota operand is added to Sort. The
// comparison computation is adjusted to break ties using the values from the
// iota operand.
StatusOr<HloInstruction*> StableSortExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* sort = Cast<HloSortInstruction>(instruction);
  HloComputation* computation = sort->parent();

  HloInstruction* expanded_sort = nullptr;
  absl::flat_hash_set<int64> used_indices;
  int64 iota_index = -1;
  for (const HloInstruction* operand : sort->operands()) {
    // We can only use the iota operand if it has an iota dimension which is the
    // same as the dimension to sort. Also it should have an integral type that
    // is large enough for the number of elements in the sort dimension. For
    // now, we only allow S32, because we expect to find a S32 iota operand for
    // all Sort ops which are created by TopK.
    // TODO(b/122298745): Also support other types.
    if (operand->opcode() == HloOpcode::kIota &&
        Cast<HloIotaInstruction>(operand)->iota_dimension() ==
            sort->sort_dimension() &&
        operand->shape().element_type() == S32) {
      iota_index = sort->operand_index(operand);
      break;
    }
  }

  // If there is currently no iota operand which we could use for making the
  // sort stable, we will have to add a new such operand.
  if (iota_index == -1) {
    Shape iota_shape = sort->operand(0)->shape();
    // We might need to use S64 if the number of elements in the sort dimension
    // is bigger than 2^31 - 1.
    // TODO(b/122298745): Handle Sort ops where S32 is too small for the number
    // of elements in the sort dimension.
    if (iota_shape.dimensions(sort->sort_dimension()) >
        std::numeric_limits<int32>::max()) {
      return Unimplemented(
          "Stable sorting of more than 2^31-1 elements is not implemented");
    }
    iota_shape.set_element_type(S32);
    auto iota = computation->AddInstruction(
        HloInstruction::CreateIota(iota_shape, sort->sort_dimension()));

    // Create a new comparator.
    auto comparator = sort->to_apply();
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;
    std::vector<std::unique_ptr<HloInstruction>> extra_parameters;
    std::vector<HloInstruction*> extra_parameter_ptrs;
    Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
    extra_parameters.push_back(HloInstruction::CreateParameter(
        sort->operand_count() * 2, scalar_shape,
        absl::StrCat("p.", sort->operand_count(), ".lhs")));
    extra_parameter_ptrs.push_back(extra_parameters.back().get());
    extra_parameters.push_back(HloInstruction::CreateParameter(
        sort->operand_count() * 2 + 1, scalar_shape,
        absl::StrCat("p.", sort->operand_count(), ".rhs")));
    extra_parameter_ptrs.push_back(extra_parameters.back().get());
    sort->set_to_apply(sort->GetModule()->AddEmbeddedComputation(
        comparator->CloneWithReplacements(std::move(replacements),
                                          extra_parameter_ptrs)));

    // Replace the original sort op.
    std::vector<HloInstruction*> new_operands(sort->operands().begin(),
                                              sort->operands().end());
    new_operands.push_back(iota);
    std::vector<Shape> new_shapes = sort->operand_count() == 1
                                        ? std::vector<Shape>{sort->shape()}
                                        : sort->shape().tuple_shapes();
    new_shapes.push_back(iota_shape);
    Shape new_sort_shape = ShapeUtil::MakeTupleShape(new_shapes);
    HloInstruction* new_sort = computation->AddInstruction(
        sort->CloneWithNewOperands(new_sort_shape, new_operands));

    // Add a "wrapper" around the new sort op to make sure we have the same
    // shape as before. For the rank 1 case, we only need a GetTupleElement,
    // otherwise we create a Tuple consisting of GetTupleElements of the new
    // sort.
    std::vector<HloInstruction*> tuple_elements;
    tuple_elements.reserve(sort->operand_count());
    for (int64 i = 0; i < sort->operand_count(); ++i) {
      tuple_elements.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              sort->operand(i)->shape(), new_sort, i)));
    }
    expanded_sort = tuple_elements[0];
    if (tuple_elements.size() > 1) {
      expanded_sort = computation->AddInstruction(
          HloInstruction::CreateTuple(tuple_elements));
    }
    sort = Cast<HloSortInstruction>(new_sort);
    iota_index = sort->operand_count() - 1;
  }

  // Modify the computation to break ties using the iota operand.
  auto comparator = sort->to_apply();
  std::vector<HloInstruction*> instructions_postorder =
      comparator->MakeInstructionPostOrder();
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
  // Look up instr in the replacements map, and return either the replacement,
  // or instr, if the replacement isn't present.
  auto replace = [&](HloInstruction* instr) {
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      return instr;
    }
    return it->second;
  };
  HloInstruction* old_root = comparator->root_instruction();
  // The comparison computation gets 2 * n parameters (n being the number of
  // operands of Sort), where parameters 2 * i and 2 * i + 1 correspond to two
  // different scalars of operand i of Sort which are to be compared. The
  // comparison computation should induce a strict weak order, so if
  // to_apply(p1.lhs, p1.rhs, ..., pn.lhs, pn.rhs) is equal to
  // to_apply(p1.rhs, p1.lhs, ..., pn.rhs, pn.lhs), we can conclude that the
  // values to be compared are equivalent, and perform a tie-breaker comparison.
  //
  // We clone each instruction with at least one operand, but use as new
  // operands of the instruction the replacements of the original operands.
  // Parameter 2 * i is replaced by parameter 2 * i + 1 and vice versa. This
  // should make sure that the cloned root instruction gives the result of the
  // comparison computation when being called with each scalar pair reversed.
  // parameters corresponding to the iota operand.
  for (int64 i = 0; i < comparator->num_parameters(); ++i) {
    replacements[comparator->parameter_instruction(i)] =
        comparator->parameter_instruction(i ^ 1);
  }
  HloInstruction* cloned_root = nullptr;
  for (HloInstruction* inst : instructions_postorder) {
    if (inst->operand_count() == 0) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(inst->operand_count());
    for (HloInstruction* operand : inst->operands()) {
      new_operands.push_back(replace(operand));
    }
    auto new_instruction =
        inst->CloneWithNewOperands(inst->shape(), new_operands);
    replacements[inst] = new_instruction.get();
    if (inst == old_root) {
      cloned_root = new_instruction.get();
    }
    comparator->AddInstruction(std::move(new_instruction));
  }
  CHECK_NE(cloned_root, nullptr);
  Shape scalar_pred = ShapeUtil::MakeShape(PRED, {});
  HloInstruction* same =
      comparator->AddInstruction(HloInstruction::CreateCompare(
          scalar_pred, old_root, cloned_root, ComparisonDirection::kEq));
  HloInstruction* tie_breaker =
      comparator->AddInstruction(HloInstruction::CreateCompare(
          scalar_pred, comparator->parameter_instruction(2 * iota_index),
          comparator->parameter_instruction(2 * iota_index + 1),
          ComparisonDirection::kLt));
  HloInstruction* new_root =
      comparator->AddInstruction(HloInstruction::CreateTernary(
          ShapeUtil::MakeShape(PRED, {}), HloOpcode::kSelect, same, tie_breaker,
          old_root));
  comparator->set_root_instruction(new_root);

  return expanded_sort;
}

bool StableSortExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSort &&
         Cast<HloSortInstruction>(instruction)->is_stable();
}
}  // namespace xla
