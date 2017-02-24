/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include <algorithm>
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {

// Returns whether `a` and `b` are equivalent for the purposes of this pass.
bool AreEquivalentReshapes(const HloInstruction* a, const HloInstruction* b) {
  if (a->opcode() != b->opcode() ||
      !ShapeUtil::SameDimensions(a->shape(), b->shape())) {
    return false;
  }
  switch (a->opcode()) {
    case HloOpcode::kTranspose:
      return a->dimensions() == b->dimensions();
    case HloOpcode::kReshape:
      return ShapeUtil::SameDimensions(a->operand(0)->shape(),
                                       b->operand(0)->shape());
    default:
      return false;
  }
}

bool IsElementwiseOfEquivalentReshapesOrTransposes(
    const HloInstruction* instruction) {
  const std::vector<HloInstruction*>& operands = instruction->operands();
  return instruction->IsElementwise() && instruction->operand_count() > 0 &&
         std::all_of(operands.begin(), operands.end(),
                     [](const HloInstruction* instruction) {
                       // We require operand have no other users as otherwise
                       // this is not a clear win.
                       return 1 == instruction->users().size();
                     }) &&
         // Check whether each operand beyond the first is equivalent to the
         // first.
         std::all_of(operands.begin(), operands.end(),
                     [&operands](const HloInstruction* operand) {
                       return AreEquivalentReshapes(operands[0], operand);
                     });
}

// Try to sink any reshape or transpose operands of `instruction` across it. We
// do so if `instruction` is elementwise and all operands are equivalent
// reshapes or transposes.
bool TrySinkReshapeOrTranspose(HloComputation* computation,
                               HloInstruction* instruction) {
  if (IsElementwiseOfEquivalentReshapesOrTransposes(instruction)) {
    std::vector<HloInstruction*> operands = instruction->operands();
    auto old_reshape = operands[0];
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = operands[i]->mutable_operand(0);
    }
    if (HloOpcode::kFusion == instruction->opcode()) {
      // Here we already know `instruction` is elementwise, and no operand is
      // implicit broadcast as if it were the operands would not be equivalent
      // reshapes, so all the fused instructions have the same dimensions.
      for (const auto& fused_instruction : instruction->fused_instructions()) {
        *fused_instruction->mutable_shape()->mutable_dimensions() =
            operands[0]->shape().dimensions();
      }
    }
    auto new_elementwise =
        computation->AddInstruction(instruction->CloneWithNewOperands(
            // `instruction` may change the element type, e.g., from
            //   operands[0] -> reshape -> convert (`instruction`)
            // to
            //   operands[0] -> convert' -> reshape'
            //
            // In this case, convert' should have the same element type as
            // `convert` and the same dimensions as operands[0].
            ShapeUtil::MakeShape(
                instruction->shape().element_type(),
                AsInt64Slice(operands[0]->shape().dimensions())),
            operands));
    std::unique_ptr<HloInstruction> new_reshape;
    switch (old_reshape->opcode()) {
      case HloOpcode::kReshape:
        new_reshape = HloInstruction::CreateReshape(instruction->shape(),
                                                    new_elementwise);
        break;
      case HloOpcode::kTranspose:
        new_reshape = HloInstruction::CreateTranspose(
            instruction->shape(), new_elementwise, old_reshape->dimensions());
        break;
      default:
        LOG(FATAL) << "Bad opcode";
    }
    TF_CHECK_OK(computation->ReplaceWithNewInstruction(instruction,
                                                       std::move(new_reshape)));
    return true;
  }
  return false;
}

}  // namespace

StatusOr<bool> ReshapeMover::Run(HloModule* module) {
  return std::any_of(
      module->computations().begin(), module->computations().end(),
      [](const std::unique_ptr<HloComputation>& computation) {
        std::list<HloInstruction*> postorder =
            computation->MakeInstructionPostOrder();
        return std::any_of(postorder.begin(), postorder.end(),
                           [&computation](HloInstruction* instruction) {
                             return TrySinkReshapeOrTranspose(computation.get(),
                                                              instruction);
                           });
      });
}

}  // namespace xla
