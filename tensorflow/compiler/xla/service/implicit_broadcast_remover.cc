/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/implicit_broadcast_remover.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

// Visitor for removing implicit broadcasts.
class ImplicitBroadcastVisitor : public DfsHloVisitorWithDefault {
 public:
  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Status::OK();
  }

  Status HandleElementwiseBinary(HloInstruction* hlo) override {
    return ReplaceImplicitBroadcastOperands(hlo);
  }

  Status HandleClamp(HloInstruction* hlo) override {
    // Clamp is the only element-wise ternary operation.
    return ReplaceImplicitBroadcastOperands(hlo);
  }

  // Returns whether any modification has been made to any visited instruction.
  bool changed() const { return changed_; }

 private:
  // Iterates through the operands of 'hlo' and replace any operands which are
  // implicitly broadcast with the equivalent sequence of broadcast and reshape
  // instructions. An operand is considered to be implicitly broadcast if the
  // operand shape does have the same dimensions as the shape of 'hlo'.
  Status ReplaceImplicitBroadcastOperands(HloInstruction* hlo) {
    auto fadd = [hlo](std::unique_ptr<HloInstruction> x) {
      return hlo->parent()->AddInstruction(std::move(x));
    };
    std::vector<HloInstruction*> operands;
    bool operands_changed = false;
    for (int i = 0; i < hlo->operand_count(); ++i) {
      HloInstruction* operand = hlo->mutable_operand(i);
      if (!ShapeUtil::SameDimensions(hlo->shape(), operand->shape())) {
        HloInstruction* new_operand = hlo->parent()->AddInstruction(
            HloInstruction::CreateBroadcastSequence(hlo->shape(), operand,
                                                    fadd));
        operands.push_back(new_operand);
        operands_changed = true;
      } else {
        operands.push_back(operand);
      }
    }
    if (operands_changed) {
      // Create a new HLO instruction because the HloInstruction::Replace*
      // methods check that the shape does not change with the replacement.
      HloInstruction* new_hlo = hlo->parent()->AddInstruction(
          hlo->CloneWithNewOperands(hlo->shape(), operands));
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_hlo));
      changed_ = true;
    }
    return Status::OK();
  }

  bool changed_ = false;
};

}  // namespace

StatusOr<bool> ImplicitBroadcastRemover::Run(HloModule* module) {
  VLOG(1) << "Removing implicit broadcast from module " << module->name();
  XLA_VLOG_LINES(2,
                 "Before removing implicit broadcasts:\n" + module->ToString());

  ImplicitBroadcastVisitor visitor;
  for (HloComputation* computation : module->computations()) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }

  if (visitor.changed()) {
    // HLO instructions with implicitly broadcast operands are cloned and left
    // for dead. Remove them.
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }

  XLA_VLOG_LINES(2,
                 "After removing implicit broadcasts:\n" + module->ToString());

  return visitor.changed();
}

}  // namespace xla
