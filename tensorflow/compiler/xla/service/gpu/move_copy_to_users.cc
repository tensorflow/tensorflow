/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/move_copy_to_users.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class MoveCopyToUsersVisitor : public DfsHloRewriteVisitor {
  // Turn copy->pad into pad->copy
  Status HandlePad(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* c = hlo->mutable_operand(1);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_pad,
          MakePadHlo(copied, c, hlo->padding_config(), &hlo->metadata()));
      HloInstruction* later_copy = MakeCopyHlo(earlier_pad, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return OkStatus();
  }

  // Sink kCopy across elementwise unary.
  Status HandleElementwiseUnary(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_elementwise,
          MakeUnaryHlo(hlo->opcode(), copied, &hlo->metadata()));
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_elementwise, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return OkStatus();
  }

  // Sink kCopy across reverse
  Status HandleReverse(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reverse,
          MakeReverseHlo(copied, hlo->dimensions(), &hlo->metadata()));
      HloInstruction* later_copy = MakeCopyHlo(earlier_reverse, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return OkStatus();
  }

  // Sink kCopy across convert.
  Status HandleConvert(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (operand->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied = operand->mutable_operand(0);
      HloInstruction* earlier_convert = MakeConvertToHlo(
          copied, hlo->shape().element_type(), &hlo->metadata());
      HloInstruction* later_copy = MakeCopyHlo(earlier_convert, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return OkStatus();
  }

  // Sink kCopy across elementwise binary.
  Status HandleElementwiseBinary(HloInstruction* hlo) override {
    HloInstruction* a = hlo->mutable_operand(0);
    HloInstruction* b = hlo->mutable_operand(1);
    if (a->opcode() == HloOpcode::kCopy && b->opcode() == HloOpcode::kCopy) {
      HloInstruction* copied_a = a->mutable_operand(0);
      HloInstruction* copied_b = b->mutable_operand(0);
      if (copied_a->shape() == copied_b->shape()) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * earlier_elementwise,
            MakeBinaryHlo(hlo->opcode(), copied_a, copied_b, &hlo->metadata()));
        HloInstruction* later_copy =
            MakeCopyHlo(earlier_elementwise, hlo->shape());
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
      }
    }
    return OkStatus();
  }
};

}  // end namespace

StatusOr<bool> MoveCopyToUsers::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return MoveCopyToUsersVisitor{}.RunOnModule(module, execution_threads);
}

}  // end namespace xla
