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

#include "xla/service/gpu/transforms/move_copy_to_users.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/hlo_creation_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class MoveCopyToUsersVisitor : public DfsHloRewriteVisitor {
  // Turn copy->pad into pad->copy
  absl::Status HandlePad(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* c = hlo->mutable_operand(1);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_pad,
          MakePadHlo(copied, c, hlo->padding_config(), &hlo->metadata()));
      // MakePadHlo fails to propagate layout.
      *earlier_pad->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_pad, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->slice into slice->copy, as slice is layout-preserving.
  absl::Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeSliceHlo(copied, hlo->slice_starts(), hlo->slice_limits(),
                       hlo->slice_strides(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->dynamic-slice into dynamic-slice->copy, as dynamic-slice is
  // layout-preserving.
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeDynamicSliceHlo(
              copied,
              absl::Span<HloInstruction* const>(hlo->operands()).subspan(1),
              hlo->dynamic_slice_sizes(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->reduce_window into reduce_window->copy, as reduce_window is
  // layout-preserving.
  absl::Status HandleReduceWindow(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reduce_window,
          MakeReduceWindowHlo(copied, hlo->mutable_operand(1), hlo->window(),
                              hlo->called_computations()[0], &hlo->metadata()));
      *earlier_reduce_window->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_reduce_window, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  absl::Status HandleReduce(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    // Reductions can handle transposes, e.g. via column reduction.
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand) &&
        !hlo->shape().IsTuple()) {
      HloInstruction* new_reduce = hlo->AddInstruction(
          hlo->CloneWithNewOperands(hlo->shape(), {operand->mutable_operand(0),
                                                   hlo->mutable_operand(1)}));
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_reduce));
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitcastConvert(HloInstruction* hlo) override {
    return absl::OkStatus();
  }

  // Sink kCopy across elementwise unary.
  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kReducePrecision>(hlo)) {
      return absl::OkStatus();
    }
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_elementwise,
          MakeUnaryHlo(hlo->opcode(), copied, &hlo->metadata()));
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_elementwise, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Sink kCopy across reverse
  absl::Status HandleReverse(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reverse,
          MakeReverseHlo(copied, hlo->dimensions(), &hlo->metadata()));
      HloInstruction* later_copy = MakeCopyHlo(earlier_reverse, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Sink kCopy across convert.
  absl::Status HandleConvert(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      HloInstruction* earlier_convert = MakeConvertToHlo(
          copied, hlo->shape().element_type(), &hlo->metadata());
      HloInstruction* later_copy = MakeCopyHlo(earlier_convert, hlo->shape());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Sink kCopy across elementwise binary.
  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override {
    HloInstruction* a = hlo->mutable_operand(0);
    HloInstruction* b = hlo->mutable_operand(1);
    if (HloPredicateIsOp<HloOpcode::kCopy>(a) &&
        HloPredicateIsOp<HloOpcode::kCopy>(b)) {
      HloInstruction* copied_a = a->mutable_operand(0);
      HloInstruction* copied_b = b->mutable_operand(0);
      if (copied_a->shape() == copied_b->shape()) {
        HloInstruction* earlier_elementwise;
        if (HloPredicateIsOp<HloOpcode::kCompare>(hlo)) {
          TF_ASSIGN_OR_RETURN(
              earlier_elementwise,
              MakeCompareHlo(hlo->comparison_direction(), copied_a, copied_b,
                             &hlo->metadata()));
        } else {
          TF_ASSIGN_OR_RETURN(earlier_elementwise,
                              MakeBinaryHlo(hlo->opcode(), copied_a, copied_b,
                                            &hlo->metadata()));
        }
        HloInstruction* later_copy =
            MakeCopyHlo(earlier_elementwise, hlo->shape());
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
      }
    }
    return absl::OkStatus();
  }

  // Move copy across kConcat if it occurs on all operands.
  absl::Status HandleConcatenate(HloInstruction* hlo) override {
    const HloInstruction* first = hlo->operand(0);
    if (HloPredicateIsNotOp<HloOpcode::kCopy>(first)) {
      return absl::OkStatus();
    }
    const HloInstruction* inner_op = first->operand(0);
    const Layout& inner_op_layout = inner_op->shape().layout();

    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(hlo->operand_count());
    for (HloInstruction* op : hlo->mutable_operands()) {
      if (HloPredicateIsNotOp<HloOpcode::kCopy>(op) ||
          op->operand(0)->shape().layout() != inner_op_layout) {
        VLOG(3) << "Mismatch between " << op->ToString()
                << " and expected op layout " << inner_op_layout.ToString();
        return absl::OkStatus();
      }
      new_operands.push_back(op->mutable_operand(0));
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_concat,
        MakeConcatHlo(new_operands, hlo->concatenate_dimension()));
    *new_concat->mutable_shape()->mutable_layout() = inner_op_layout;

    HloInstruction* new_copy = MakeCopyHlo(new_concat, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_copy));
    return absl::OkStatus();
  }
};

}  // end namespace

absl::StatusOr<bool> MoveCopyToUsers::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return MoveCopyToUsersVisitor{}.RunOnModule(module, execution_threads);
}

}  // end namespace xla
