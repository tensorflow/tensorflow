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

#include "xla/backends/gpu/hlo/transforms/move_copy_to_users.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class MoveCopyToUsersVisitor : public DfsHloRewriteVisitor {
  // Turn copy->pad into pad->copy
  absl::Status HandlePad(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* c = hlo->mutable_operand(1);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * earlier_pad,
          MakePadHlo(copied, c, hlo->padding_config(), &hlo->metadata()));
      // MakePadHlo fails to propagate layout.
      *earlier_pad->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_pad, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->slice into slice->copy, as slice is layout-preserving.
  absl::Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeSliceHlo(copied, hlo->slice_starts(), hlo->slice_limits(),
                       hlo->slice_strides(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->dynamic-slice into dynamic-slice->copy, as dynamic-slice is
  // layout-preserving.
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * earlier_slice,
          MakeDynamicSliceHlo(
              copied,
              absl::Span<HloInstruction* const>(hlo->operands()).subspan(1),
              hlo->dynamic_slice_sizes(), &hlo->metadata()));
      *earlier_slice->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy = MakeCopyHlo(earlier_slice, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Turn copy->reduce_window into reduce_window->copy, as reduce_window is
  // layout-preserving.
  absl::Status HandleReduceWindow(HloInstruction* hlo) override {
    if (hlo->shape().IsTuple()) {
      // TODO: Handle variadic reductions.
      return absl::OkStatus();
    }

    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reduce_window,
          MakeReduceWindowHlo(copied, hlo->mutable_operand(1), hlo->window(),
                              hlo->called_computations()[0], &hlo->metadata()));
      *earlier_reduce_window->mutable_shape()->mutable_layout() =
          copied->shape().layout();
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_reduce_window, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
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
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_reduce));
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitcastConvert(HloInstruction* hlo) override {
    return absl::OkStatus();
  }

  // Sink kCopy across elementwise unary.
  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kReducePrecision>(hlo) ||
        HloPredicateIsOp<HloOpcode::kCopy>(hlo)) {
      return absl::OkStatus();
    }
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(HloInstruction * earlier_elementwise,
                       MakeUnaryHlo(hlo->opcode(), copied, &hlo->metadata()));
      HloInstruction* later_copy =
          MakeCopyHlo(earlier_elementwise, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Sink kCopy across reverse
  absl::Status HandleReverse(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * earlier_reverse,
          MakeReverseHlo(copied, hlo->dimensions(), &hlo->metadata()));
      HloInstruction* later_copy = MakeCopyHlo(earlier_reverse, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
    }
    return absl::OkStatus();
  }

  // Sink kCopy across convert.
  absl::Status HandleConvert(HloInstruction* hlo) override {
    if (hlo->user_count() == 1 &&
        HloPredicateIsOp<HloOpcode::kConvolution>(hlo->users()[0])) {
      // Don't sink copy across convert if the convert is consumed by conv,
      // it is likely convert can be fused with conv.
      return absl::OkStatus();
    }
    HloInstruction* operand = hlo->mutable_operand(0);
    if (HloPredicateIsOp<HloOpcode::kCopy>(operand)) {
      HloInstruction* copied = operand->mutable_operand(0);
      HloInstruction* earlier_convert = MakeConvertToHlo(
          copied, hlo->shape().element_type(), &hlo->metadata());
      HloInstruction* later_copy = MakeCopyHlo(earlier_convert, hlo->shape());
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
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
          ABSL_ASSIGN_OR_RETURN(earlier_elementwise,
                           MakeCompareHlo(hlo->comparison_direction(), copied_a,
                                          copied_b, &hlo->metadata()));
        } else {
          ABSL_ASSIGN_OR_RETURN(earlier_elementwise,
                           MakeBinaryHlo(hlo->opcode(), copied_a, copied_b,
                                         &hlo->metadata()));
        }
        HloInstruction* later_copy =
            MakeCopyHlo(earlier_elementwise, hlo->shape());
        ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, later_copy));
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

    ABSL_ASSIGN_OR_RETURN(HloInstruction * new_concat,
                     MakeConcatHlo(new_operands, hlo->concatenate_dimension()));
    *new_concat->mutable_shape()->mutable_layout() = inner_op_layout;

    HloInstruction* new_copy = MakeCopyHlo(new_concat, hlo->shape());
    ABSL_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_copy));
    return absl::OkStatus();
  }
};

// A use of an instruction as operand `operand_no` of `user`.
struct Use {
  HloInstruction* user;
  int64_t operand_no;
};

// Returns the get-tuple-element instructions that read element `index` of
// `tuple`.
std::vector<HloInstruction*> ElementReaders(const HloInstruction* tuple,
                                            int64_t index) {
  std::vector<HloInstruction*> readers;
  for (HloInstruction* user : tuple->users()) {
    if (HloPredicateIsOp<HloOpcode::kGetTupleElement>(user) &&
        user->tuple_index() == index) {
      readers.push_back(user);
    }
  }
  return readers;
}

bool AllUsersAreGetTupleElements(const HloInstruction* instruction) {
  return absl::c_all_of(instruction->users(), [](const HloInstruction* user) {
    return HloPredicateIsOp<HloOpcode::kGetTupleElement>(user);
  });
}

bool IsUnused(const HloInstruction* instruction) {
  return instruction->user_count() == 0 && !instruction->IsRoot();
}

// Returns true if `copy` only changes the dimension order of its operand: it
// has the same memory space, element size, tiling and so on, and static
// dimensions. Other copies, e.g. to another memory space or to a packed
// sub-byte layout, are not layout choices and must stay where they are.
bool ChangesOnlyDimensionOrder(const HloInstruction* copy) {
  const Shape& dst = copy->shape();
  const Shape& src = copy->operand(0)->shape();
  if (!dst.IsArray() || !src.IsArray() || !dst.is_static() ||
      !src.is_static() || !dst.has_layout() || !src.has_layout() ||
      dst.layout().minor_to_major() == src.layout().minor_to_major()) {
    return false;
  }
  Layout dst_with_src_order = dst.layout();
  dst_with_src_order.mutable_minor_to_major()->assign(
      src.layout().minor_to_major().begin(),
      src.layout().minor_to_major().end());
  return dst_with_src_order == src.layout();
}

std::optional<int64_t> TripCount(const HloInstruction* while_op) {
  absl::StatusOr<WhileLoopBackendConfig> config =
      while_op->backend_config<WhileLoopBackendConfig>();
  if (config.ok() && config->has_known_trip_count()) {
    return config->known_trip_count().n();
  }
  return ComputeWhileLoopTripCount(while_op);
}

// Describes how to move the copy that produces element `index` of the initial
// state of a while loop into the body of the loop.
struct CopyIntoWhileBody {
  HloInstruction* copy;
  int64_t index;
  // The uses of the element, in the body and after the loop, that need the
  // layout the copy produces. They are slices and dynamic-slices of the
  // element.
  std::vector<Use> slice_uses;
};

// The copy of an element of while loop state is only moved into the loop if
// the slices that are copied instead are, in total, not larger than this many
// times the element. The copy of a slice can be fused with the slice, so
// copying more than the element is accepted for not keeping the element in two
// layouts.
constexpr int64_t kMaxSliceBytesPerCopiedElementByte = 2;

// Adds the uses of `reader` to `uses` and returns the total size of the
// results of the uses, or nullopt if `reader` is used by anything but a slice
// or dynamic-slice of it. The use as operand `passed_on_as` of `passed_on_to`
// is skipped.
std::optional<int64_t> CollectSliceUses(HloInstruction* reader,
                                        const HloInstruction* passed_on_to,
                                        int64_t passed_on_as,
                                        std::vector<Use>& uses) {
  if (reader->IsRoot()) {
    return std::nullopt;
  }
  int64_t slice_bytes = 0;
  for (HloInstruction* user : reader->users()) {
    for (int64_t operand_no : user->OperandIndices(reader)) {
      if (user == passed_on_to && operand_no == passed_on_as) {
        continue;
      }
      if (operand_no != 0 ||
          HloPredicateIsNotOp<HloOpcode::kSlice, HloOpcode::kDynamicSlice>(
              user)) {
        return std::nullopt;
      }
      uses.push_back({user, operand_no});
      slice_bytes += ShapeUtil::ByteSizeOf(user->shape());
    }
  }
  return slice_bytes;
}

// Returns how to move the copy that produces element `index` of the initial
// state of `while_op` into the body of the loop, or nullopt if the copy can not
// be moved or moving it does not pay off.
//
// The copy can be moved if the loop only reads the element: the body passes
// it on unchanged, so the loop can carry the element in the layout it has
// before the copy. Moving pays off if the element is only read by slices, in
// the body and after the loop, and the slices of all iterations and the slices
// after the loop together are not larger than
// kMaxSliceBytesPerCopiedElementByte times the element. The copies of the
// slices then replace the copy of the element, and the element is not kept in
// two layouts while the loop runs.
std::optional<CopyIntoWhileBody> MatchCopyIntoWhileBody(
    HloInstruction* while_op, int64_t index) {
  HloInstruction* init = while_op->mutable_operand(0);
  HloInstruction* copy = init->mutable_operand(index);
  if (HloPredicateIsNotOp<HloOpcode::kCopy>(copy) ||
      !ChangesOnlyDimensionOrder(copy)) {
    return std::nullopt;
  }

  // The state must only be read element by element, so that the layout of one
  // element can be changed.
  HloComputation* body = while_op->while_body();
  const HloInstruction* body_parameter = body->parameter_instruction(0);
  const HloInstruction* condition_parameter =
      while_op->while_condition()->parameter_instruction(0);
  if (while_op->IsRoot() || !AllUsersAreGetTupleElements(while_op) ||
      !AllUsersAreGetTupleElements(body_parameter) ||
      !AllUsersAreGetTupleElements(condition_parameter) ||
      !absl::c_all_of(ElementReaders(condition_parameter, index), IsUnused)) {
    return std::nullopt;
  }

  // The body must pass the element on unchanged. Nothing in the body may read
  // the state the body returns, as such a reader would see the element in a
  // different layout.
  const HloInstruction* body_root = body->root_instruction();
  if (HloPredicateIsNotOp<HloOpcode::kTuple>(body_root) ||
      body_root->user_count() != 0) {
    return std::nullopt;
  }
  const HloInstruction* passed_on = body_root->operand(index);
  if (HloPredicateIsNotOp<HloOpcode::kGetTupleElement>(passed_on) ||
      passed_on->operand(0) != body_parameter ||
      passed_on->tuple_index() != index) {
    return std::nullopt;
  }

  // Apart from that, the element must only be read by slices.
  CopyIntoWhileBody result{copy, index, {}};
  int64_t slice_bytes_per_iteration = 0;
  for (HloInstruction* reader : ElementReaders(body_parameter, index)) {
    std::optional<int64_t> slice_bytes =
        CollectSliceUses(reader, body_root, index, result.slice_uses);
    if (!slice_bytes.has_value()) {
      return std::nullopt;
    }
    slice_bytes_per_iteration += *slice_bytes;
  }
  int64_t total_slice_bytes = 0;
  for (HloInstruction* reader : ElementReaders(while_op, index)) {
    std::optional<int64_t> slice_bytes =
        CollectSliceUses(reader, nullptr, -1, result.slice_uses);
    if (!slice_bytes.has_value()) {
      return std::nullopt;
    }
    total_slice_bytes += *slice_bytes;
  }
  const int64_t max_slice_bytes =
      kMaxSliceBytesPerCopiedElementByte * ShapeUtil::ByteSizeOf(copy->shape());
  if (slice_bytes_per_iteration > 0) {
    std::optional<int64_t> trip_count = TripCount(while_op);
    // The trip count is checked on its own first, so that the size of the
    // slices of all iterations does not overflow for a large trip count.
    if (!trip_count.has_value() || *trip_count < 0 ||
        *trip_count > max_slice_bytes / slice_bytes_per_iteration) {
      return std::nullopt;
    }
    total_slice_bytes += *trip_count * slice_bytes_per_iteration;
  }
  if (total_slice_bytes > max_slice_bytes) {
    return std::nullopt;
  }
  return result;
}

// Moves the copy described by `match` into the body of `while_op`: the loop
// carries the element in the layout it has before the copy, and each slice of
// the element, in the body or after the loop, reads a copy of the element.
// MoveCopyToUsersVisitor then moves that copy below the slice.
absl::Status MoveCopyIntoWhileBody(HloInstruction* while_op,
                                   const CopyIntoWhileBody& match) {
  const int64_t index = match.index;
  HloInstruction* init = while_op->mutable_operand(0);
  HloInstruction* copied = match.copy->mutable_operand(0);
  HloComputation* body = while_op->while_body();
  const Shape copy_shape = match.copy->shape();
  const Shape copied_shape = copied->shape();

  // The slices read the element in the layout of the copy.
  for (const Use& use : match.slice_uses) {
    HloInstruction* reader = use.user->mutable_operand(use.operand_no);
    ABSL_RETURN_IF_ERROR(use.user->ReplaceOperandWith(
        use.operand_no, MakeCopyHlo(reader, copy_shape)));
  }

  // Everything that carries the element has the layout of the copied array.
  ABSL_RETURN_IF_ERROR(init->ReplaceOperandWithDifferentShape(index, copied));
  for (HloInstruction* state :
       {init, while_op, body->parameter_instruction(0),
        body->root_instruction(),
        while_op->while_condition()->parameter_instruction(0)}) {
    *state->mutable_shape()->mutable_tuple_shapes(index) = copied_shape;
    for (HloInstruction* reader : ElementReaders(state, index)) {
      *reader->mutable_shape() = copied_shape;
    }
  }

  if (IsUnused(match.copy)) {
    ABSL_RETURN_IF_ERROR(match.copy->SafelyDropAllControlDependencies());
    ABSL_RETURN_IF_ERROR(match.copy->parent()->RemoveInstruction(match.copy));
  }
  return absl::OkStatus();
}

// Moves copies that produce an element of the initial state of a while loop
// into the body of the loop, see MatchCopyIntoWhileBody.
absl::StatusOr<bool> MoveCopiesIntoWhileBodies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> while_ops;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (HloPredicateIsOp<HloOpcode::kWhile>(instruction)) {
        while_ops.push_back(instruction);
      }
    }
  }

  bool changed = false;
  for (HloInstruction* while_op : while_ops) {
    // The initial state must be a tuple that is only used by the loop, and the
    // body and the condition must only be called by the loop, so that the
    // layout of an element can be changed in all of them.
    const HloInstruction* init = while_op->operand(0);
    if (HloPredicateIsNotOp<HloOpcode::kTuple>(init) ||
        init->user_count() != 1 ||
        while_op->while_body()->caller_instructions().size() != 1 ||
        while_op->while_condition()->caller_instructions().size() != 1) {
      continue;
    }
    for (int64_t index = 0; index < init->operand_count(); ++index) {
      std::optional<CopyIntoWhileBody> match =
          MatchCopyIntoWhileBody(while_op, index);
      if (match.has_value()) {
        ABSL_RETURN_IF_ERROR(MoveCopyIntoWhileBody(while_op, *match));
        changed = true;
      }
    }
  }
  return changed;
}

}  // end namespace

absl::StatusOr<bool> MoveCopyToUsers::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_ASSIGN_OR_RETURN(bool moved_into_while_bodies,
                   MoveCopiesIntoWhileBodies(module, execution_threads));
  ABSL_ASSIGN_OR_RETURN(bool moved_to_users, MoveCopyToUsersVisitor{}.RunOnModule(
                                            module, execution_threads));
  return moved_into_while_bodies || moved_to_users;
}

}  // end namespace xla
