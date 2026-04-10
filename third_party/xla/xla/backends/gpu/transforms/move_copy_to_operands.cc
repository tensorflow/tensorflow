/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/move_copy_to_operands.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

// Maps a pair of (original operand, target layout) to the new operation
// created with that layout. This is used to deduplicate operations when
// multiple copies with the same layout are moved over the same operand.
using MovedOpsMap =
    absl::flat_hash_map<std::pair<HloInstruction*, Layout>, HloInstruction*>;

struct MoveResult {
  HloInstruction* new_op;
  std::vector<HloInstruction*> new_copies;
};

absl::StatusOr<std::optional<MoveResult>> DoBroadcast(
    HloInstruction* copy, HloInstruction* broadcast,
    HloComputation* computation) {
  const Layout& target_layout = copy->shape().layout();
  const Shape& operand_shape = broadcast->operand(0)->shape();
  int64_t operand_rank = operand_shape.dimensions().size();

  std::vector<int64_t> new_layout_dimensions;
  new_layout_dimensions.reserve(operand_rank);

  auto broadcast_dims = broadcast->dimensions();
  for (int64_t dim : target_layout.minor_to_major()) {
    auto it = absl::c_find(broadcast_dims, dim);
    if (it != broadcast_dims.end()) {
      new_layout_dimensions.push_back(
          std::distance(broadcast_dims.begin(), it));
    }
  }

  Layout new_layout = LayoutUtil::MakeLayout(new_layout_dimensions);

  HloInstruction* operand = broadcast->mutable_operand(0);
  std::vector<HloInstruction*> new_copies;
  if (operand_shape.layout() != new_layout) {
    Shape new_copy_shape = operand_shape;
    *new_copy_shape.mutable_layout() = new_layout;
    operand = MakeCopyHlo(operand, new_copy_shape);
    new_copies.push_back(operand);
  }

  std::unique_ptr<HloInstruction> new_broadcast =
      broadcast->CloneWithNewOperands(copy->shape(), {operand});

  HloInstruction* new_op =
      computation->AddInstruction(std::move(new_broadcast));

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_op,
                                        /*preserve_sharding=*/false));
  if (changed) {
    return MoveResult{new_op, std::move(new_copies)};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<MoveResult>> DoConcatenate(
    HloInstruction* copy, HloInstruction* concat, HloComputation* computation) {
  const HloInstruction* first = concat->operand(0);
  const Layout& first_layout = first->shape().layout();

  for (const HloInstruction* op : concat->operands()) {
    if (op->shape().layout() != first_layout) {
      return std::nullopt;
    }
  }

  const Layout& target_layout = copy->shape().layout();

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(concat->operand_count());
  for (HloInstruction* op : concat->mutable_operands()) {
    Shape new_copy_shape = op->shape();
    *new_copy_shape.mutable_layout() = target_layout;
    new_operands.push_back(MakeCopyHlo(op, new_copy_shape));
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_concat,
      MakeConcatHlo(new_operands, concat->concatenate_dimension()));
  *new_concat->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_concat,
                                        /*preserve_sharding=*/false));
  if (changed) {
    return MoveResult{new_concat, new_operands};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<MoveResult>> DoDynamicSlice(
    HloInstruction* copy, HloInstruction* ds, HloComputation* computation) {
  HloInstruction* x = ds->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_ds,
      MakeDynamicSliceHlo(
          new_copy,
          absl::Span<HloInstruction* const>(ds->operands()).subspan(1),
          ds->dynamic_slice_sizes(), &ds->metadata()));
  *new_ds->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_ds,
                                        /*preserve_sharding=*/false));
  if (changed) {
    std::vector<HloInstruction*> new_copies = {new_copy};
    return MoveResult{new_ds, new_copies};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<MoveResult>> DoPad(HloInstruction* copy,
                                                HloInstruction* pad,
                                                HloComputation* computation) {
  HloInstruction* x = pad->mutable_operand(0);
  HloInstruction* c = pad->mutable_operand(1);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_pad,
      MakePadHlo(new_copy, c, pad->padding_config(), &pad->metadata()));
  *new_pad->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_pad,
                                        /*preserve_sharding=*/false));
  if (changed) {
    return MoveResult{new_pad, {new_copy}};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<MoveResult>> DoReduceWindow(
    HloInstruction* copy, HloInstruction* rw, HloComputation* computation) {
  if (rw->shape().IsTuple()) {
    return std::nullopt;
  }
  HloInstruction* x = rw->mutable_operand(0);
  HloInstruction* init = rw->mutable_operand(1);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_rw,
      MakeReduceWindowHlo(new_copy, init, rw->window(),
                          rw->called_computations()[0], &rw->metadata()));
  *new_rw->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_rw,
                                        /*preserve_sharding=*/false));
  if (changed) {
    return MoveResult{new_rw, {new_copy}};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<MoveResult>> DoScatter(
    HloInstruction* copy, HloInstruction* hlo, HloComputation* computation) {
  // Variadic scatter is not supported here.
  if (hlo->operand_count() != 3) {
    return std::nullopt;
  }

  const Layout& target_layout = copy->shape().layout();
  int64_t result_rank = hlo->shape().dimensions().size();

  std::vector<HloInstruction*> new_operands;
  std::vector<HloInstruction*> new_copies;
  new_operands.reserve(hlo->operand_count());

  auto maybe_copy_operand = [&](HloInstruction* operand) {
    if (operand->shape().dimensions().size() == result_rank &&
        operand->shape().layout() != target_layout) {
      Shape new_shape = operand->shape();
      *new_shape.mutable_layout() = target_layout;
      HloInstruction* c = MakeCopyHlo(operand, new_shape);
      new_copies.push_back(c);
      return c;
    }
    return operand;
  };

  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    new_operands.push_back(maybe_copy_operand(hlo->mutable_operand(i)));
  }

  std::unique_ptr<HloInstruction> new_scatter =
      hlo->CloneWithNewOperands(copy->shape(), new_operands);

  HloInstruction* new_scatter_ptr = new_scatter.get();
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(copy, std::move(new_scatter)));
  return MoveResult{new_scatter_ptr, new_copies};
}

absl::StatusOr<std::optional<MoveResult>> DoSimpleOp(
    HloInstruction* copy, HloInstruction* hlo, HloComputation* computation) {
  const Layout& target_layout = copy->shape().layout();

  std::vector<HloInstruction*> new_operands;
  std::vector<HloInstruction*> new_copies;
  new_operands.reserve(hlo->operand_count());
  new_copies.reserve(hlo->operand_count());
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    HloInstruction* x = hlo->mutable_operand(i);

    Shape new_copy_shape = x->shape();
    *new_copy_shape.mutable_layout() = target_layout;
    HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);
    new_operands.push_back(new_copy);
    new_copies.push_back(new_copy);
  }

  std::unique_ptr<HloInstruction> new_hlo =
      hlo->CloneWithNewOperands(copy->shape(), new_operands);

  HloInstruction* new_hlo_ptr = new_hlo.get();
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(copy, std::move(new_hlo)));

  return MoveResult{new_hlo_ptr, new_copies};
}

absl::StatusOr<std::optional<MoveResult>> DoSlice(HloInstruction* copy,
                                                  HloInstruction* slice,
                                                  HloComputation* computation) {
  HloInstruction* x = slice->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_slice,
      MakeSliceHlo(new_copy, slice->slice_starts(), slice->slice_limits(),
                   slice->slice_strides(), &slice->metadata()));
  *new_slice->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_slice,
                                        /*preserve_sharding=*/false));
  if (changed) {
    return MoveResult{new_slice, {new_copy}};
  }
  return std::nullopt;
}

// Tries to fold a copy into a parameter by changing the parameter's layout.
// This is only done if all users of the parameter are copies with the same
// layout as 'copy', and xla_pjrt_allow_auto_layout_in_hlo is enabled.
absl::StatusOr<bool> TryFoldParameter(HloInstruction* copy,
                                      HloInstruction* parameter,
                                      HloModule* module) {
  if (!module->config().debug_options().xla_pjrt_allow_auto_layout_in_hlo()) {
    return false;
  }
  bool all_users_agree = true;
  for (const HloInstruction* user : parameter->users()) {
    if (user->opcode() != HloOpcode::kCopy ||
        user->shape().layout() != copy->shape().layout()) {
      all_users_agree = false;
      break;
    }
  }
  if (!all_users_agree) {
    return false;
  }
  *parameter->mutable_shape()->mutable_layout() = copy->shape().layout();
  TF_RETURN_IF_ERROR(
      module->mutable_entry_computation_layout()
          ->mutable_parameter_layout(parameter->parameter_number())
          ->CopyLayoutFromShape(copy->shape()));
  TF_ASSIGN_OR_RETURN(bool local_changed,
                      module->entry_computation()->ReplaceInstruction(
                          copy, parameter, /*preserve_sharding=*/false));
  return local_changed;
}

absl::StatusOr<std::optional<MoveResult>> MoveCopyOverOp(
    HloInstruction* copy, HloInstruction* operand,
    HloComputation* computation) {
  absl::StatusOr<std::optional<MoveResult>> result = std::nullopt;
  switch (operand->opcode()) {
    case HloOpcode::kBitcastConvert:
      if (ShapeUtil::ElementsIn(operand->shape()) !=
          ShapeUtil::ElementsIn(operand->operand(0)->shape())) {
        return std::nullopt;
      }
      result = DoSimpleOp(copy, operand, computation);
      break;
    case HloOpcode::kBroadcast:
      result = DoBroadcast(copy, operand, computation);
      break;
    case HloOpcode::kConcatenate:
      result = DoConcatenate(copy, operand, computation);
      break;
    case HloOpcode::kConstant:
      return std::nullopt;
    case HloOpcode::kDynamicSlice:
      result = DoDynamicSlice(copy, operand, computation);
      break;
    case HloOpcode::kPad:
      result = DoPad(copy, operand, computation);
      break;
    case HloOpcode::kReduceWindow:
      result = DoReduceWindow(copy, operand, computation);
      break;
    case HloOpcode::kReverse:
      result = DoSimpleOp(copy, operand, computation);
      break;
    case HloOpcode::kScatter:
      result = DoScatter(copy, operand, computation);
      break;
    case HloOpcode::kSlice:
      result = DoSlice(copy, operand, computation);
      break;
    default:
      if (operand->IsElementwise()) {
        result = DoSimpleOp(copy, operand, computation);
      }
      break;
  }
  return result;
}

}  // end namespace

absl::StatusOr<bool> MoveCopyToOperands::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> post_order =
        computation->MakeInstructionPostOrder();

    // Use a worklist initialized with post-order traversal to process
    // instructions. This allows us to handle "multi-hop" copy movements (e.g.,
    // moving a copy over a slice and then over a pad) in a single pass, as
    // newly created copies are added back to the worklist.
    std::vector<HloInstruction*> worklist = post_order;
    MovedOpsMap moved_ops;

    while (!worklist.empty()) {
      HloInstruction* hlo = worklist.back();
      worklist.pop_back();

      if (computation->IsMarkedAsDead(hlo)) {
        continue;
      }

      if (hlo->opcode() != HloOpcode::kCopy) {
        continue;
      }

      HloInstruction* operand = hlo->mutable_operand(0);
      if (hlo->shape().layout() == operand->shape().layout()) {
        TF_ASSIGN_OR_RETURN(bool local_changed,
                            computation->ReplaceInstruction(
                                hlo, operand, /*preserve_sharding=*/false));
        changed |= local_changed;
        continue;
      }

      // Deduplication: If we have already moved a copy with this target layout
      // over this operand, reuse the previously created operation. This avoids
      // creating duplicate identical instructions.
      const Layout& target_layout = hlo->shape().layout();
      auto it = moved_ops.find({operand, target_layout});
      if (it != moved_ops.end()) {
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, it->second));
        changed = true;
        continue;
      }

      // Parameter folding logic
      if (operand->opcode() == HloOpcode::kParameter &&
          computation == module->entry_computation()) {
        TF_ASSIGN_OR_RETURN(bool folded,
                            TryFoldParameter(hlo, operand, module));
        if (folded) {
          changed = true;
          continue;
        }
      }

      TF_ASSIGN_OR_RETURN(std::optional<MoveResult> result,
                          MoveCopyOverOp(hlo, operand, computation));
      if (result.has_value()) {
        moved_ops[{operand, target_layout}] = result->new_op;
        for (HloInstruction* new_copy : result->new_copies) {
          worklist.push_back(new_copy);
        }
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla::gpu
