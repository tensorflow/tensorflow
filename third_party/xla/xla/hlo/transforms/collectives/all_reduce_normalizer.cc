/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/all_reduce_normalizer.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Split a tupled all-reduce into individual all-reduces.
std::vector<HloInstruction*> SplitAllReduces(HloAllReduceInstruction* ar) {
  std::vector<HloInstruction*> separate_all_reduces;
  separate_all_reduces.reserve(ar->operand_count());
  for (int64_t i = 0; i < ar->operand_count(); ++i) {
    HloInstruction* separate_all_reduce =
        ar->parent()->AddInstruction(HloInstruction::CreateAllReduce(
            ar->shape().tuple_shapes(i), {ar->mutable_operand(i)},
            ar->to_apply(), ar->device_list(), ar->constrain_layout(),
            hlo_query::NextChannelId(*ar->GetModule()),
            ar->use_global_device_ids()));
    separate_all_reduces.push_back(separate_all_reduce);
  }
  return separate_all_reduces;
}

int64_t FindLeadingNonOneDimension(const Shape& shape) {
  int64_t i;
  for (i = shape.rank() - 1; i > 0; --i) {
    if (shape.dimensions_minor(i) > 1) {
      break;
    }
  }
  return shape.layout().minor_to_major(i);
}

void MaybeUpdateDefaultLayout(Shape& shape) {
  if (!shape.has_layout()) {
    *shape.mutable_layout() = LayoutUtil::GetDefaultLayoutForShape(shape);
  }
}

// Convert a single-operand all-reduce to all-to-all + reduce + all-gather.
absl::StatusOr<bool> NormalizeSingleOperandAllReduce(
    HloInstruction* hlo,
    std::function<bool(const HloInstruction*)> is_supported_all_reduce) {
  if (is_supported_all_reduce(hlo)) {
    return false;
  }
  HloComputation* computation = hlo->parent();
  HloAllReduceInstruction* ar = Cast<HloAllReduceInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(auto replica_group_count_and_size,
                      GetReplicaGroupCountAndSize(ar));
  if (!replica_group_count_and_size.has_value()) {
    return absl::InvalidArgumentError("Unsupported all-reduce with : " +
                                      ar->ToString());
  }
  const int64_t replica_group_size = replica_group_count_and_size->second;
  Shape ar_shape = ar->shape();
  MaybeUpdateDefaultLayout(ar_shape);
  std::vector<hlo_sharding_util::FormattingStep> formatting_steps;
  if (ar_shape.rank() == 0) {
    Shape scalar_ar_shape = ar_shape;
    ar_shape = ShapeUtil::MakeShape(ar_shape.element_type(), {1});
    MaybeUpdateDefaultLayout(scalar_ar_shape);
    formatting_steps.push_back(hlo_sharding_util::FormattingStep{
        .input_shape = scalar_ar_shape,
        .output_shape = ar_shape,
        .formatting_opcode = HloOpcode::kBitcast});
  }
  const int64_t leading_non_one_dim = FindLeadingNonOneDimension(ar_shape);
  // 1. Pad the leading non-one dimension to the nearest multiple of replica
  // group size, so that it's divisible, and reshape it to have a leading
  // dimension of replica group size.
  Shape padded_shape = ar_shape;
  padded_shape.set_dimensions(
      leading_non_one_dim,
      RoundUpTo(ar_shape.dimensions(leading_non_one_dim), replica_group_size));
  MaybeUpdateDefaultLayout(padded_shape);
  std::optional<ReductionKind> kind =
      MatchReductionInstruction(ar->to_apply()->root_instruction());
  if (!kind) {
    return absl::InvalidArgumentError(
        "Unsupported reduction type with : " + ar->ToString() + "\n" +
        ar->to_apply()->ToString());
  }
  std::optional<Literal> reduction_identity =
      GetReductionIdentity(*kind, ar->shape().element_type());
  if (!reduction_identity) {
    return absl::InvalidArgumentError(
        "Unsupported reduction identity with : " + ar->ToString() + "\n" +
        ar->to_apply()->ToString());
  }
  HloInstruction* identity = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(reduction_identity.value())));
  formatting_steps.push_back(
      hlo_sharding_util::FormattingStep{.input_shape = ar_shape,
                                        .output_shape = padded_shape,
                                        .formatting_opcode = HloOpcode::kPad,
                                        .padding_value = identity});
  const int64_t fake_dim = 0;
  Shape reshape_shape = padded_shape;
  reshape_shape.set_dimensions(
      leading_non_one_dim,
      reshape_shape.dimensions(leading_non_one_dim) / replica_group_size);
  reshape_shape.add_dimensions(replica_group_size, /*index=*/fake_dim);
  MaybeUpdateDefaultLayout(reshape_shape);

  formatting_steps.push_back(hlo_sharding_util::FormattingStep{
      .input_shape = padded_shape,
      .output_shape = reshape_shape,
      .formatting_opcode = HloOpcode::kReshape});

  HloInstruction* formatted = hlo_sharding_util::FormatShape(
      ar->mutable_operand(0), formatting_steps, computation);

  // 2. Create an all-to-all on the leading non-one dimension.
  HloInstruction* all_to_all =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          reshape_shape, {formatted}, ar->device_list(), ar->constrain_layout(),
          hlo_query::NextChannelId(*ar->GetModule()),
          /*split_dimension=*/fake_dim));
  // 3. Do a local reduce and an all-gather
  Shape reduce_shape = reshape_shape;
  reduce_shape.DeleteDimension(fake_dim);
  HloInstruction* reduce =
      computation->AddInstruction(HloInstruction::CreateReduce(
          reduce_shape, all_to_all, identity,
          /*dimensions_to_reduce=*/{fake_dim}, ar->to_apply()));
  Shape ag_operand_shape = reshape_shape;
  ag_operand_shape.set_dimensions(fake_dim, 1);
  HloInstruction* ag_operand = computation->AddInstruction(
      HloInstruction::CreateReshape(ag_operand_shape, reduce));
  HloInstruction* ag =
      computation->AddInstruction(HloInstruction::CreateAllGather(
          reshape_shape, {ag_operand}, /*all_gather_dimension=*/fake_dim,
          ar->device_list(), ar->constrain_layout(),
          hlo_query::NextChannelId(*ar->GetModule()),
          ar->use_global_device_ids()));
  // 4. Reshape and slice back to the original shape.
  HloInstruction* unformatted =
      hlo_sharding_util::ReverseFormatShape(ag, formatting_steps, computation);

  unformatted->set_metadata(ar->metadata());
  TF_RETURN_IF_ERROR(ar->ReplaceAllUsesWith(unformatted));
  TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
  return true;
}

}  // namespace

absl::StatusOr<bool> AllReduceNormalizer::NormalizeAllReduce(
    HloInstruction* hlo) {
  HloComputation* computation = hlo->parent();
  HloAllReduceInstruction* ar = Cast<HloAllReduceInstruction>(hlo);
  if (ar->operand_count() > 1) {
    // Tupled all-reduce's counterpart all-gathers might not be combinable, so
    // we split them into individual all-reduces first and then convert to
    // all-gathers and local reduces, and run all-reduce combiner after this if
    // we want to combine them back.
    bool changed = false;
    std::vector<HloInstruction*> separate_all_reduces = SplitAllReduces(ar);
    for (HloInstruction* separate_all_reduce : separate_all_reduces) {
      TF_ASSIGN_OR_RETURN(bool converted,
                          NormalizeSingleOperandAllReduce(
                              separate_all_reduce, is_supported_all_reduce_));
      changed |= converted;
    }
    if (changed) {
      TF_RETURN_IF_ERROR(ar->ReplaceAllUsesWith(computation->AddInstruction(
          HloInstruction::CreateTuple(separate_all_reduces))));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
    } else {
      for (HloInstruction* separate_all_reduce : separate_all_reduces) {
        TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
            separate_all_reduce));
      }
    }
    return changed;
  } else {
    return NormalizeSingleOperandAllReduce(hlo, is_supported_all_reduce_);
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AllReduceNormalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kAllReduce) {
        TF_ASSIGN_OR_RETURN(bool inst_changed, NormalizeAllReduce(inst));
        changed |= inst_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla
