/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the apecific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/transforms/simplifiers/all_gather_permuted_ds_simplifier.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::Status
AllGatherDynamicSlicePermutedOffsetSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice_hlo) {
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(dynamic_slice_hlo);
  HloInstruction* operand = dynamic_slice->mutable_operand(0);

  // Check if the operand is a reshape or all-gather instruction
  namespace m = match;
  HloInstruction* all_gather_hlo;
  if (!Match(operand, m::AllGather(&all_gather_hlo)) &&
      !Match(operand, m::Reshape(m::AllGather(&all_gather_hlo)))) {
    return absl::OkStatus();
  }

  HloAllGatherInstruction* all_gather =
      Cast<HloAllGatherInstruction>(all_gather_hlo);

  // Shape check: dynamic-slice shape should match the all-gather operand shape.
  if (!ShapeUtil::Compatible(dynamic_slice->shape(),
                             all_gather->operand(0)->shape())) {
    return absl::OkStatus();
  }

  const HloModuleConfig& config = dynamic_slice->GetModule()->config();
  std::optional<AllGatherDynamicSliceMatchSpec> offset_spec =
      MatchPermutedSliceAndPartitionOffset(
          all_gather, config.num_partitions(), config.replica_count(),
          HloPredicateIsOp<HloOpcode::kPartitionId>,
          /*allow_multiple_users=*/false);

  if (offset_spec.has_value() && !offset_spec->permutation_pairs.empty()) {
    // Remove the duplicated pairs as no collective permute is needed.
    offset_spec->permutation_pairs.erase(
        std::remove_if(offset_spec->permutation_pairs.begin(),
                       offset_spec->permutation_pairs.end(),
                       [](const std::pair<int64_t, int64_t>& pair) {
                         return pair.first == pair.second;
                       }),
        offset_spec->permutation_pairs.end());
    // Replace the pattern with a collective permute.
    HloInstruction* cp =
        dynamic_slice->AddInstruction(HloInstruction::CreateCollectivePermute(
            dynamic_slice->shape(), all_gather->mutable_operand(0),
            offset_spec->permutation_pairs, all_gather->channel_id()));
    return ReplaceInstruction(dynamic_slice, cp);
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> AllGatherDynamicSlicePermutedOffsetSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    AllGatherDynamicSlicePermutedOffsetSimplifierVisitor visitor;
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla
