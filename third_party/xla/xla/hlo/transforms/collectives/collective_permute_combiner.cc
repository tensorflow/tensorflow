/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_combiner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/collective_permute_key.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single CollectivePermute op. All
// entries in to_combine must be CollectivePermute ops with the same
// source-target pairs and channel id.
absl::Status CombineCollectivePermutes(
    absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.size() < 2) {
    return absl::OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " CollectivePermute ops";

  HloComputation& computation = *to_combine.back()->parent();

  // Create a single bigger CollectivePermute of the operands of the smaller
  // CollectivePermutes.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> operand_shapes;
  const auto source_target_pairs = to_combine.at(0)->source_target_pairs();
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kCollectivePermute);
    TF_RET_CHECK(hlo->operand_count() == 1);
    TF_RET_CHECK(hlo->shape().IsArray());
    TF_RET_CHECK(hlo->source_target_pairs() == source_target_pairs);
    operands.push_back(hlo->operands().front());
    operand_shapes.push_back(&hlo->operands().front()->shape());
  }

  HloInstruction* combined;
  // CollectivePermute ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateCollectivePermute(
      ShapeUtil::MakeValidatedTupleShapeWithPtrs(operand_shapes).value(),
      operands, source_target_pairs, to_combine.front()->channel_id()));

  // Replace all the smaller CollectivePermutes with elements of the tuple
  // output of the single bigger CollectivePermute.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return absl::OkStatus();
}
}  // namespace

CollectivePermuteCombiner::CollectivePermuteCombiner(
    int64_t combine_threshold_in_bytes, int64_t combine_threshold_count)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count) {}

absl::StatusOr<bool> CollectivePermuteCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running CollectivePermuteCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip CollectivePermuteCombiner because the threshold is zero";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    auto key_fn = [](const HloInstruction* instruction)
        -> std::optional<CollectivePermuteKey> {
      if (instruction->opcode() != HloOpcode::kCollectivePermute) {
        return std::nullopt;
      }
      return GetCollectivePermuteKey(instruction);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<CollectivePermuteKey>(
            computation, key_fn, &CombineCollectivePermutes,
            combine_threshold_in_bytes_, combine_threshold_count_));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
