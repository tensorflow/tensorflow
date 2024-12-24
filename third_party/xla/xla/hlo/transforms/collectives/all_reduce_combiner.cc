/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/all_reduce_combiner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single AllReduce op. All
// entries in to_combine must be AllReduce ops with exactly one operand
// and the same reduction operation.
absl::Status CombineAllReduces(absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.size() < 2) {
    return absl::OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " CRS ops";

  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  const HloOpcode type = reduction->root_instruction()->opcode();

  // Create a single bigger AllReduce of the operands of the smaller
  // AllReduces.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> operand_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllReduce);
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(hlo->to_apply() == reduction ||
                 (hlo->to_apply()->instruction_count() == 3 &&
                  hlo->to_apply()->num_parameters() == 2 &&
                  hlo->to_apply()->root_instruction()->opcode() == type));
    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      operand_shapes.push_back(&operand->shape());
    }
  }

  HloInstruction* combined;
  // AllReduce ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes), operands, reduction,
      to_combine.front()->device_list(),
      /*constrain_layout=*/false, to_combine.front()->channel_id(),
      Cast<HloAllReduceInstruction>(to_combine.front())
          ->use_global_device_ids()));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  combined->set_sharding(
      hlo_sharding_util::CreateTupleSharding(combined->shape(), to_combine));
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller AllReduces with elements of the tuple output
  // of the single bigger AllReduce.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return absl::OkStatus();
}
}  // namespace

/*static*/ std::string AllReduceCombiner::GetGroupKeyExtraArgs(
    AllReduceCombiner::GroupKey& key) {
  return std::get<1>(key);
}

/*static*/ std::optional<AllReduceCombiner::GroupKey>
AllReduceCombiner::CombineKey(const HloInstruction* instruction,
                              const HloDomainMap& domain_map) {
  std::optional<AllReduceKey> key = GetAllReduceKey(instruction, &domain_map);
  if (!key.has_value()) {
    return std::nullopt;
  }
  return AllReduceCombiner::GroupKey{*key, /*extra_args*/ ""};
}

absl::StatusOr<bool> AllReduceCombiner::RunWithKeyCombiner(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<std::optional<AllReduceCombiner::GroupKey>(
        const HloInstruction*, const HloDomainMap&)>
        combine_key) {
  VLOG(1) << "Running AllReduceCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip AllReduceCombiner because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceCombiner because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&domain_map, &combine_key](const HloInstruction* instruction)
        -> std::optional<AllReduceCombiner::GroupKey> {
      if (instruction->opcode() != HloOpcode::kAllReduce) {
        return std::nullopt;
      }
      return combine_key(instruction, *domain_map);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<AllReduceCombiner::GroupKey>(
            computation, key_fn, &CombineAllReduces,
            combine_threshold_in_bytes_, combine_threshold_count_));
    changed |= computation_changed;
  }

  return changed;
}

AllReduceCombiner::AllReduceCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count) {}

absl::StatusOr<bool> AllReduceCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return RunWithKeyCombiner(module, execution_threads, CombineKey);
}

}  // namespace xla
