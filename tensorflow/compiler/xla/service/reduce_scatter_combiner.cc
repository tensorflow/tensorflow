/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reduce_scatter_combiner.h"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/all_reduce_key.h"
#include "tensorflow/compiler/xla/service/collective_combiner_utils.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

using ReduceScatterKey =
    std::tuple<AllReduceKey, /*scatter_dimension*/ int64_t>;

// Combines the elements of to_combine into a single ReduceScatter op. All
// entries in to_combine must be ReduceScatter ops with exactly one operand
// and the same reduction operation.
Status CombineReduceScatters(absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.size() < 2) {
    return OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " reduce-scatter ops";

  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  std::optional<ReductionKind> first_reduction_kind =
      MatchReductionComputation(reduction);
  TF_RET_CHECK(first_reduction_kind);

  // Create a single bigger ReduceScatter of the operands of the smaller
  // ReduceScatters.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> output_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kReduceScatter);
    TF_RET_CHECK(hlo->operands().size() == 1);
    std::optional<ReductionKind> reduction_kind =
        MatchReductionComputation(hlo->to_apply());
    TF_RET_CHECK(reduction_kind);
    TF_RET_CHECK(*reduction_kind == *first_reduction_kind);
    TF_RET_CHECK(hlo->shape().IsArray());
    operands.push_back(hlo->operands()[0]);
    output_shapes.push_back(&hlo->shape());
  }

  const auto* rs = Cast<HloReduceScatterInstruction>(to_combine.front());

  HloInstruction* combined;
  // AllReduce ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateReduceScatter(
      ShapeUtil::MakeTupleShapeWithPtrs(output_shapes), operands, reduction,
      to_combine.front()->replica_groups(),
      /*constrain_layout=*/false, to_combine.front()->channel_id(),
      rs->use_global_device_ids(), rs->scatter_dimension()));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  if (to_combine.front()->has_sharding()) {
    combined->set_sharding(to_combine.front()->sharding());
  }
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller ReduceScatters with elements of the tuple output
  // of the single bigger ReduceScatter.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return OkStatus();
}
}  // namespace

ReduceScatterCombiner::ReduceScatterCombiner(int64_t combine_threshold_in_bytes,
                                             int64_t combine_threshold_count)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count) {}

StatusOr<bool> ReduceScatterCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running ReduceScatterCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip ReduceScatterCombiner because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedCollective(
          *module, HloOpcode::kReduceScatter)) {
    VLOG(1) << "Skip ReduceScatterCombiner because the module contains "
               "reduce-scatter "
               "with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&domain_map](const HloInstruction* instruction)
        -> std::optional<ReduceScatterKey> {
      auto* rs = DynCast<HloReduceScatterInstruction>(instruction);
      std::optional<AllReduceKey> key =
          GetAllReduceKey(instruction, domain_map.get());

      if (!rs || !key) {
        return std::nullopt;
      }
      if (!MatchReductionComputation(rs->to_apply())) {
        return std::nullopt;
      }
      return ReduceScatterKey{std::move(*key), rs->scatter_dimension()};
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<ReduceScatterKey>(
            computation, key_fn, &CombineReduceScatters,
            combine_threshold_in_bytes_, combine_threshold_count_));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
