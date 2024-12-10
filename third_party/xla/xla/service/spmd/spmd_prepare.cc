/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/spmd/spmd_prepare.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {
namespace {

absl::StatusOr<bool> ProcessScatter(HloInstruction* hlo,
                                    const CallGraph& call_graph) {
  if (hlo->opcode() != HloOpcode::kScatter) {
    return false;
  }
  // Revert a Scatter optimization that could be applied by the
  // algebraic simplifier that concatenates updates and indices for
  // scatter and folds an add of two scatter of the operands. If we
  // detect this optimization has triggered we try to revert it back to
  // two scatters if it is blocking scatter parallel sharding.
  HloScatterInstruction* scatter = Cast<HloScatterInstruction>(hlo);
  HloComputation* computation = hlo->parent();
  // Only support single operand scatters (the optimization triggers
  // only on those anyway).
  if (scatter->scatter_operand_count() > 1) {
    return false;
  }
  HloInstruction* operand = scatter->scatter_operands()[0];
  HloInstruction* indices = scatter->scatter_indices();
  HloInstruction* updates = scatter->scatter_updates()[0];
  // Pattern we are looking for looks like:
  //   scatter(add, concatenate, concatenate), to_apply=add
  if (operand->opcode() != HloOpcode::kAdd ||
      indices->opcode() != HloOpcode::kConcatenate ||
      indices->operand_count() != 2 ||
      updates->opcode() != HloOpcode::kConcatenate ||
      updates->operand_count() != 2 ||
      !Match(scatter->to_apply()->root_instruction(),
             match::AddAnyOrder(match::Parameter(0), match::Parameter(1)))) {
    return false;
  }
  const auto& dnums = scatter->scatter_dimension_numbers();
  // Helper to extract parallel dims based on operand/indices/updates triple.
  auto get_parallel_dims_for_scatter = [&dnums, &call_graph](
                                           const HloInstruction* operand,
                                           const HloInstruction* indices,
                                           const HloInstruction* updates) {
    std::vector<int64_t> slice_sizes = hlo_sharding_util::GetScatterSliceSize(
        operand->shape(), updates->shape(), dnums);
    return hlo_sharding_util::GetGatherScatterBatchParallelDims(
        operand, indices, slice_sizes, dnums.index_vector_dim(),
        dnums.scatter_dims_to_operand_dims(),
        dnums.scatter_indices_batching_dims(), dnums.update_window_dims(),
        call_graph);
  };
  // Parallel dim already detected. Assume everything is good.
  if (get_parallel_dims_for_scatter(operand, indices, updates).has_value()) {
    return false;
  }
  HloInstruction* lhs_indices = indices->mutable_operand(0);
  HloInstruction* rhs_indices = indices->mutable_operand(1);
  HloInstruction* lhs_updates = updates->mutable_operand(0);
  HloInstruction* rhs_updates = updates->mutable_operand(1);
  std::optional<hlo_sharding_util::GatherScatterDims> lhs_parallel_dims;
  std::optional<hlo_sharding_util::GatherScatterDims> rhs_parallel_dims;
  lhs_parallel_dims =
      get_parallel_dims_for_scatter(operand, lhs_indices, lhs_updates);
  // Didn't find any LHS parallel dimension when looking through concat.
  if (!lhs_parallel_dims.has_value()) {
    return false;
  }
  rhs_parallel_dims =
      get_parallel_dims_for_scatter(operand, rhs_indices, rhs_updates);
  // Didn't find any RHS parallel dimension when looking through concat.
  if (!rhs_parallel_dims.has_value()) {
    return false;
  }
  // Make sure the parallel dims are the same between the two pieces.
  if (lhs_parallel_dims->operand_dims != rhs_parallel_dims->operand_dims ||
      lhs_parallel_dims->indices_dims != rhs_parallel_dims->indices_dims) {
    return false;
  }
  if (lhs_parallel_dims->operand_dims.size() !=
      lhs_parallel_dims->indices_dims.size()) {
    return false;
  }
  HloInstruction* lhs_operand = operand->mutable_operand(0);
  HloInstruction* rhs_operand = operand->mutable_operand(1);
  bool any_sharded_parallel_dim = false;
  // Unspecified sharding on operand/indices. Do not continue.
  if (!lhs_operand->has_sharding() || !rhs_operand->has_sharding() ||
      !lhs_indices->has_sharding() || !rhs_indices->has_sharding()) {
    return false;
  }
  // Check any parallel dimension is actually sharded, otherwise splitting the
  // scatter would have no value.
  for (int i = 0; i < lhs_parallel_dims->operand_dims.size(); ++i) {
    if (lhs_operand->sharding().IsTiled() &&
        lhs_operand->sharding().tile_assignment().dim(
            lhs_parallel_dims->operand_dims[i]) != 1 &&
        lhs_indices->sharding().tile_assignment().dim(
            lhs_parallel_dims->indices_dims[i]) != 1) {
      any_sharded_parallel_dim = true;
      break;
    }
  }
  if (!any_sharded_parallel_dim) {
    return false;
  }
  // Split the scatter to:
  //   scatter0 = scatter(operand, indices0, updates0)
  //   scatter1 = scatter(scatter0, indices1, updates1)
  HloInstruction* scatter0 =
      computation->AddInstruction(HloInstruction::CreateScatter(
          scatter->shape(), operand, lhs_indices, lhs_updates,
          scatter->to_apply(), dnums, false, false));
  scatter0->set_metadata(scatter->metadata());
  scatter0->set_sharding(scatter->sharding());
  HloInstruction* scatter1 =
      computation->AddInstruction(HloInstruction::CreateScatter(
          scatter->shape(), scatter0, rhs_indices, rhs_updates,
          scatter->to_apply(), dnums, false, false));
  scatter1->set_metadata(scatter->metadata());
  scatter1->set_sharding(scatter->sharding());
  TF_RETURN_IF_ERROR(scatter->ReplaceAllUsesWith(scatter1));
  return true;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      const CallGraph& call_graph) {
  bool changed = false;
  for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    if (!hlo->has_sharding()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool scatter_changed, ProcessScatter(hlo, call_graph));
    if (scatter_changed) {
      changed = true;
      continue;
    }
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> SpmdPrepare::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (auto comp : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp, *call_graph));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace spmd
}  // namespace xla
