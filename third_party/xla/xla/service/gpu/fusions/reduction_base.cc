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
#include "xla/service/gpu/fusions/reduction_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/union_find.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

int RowReductionGetRowsPerWarp(int reduced_dimension_size) {
  if (WarpSize() % reduced_dimension_size != 0 ||
      reduced_dimension_size >= WarpSize()) {
    return 1;
  }
  return WarpSize() / reduced_dimension_size;
}

int GetVectorSize(const HloFusionAnalysis& analysis,
                  const ReductionDimensions& reduction_dimensions,
                  int num_threads, Vector3 reduction_tiling) {
  // If the minor dimension is not divisible by 2, we can't currently vectorize.
  int64_t minor_dim = reduction_dimensions.dimensions.back();
  if (minor_dim % 2 != 0) {
    return 1;
  }

  // Only enable vectorization if all threads will still have work.
  if (num_threads * 2 > minor_dim) {
    return 1;
  }
  if (MayPreventVectorization(analysis.fusion())) {
    return 1;
  }
  if (reduction_dimensions.is_row_reduction) {
    constexpr int kRowMinorReduced =
        ReductionDimensions::kRowMinorReducedDimension;

    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
        &analysis.device_info().gpu_compute_capability());
    if (cuda_cc == nullptr) return 1;
    if (cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) return 2;
    if (cuda_cc->IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
      return analysis.input_output_info().smallest_input_dtype_bits <= 32 &&
                     reduction_dimensions.dimensions[kRowMinorReduced] %
                             (reduction_tiling[kRowMinorReduced] *
                              num_threads) ==
                         0
                 ? 2
                 : 1;
    }
    return 1;
  }
  return 1;
}

int GetVectorSizeForMlir(const HloFusionAnalysis& analysis,
                         const ReductionDimensions& reduction_dimensions,
                         int num_threads) {
  // If the minor dimension is not divisible by 2, we can't currently vectorize.
  int64_t minor_dim = reduction_dimensions.dimensions.back();
  if (minor_dim % 2 != 0) {
    return 1;
  }
  // Only enable vectorization if all threads will still have work.
  if (num_threads * 2 > minor_dim) {
    return 1;
  }
  // MLIR's vectorization doesn't work with complex types. However, complex
  // load/stores are effectively always vectorized and have a size
  // of at least 8 bytes, which is sufficient.
  for (HloInstructionAdaptor hero : analysis.fusion_heroes()) {
    for (HloInstructionAdaptor operand : hero.GetOperands()) {
      if (primitive_util::IsComplexType(operand.shape().element_type())) {
        return 1;
      }
    }
  }
  // 16 byte vector loads are often slower than 8 byte loads.
  if (analysis.input_output_info().smallest_input_dtype_bits >= 32) {
    return 2;
  }
  if (analysis.input_output_info().smallest_input_dtype_bits >= 64) {
    return 1;
  }
  // Like above, if the size of the minor dimension is not sufficiently large,
  // the vectorization is not helpful.
  if (num_threads * 4 > minor_dim) {
    return 2;
  }
  return minor_dim % 4 == 0 ? 4 : 2;
}

ReductionGroups GroupDisjointReductions(const HloFusionAnalysis& analysis,
                                        bool for_mlir) {
  const int num_fusion_outputs = analysis.fusion_root_count();

  CHECK_NE(0, num_fusion_outputs);
  if (num_fusion_outputs == 1) {
    return {{{&analysis.fusion_root(0).instruction()}}, {0}, {true}};
  }

  absl::node_hash_map<HloInstructionAdaptor,
                      tensorflow::UnionFind<HloInstructionAdaptor>>
      disjoint_sets;

  // TODO(b/249976438): we currently do not treat properly
  // aliasing between inputs and outputs of the fusion, so for now put all
  // non-reduction roots into one group to avoid read-after-write conflicts.
  std::optional<HloInstructionAdaptor> first_non_reduction_root = std::nullopt;

  absl::node_hash_map<HloInstructionAdaptor,
                      absl::flat_hash_set<HloInstructionAdaptor>>
      reachable_outputs;
  absl::flat_hash_set<HloInstructionAdaptor> roots_with_reduction;
  absl::flat_hash_map<const HloInstruction*, int> root_indices;
  const auto& roots = analysis.fusion().GetRoots();
  ReductionGroups result;
  result.group_id_per_root.resize(roots.size());
  result.is_reduction_root.reserve(roots.size());
  for (auto [root, hero] : llvm::zip(roots, analysis.fusion_heroes())) {
    int index = root_indices.size();
    root_indices[&root.instruction()] = index;
    disjoint_sets[root].Get() = root;
    reachable_outputs[root].insert(root);
    result.is_reduction_root.push_back(
        IsRealReductionHero(root.instruction(), hero.instruction()));
    if (result.is_reduction_root.back()) {
      roots_with_reduction.insert(root);
    } else if (first_non_reduction_root) {
      disjoint_sets[*first_non_reduction_root].Merge(&disjoint_sets[root]);
    } else {
      first_non_reduction_root = root;
    }
  }

  absl::flat_hash_set<HloInstructionAdaptor> instructions;

  auto visit = [&](absl::Span<const HloInstructionAdaptor> roots) {
    HloBfsConsumersFirstTraversal(
        roots, analysis.fusion(),
        [&](HloInstructionAdaptor consumer) {
          auto& consumer_reachable = reachable_outputs[consumer];
          for (auto producer : consumer.GetOperands()) {
            reachable_outputs[producer].insert(consumer_reachable.begin(),
                                               consumer_reachable.end());
          }
          instructions.insert(consumer);
          return TraversalResult::kAdvance;
        },
        [&](HloInstructionAdaptor argument) { instructions.insert(argument); });
  };

  // The legacy emitter grouping is buggy: it does not visit instructions in the
  // right order. We fix this only for the MLIR emitters, since we are migrating
  // to them, and we can't rule out that some models rely on the buggy behavior.
  if (for_mlir) {
    for (auto root : roots) {
      visit({root});
    }
  } else {
    visit(roots);
  }

  for (auto instr : instructions) {
    const auto& reachable = reachable_outputs[instr];
    std::vector<HloInstructionAdaptor> reached_output_ids;
    bool added_to_reduce = false;
    for (auto output : roots) {
      bool has_real_hero = roots_with_reduction.contains(output);
      if (has_real_hero &&
          (hlo_query::IsBroadcastedConstantOrScalar(instr.instruction()))) {
        if (added_to_reduce) {
          // Do not group more than one output reduce instructions through
          // broadcasted constants or scalars, as the recomputation should be
          // acceptable.
          VLOG(3) << "Skip broadcasted constant or scalar " << instr.ToString();
          continue;
        }
      }
      // Now group output instructions if they have common predecessors.
      if (reachable.contains(output)) {
        VLOG(3) << "Reaching " << output.ToString() << " from "
                << instr.ToString();
        reached_output_ids.push_back(output);
        if (has_real_hero) {
          added_to_reduce = true;
        }
      }
    }
    for (size_t j = 1; j < reached_output_ids.size(); ++j) {
      disjoint_sets[reached_output_ids[0]].Merge(
          &disjoint_sets[reached_output_ids[j]]);
    }
  }

  // Place output instructions in the same set into the same group.
  ConstHloInstructionMap<std::vector<const HloInstruction*>> group_map;
  for (auto root : roots) {
    group_map[&disjoint_sets[root].Get().instruction()].push_back(
        &root.instruction());
  }

  result.grouped_roots.reserve(group_map.size());
  absl::c_for_each(group_map, [&](auto& it) {
    for (auto* root : it.second) {
      result.group_id_per_root[root_indices[root]] =
          result.grouped_roots.size();
    }
    result.grouped_roots.emplace_back(std::move(it.second));
  });
  return result;
}

void AddGroupIdConstraint(IndexingMap& map, int64_t root_index,
                          const ReductionGroups& groups) {
  // Only threads with the right y block index actually do anything for each
  // particular root.
  int group_index = groups.group_id_per_root[root_index];
  map.AddConstraint(
      mlir::getAffineDimExpr(KernelFusionInterface::kIndexingMapBlockIdxDims[1],
                             map.GetMLIRContext()),
      {group_index, group_index});
}

}  // namespace gpu
}  // namespace xla
