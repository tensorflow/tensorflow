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

#include "xla/codegen/tiling/experimental/scheduling.h"

#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

namespace {

llvm::SmallVector<int64_t> GetParallelDimensionsPermutation(
    const TiledHloComputation& tiled_computation) {
  if (tiled_computation.roots().size() != 1) {
    return {};
  }

  const HloInstruction* root = tiled_computation.roots().front()->hlo();
  if (root->opcode() != HloOpcode::kDot &&
      root->opcode() != HloOpcode::kScaledDot) {
    return {};
  }

  const Shape& lhs_shape = root->operand(0)->shape();
  const Shape& rhs_shape = root->operand(1)->shape();
  const DotDimensionNumbers& dimension_numbers = root->dot_dimension_numbers();

  // We only support transposing standard [batch..., m, k] * [batch..., k, n]
  // shapes where m and n are exactly rank 1.
  int64_t num_lhs_non_contracting_dims =
      lhs_shape.dimensions().size() -
      dimension_numbers.lhs_contracting_dimensions_size() -
      dimension_numbers.lhs_batch_dimensions_size();
  int64_t num_rhs_non_contracting_dims =
      rhs_shape.dimensions().size() -
      dimension_numbers.rhs_contracting_dimensions_size() -
      dimension_numbers.rhs_batch_dimensions_size();
  if (num_lhs_non_contracting_dims != 1 || num_rhs_non_contracting_dims != 1) {
    return {};
  }

  // Heuristic: if the LHS operand is smaller than the RHS operand, it is more
  // beneficial to traverse the RHS non-contracting dimensions ('n') first (more
  // slowly) while keeping the LHS tile in the L2 cache.
  if (ShapeUtil::ByteSizeOf(lhs_shape) >= ShapeUtil::ByteSizeOf(rhs_shape)) {
    return {};
  }

  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  int64_t num_parallel_dims = tiling_space.num_parallel_dimensions();
  DCHECK_GE(num_parallel_dims, 2);

  const TilingSpace::DimensionInfo& m_dim_info = tiling_space.GetDimensionInfo(
      *root, dimension_numbers.lhs_batch_dimensions_size());
  const TilingSpace::DimensionInfo& n_dim_info = tiling_space.GetDimensionInfo(
      *root, dimension_numbers.lhs_batch_dimensions_size() + 1);
  if (m_dim_info.type != TilingSpace::DimensionSemantics::kParallel ||
      n_dim_info.type != TilingSpace::DimensionSemantics::kParallel) {
    return {};
  }

  // Return a permutation that swaps 'm' and 'n' traversal order.
  llvm::SmallVector<int64_t> permutation(num_parallel_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[m_dim_info.id.value()],
            permutation[n_dim_info.id.value()]);
  return permutation;
}

}  // namespace

std::string Schedule::ToString() const {
  std::set<int64_t> dim_ids;
  for (const auto& [dim_id, expr] : dim_id_to_pid_expr) {
    dim_ids.insert(dim_id);
  }
  std::vector<std::string> expr_strs;
  expr_strs.reserve(dim_ids.size());
  for (int64_t dim_id : dim_ids) {
    expr_strs.push_back(
        absl::StrFormat("d%d -> %s", dim_id,
                        dim_id_to_pid_expr.at(dim_id).ToString({"pid"}, {})));
  }
  return absl::StrCat(absl::StrJoin(expr_strs, ", "),
                      ", pid_bounds=", pid_bounds.ToString());
}

absl::StatusOr<Schedule> GetSchedule(
    const TiledHloComputation& tiled_computation) {
  // Compute the block counts for each parallel dimension.
  llvm::SmallVector<int64_t, 4> parallel_dim_block_counts;
  llvm::SmallVector<int64_t, 4> parallel_dim_ids;
  for (const auto& [dim_id, dimension] :
       llvm::enumerate(tiled_computation.tiling_space().dimensions())) {
    if (dimension.type != TilingSpace::DimensionSemantics::kParallel) {
      continue;
    }
    parallel_dim_block_counts.push_back(
        CeilOfRatio(dimension.dimension_size, dimension.tile_size));
    parallel_dim_ids.push_back(dim_id);
  }

  llvm::SmallVector<int64_t> permutation =
      GetParallelDimensionsPermutation(tiled_computation);
  if (!permutation.empty()) {
    parallel_dim_block_counts = llvm::to_vector<4>(
        xla::Permute(parallel_dim_block_counts, permutation));
    parallel_dim_ids =
        llvm::to_vector<4>(xla::Permute(parallel_dim_ids, permutation));
  }

  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  SymbolicExpr pid = CreateDimExpr(0, ctx);
  llvm::SmallVector<SymbolicExpr, 4> delinearized_pid =
      DelinearizeIndex(parallel_dim_block_counts, pid, ctx);
  Schedule schedule;
  for (const auto& [parallel_dim_id, expr] :
       llvm::zip(parallel_dim_ids, delinearized_pid)) {
    schedule.dim_id_to_pid_expr[parallel_dim_id] = expr;
  }
  schedule.pid_bounds = Interval{0, Product(parallel_dim_block_counts) - 1};
  return schedule;
}

}  // namespace xla::gpu::experimental
