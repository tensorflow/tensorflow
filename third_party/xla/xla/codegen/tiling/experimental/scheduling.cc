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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {

namespace {

// Computes the parallel-dimensions permutation for the ragged non-contracting
// case.
//
// kRaggedNonContracting: output is (M_total, N).  G and K are kSequential,
//   so M and N are the only two kParallel dims.
//   Use M_total/G as a static estimate of per-group M (M_avg).
//   If M_avg < N → LHS activation tiles are smaller → traverse N slowly.
llvm::SmallVector<int64_t> GetRaggedNonContractingPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    const DotDimensionNumbers& dot_dims, int64_t num_parallel_dims,
    int64_t M_avg) {
  // kRaggedNonContracting: parallel dims are M (output dim num_batch) and N
  // (output dim num_batch+1).  The M parallel dim covers M_total tokens,
  // but each group only uses M_avg rows on average.  Compare M_avg
  // (≈ per-group LHS non-contracting size) against N (fixed RHS
  // non-contracting size).
  const int64_t num_batch = dot_dims.lhs_batch_dimensions_size();
  // N is at output position num_batch+1 (after batch dims and M).
  if (M_avg >= root.shape().dimensions(num_batch + 1)) {
    return {};  // LHS not smaller; no swap beneficial.
  }
  // M is at output dim num_batch, N is at output dim num_batch+1.
  const TilingSpace::DimensionInfo& m_dim_info =
      tiling_space.GetDimensionInfo(root, num_batch);
  const TilingSpace::DimensionInfo& n_dim_info =
      tiling_space.GetDimensionInfo(root, num_batch + 1);
  if (m_dim_info.type != TilingSpace::DimensionSemantics::kParallel ||
      n_dim_info.type != TilingSpace::DimensionSemantics::kParallel) {
    return {};
  }
  llvm::SmallVector<int64_t> permutation(num_parallel_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[m_dim_info.id.value()],
            permutation[n_dim_info.id.value()]);
  return permutation;
}

// Computes the parallel-dimensions permutation for the ragged contracting case.
//
// kRaggedContracting: output is (G, K, N).  G, K, N are all kParallel.
//   The ragged accumulation dimension (M) is kSequential.
//   Apply the same heuristic to the K/N pair (treating K like M in the
//   regular dot case):
//   Use M_total/G as estimate for per-group activation rows (M_avg).
//   If M_avg * K < K * N, i.e., M_avg < N → swap K and N traversal order.
llvm::SmallVector<int64_t> GetRaggedContractingPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    const DotDimensionNumbers& dot_dims, int64_t num_parallel_dims,
    int64_t M_avg) {
  // kRaggedContracting: parallel dims are G (output dim 0), K (output dim
  // 1), N (output dim 2).  Apply the M/N-like heuristic to K vs N:
  // M_avg is the average number of contracting elements per group; if it
  // is small, the per-tile LHS slice (M_avg × K) is smaller than the
  // per-tile RHS slice (M_avg × N), so traverse N more slowly.
  const int64_t num_batch = dot_dims.lhs_batch_dimensions_size();
  // Output ordering: [G, batch..., K (lhs_nc), N (rhs_nc)].
  const int64_t k_output_dim = 1 + num_batch;
  const int64_t n_output_dim = k_output_dim + 1;

  if (n_output_dim >= root.shape().dimensions().size()) {
    return {};
  }
  if (M_avg >= root.shape().dimensions(n_output_dim)) {
    return {};  // Per-group LHS not smaller; no swap beneficial.
  }

  const TilingSpace::DimensionInfo& k_dim_info =
      tiling_space.GetDimensionInfo(root, k_output_dim);
  const TilingSpace::DimensionInfo& n_dim_info =
      tiling_space.GetDimensionInfo(root, n_output_dim);
  if (k_dim_info.type != TilingSpace::DimensionSemantics::kParallel ||
      n_dim_info.type != TilingSpace::DimensionSemantics::kParallel) {
    return {};
  }
  // Compute the parallel-position index (rank among kParallel dims only)
  // for K and N.  We must NOT use the global dimension IDs here because
  // the permutation has size num_parallel_dims, but G's global ID is 0,
  // which shifts K to global ID 1 and N to global ID 2.  Using global IDs
  // directly would index the permutation out of bounds.
  int64_t k_parallel_pos = -1, n_parallel_pos = -1;
  {
    int64_t pos = 0;
    for (const auto& d : tiling_space.dimensions()) {
      if (d.type == TilingSpace::DimensionSemantics::kParallel) {
        if (d.id == k_dim_info.id) {
          k_parallel_pos = pos;
        }
        if (d.id == n_dim_info.id) {
          n_parallel_pos = pos;
        }
        ++pos;
      }
    }
  }
  if (k_parallel_pos < 0 || n_parallel_pos < 0 ||
      k_parallel_pos >= num_parallel_dims ||
      n_parallel_pos >= num_parallel_dims) {
    return {};  // safety check
  }
  llvm::SmallVector<int64_t> permutation(num_parallel_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[k_parallel_pos], permutation[n_parallel_pos]);
  return permutation;
}

// Computes the parallel-dimensions permutation for a ragged dot.
//
// kRaggedNonContracting: output is (M_total, N).  G and K are kSequential,
//   so M and N are the only two kParallel dims.
//   Use M_total/G as a static estimate of per-group M (M_avg).
//   If M_avg < N → LHS activation tiles are smaller → traverse N slowly.
//
// kRaggedContracting: output is (G, K, N).  G, K, N are all kParallel.
//   The ragged accumulation dimension (M) is kSequential.
//   Apply the same heuristic to the K/N pair (treating K like M in the
//   regular dot case):
//   Use M_total/G as estimate for per-group activation rows (M_avg).
//   If M_avg * K < K * N, i.e., M_avg < N → swap K and N traversal order.
llvm::SmallVector<int64_t> GetRaggedDotPermutation(
    const HloInstruction& root, const TilingSpace& tiling_space,
    int64_t num_parallel_dims) {
  const auto* ragged_dot = Cast<HloRaggedDotInstruction>(&root);
  const RaggedDotDimensionNumbers& ragged_dims =
      ragged_dot->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();

  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  const bool is_contracting =
      absl::c_count(dot_dims.lhs_contracting_dimensions(), lhs_ragged_dim) > 0;
  const bool is_batch =
      absl::c_count(dot_dims.lhs_batch_dimensions(), lhs_ragged_dim) > 0;
  if (is_batch) {
    return {};  // kRaggedBatch not yet implemented.
  }

  const Shape& lhs_shape = root.operand(0)->shape();
  const Shape& group_sizes_shape = root.operand(2)->shape();
  const int64_t M_total = lhs_shape.dimensions(lhs_ragged_dim);
  const int64_t G = group_sizes_shape.dimensions(0);
  const int64_t M_avg = M_total / std::max<int64_t>(G, 1);

  if (!is_contracting) {
    return GetRaggedNonContractingPermutation(root, tiling_space, dot_dims,
                                              num_parallel_dims, M_avg);
  }
  return GetRaggedContractingPermutation(root, tiling_space, dot_dims,
                                         num_parallel_dims, M_avg);
}

llvm::SmallVector<int64_t> GetParallelDimensionsPermutation(
    const TiledHloComputation& tiled_computation) {
  if (tiled_computation.roots().size() != 1) {
    return {};
  }

  const HloInstruction* root = tiled_computation.roots().front()->hlo();
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  int64_t num_parallel_dims = tiling_space.num_parallel_dimensions();
  if (num_parallel_dims < 2) {
    return {};
  }

  // ---- Regular dot / scaled dot ----
  if (root->opcode() == HloOpcode::kDot ||
      root->opcode() == HloOpcode::kScaledDot) {
    const Shape& lhs_shape = root->operand(0)->shape();
    const Shape& rhs_shape = root->operand(1)->shape();
    const DotDimensionNumbers& dimension_numbers =
        root->dot_dimension_numbers();

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
    if (num_lhs_non_contracting_dims != 1 ||
        num_rhs_non_contracting_dims != 1) {
      return {};
    }

    // Heuristic: if the LHS operand is smaller than the RHS operand, it is more
    // beneficial to traverse the RHS non-contracting dimensions ('n') first
    // (more slowly) while keeping the LHS tile in the L2 cache.
    if (ShapeUtil::ByteSizeOf(lhs_shape) >= ShapeUtil::ByteSizeOf(rhs_shape)) {
      return {};
    }

    const TilingSpace::DimensionInfo& m_dim_info =
        tiling_space.GetDimensionInfo(
            *root, dimension_numbers.lhs_batch_dimensions_size());
    const TilingSpace::DimensionInfo& n_dim_info =
        tiling_space.GetDimensionInfo(
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

  // ---- Ragged dot ----
  if (root->opcode() == HloOpcode::kRaggedDot) {
    return GetRaggedDotPermutation(*root, tiling_space, num_parallel_dims);
  }

  return {};
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
    expr_strs.push_back(absl::StrFormat(
        "d%d -> %s", dim_id,
        dim_id_to_pid_expr.at(dim_id).ToString({"pid", "tid"}, {})));
  }
  return absl::StrFormat("%s, num_pids=%d, num_tiles=%d",
                         absl::StrJoin(expr_strs, ", "), num_pids, num_tiles);
}

absl::StatusOr<Schedule> GetSchedule(
    const TiledHloComputation& tiled_computation, int64_t num_tiles_per_pid) {
  // Compute the block counts for each parallel dimension.
  llvm::SmallVector<int64_t, 4> parallel_dim_block_counts;
  llvm::SmallVector<int64_t, 4> parallel_dim_ids;
  for (const auto& [dim_id, dimension] :
       llvm::enumerate(tiled_computation.tiling_space().dimensions())) {
    if (dimension.type != TilingSpace::DimensionSemantics::kParallel) {
      continue;
    }
    parallel_dim_block_counts.push_back(
        CeilOfRatio(dimension.dimension_size, *dimension.tile_size));
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
  SymbolicExpr program_id = CreateDimExpr(0, ctx);
  SymbolicExpr global_tile_id = program_id;
  if (num_tiles_per_pid > 1) {
    SymbolicExpr tile_id = CreateDimExpr(1, ctx);
    global_tile_id = program_id * num_tiles_per_pid + tile_id;
  }
  llvm::SmallVector<SymbolicExpr, 4> delinearized_pid =
      DelinearizeIndex(parallel_dim_block_counts, global_tile_id, ctx);
  Schedule schedule;
  for (const auto& [parallel_dim_id, expr] :
       llvm::zip(parallel_dim_ids, delinearized_pid)) {
    schedule.dim_id_to_pid_expr[parallel_dim_id] = expr;
  }
  schedule.num_tiles = Product(parallel_dim_block_counts);
  schedule.num_pids = CeilOfRatio(schedule.num_tiles, num_tiles_per_pid);
  return schedule;
}

int64_t Schedule::GetNumTilesPerPid() const {
  return CeilOfRatio(num_tiles, num_pids);
}

}  // namespace xla::gpu::experimental
