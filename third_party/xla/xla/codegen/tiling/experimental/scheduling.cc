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
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
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

// What the scheduler needs to know about the index-wise variadic instructions
// of a computation.
struct IndexWiseVariadicInfo {
  // The index-wise variadic instructions, in any region of the computation.
  llvm::SmallVector<const TiledHloInstruction*, 2> instructions;
  // Maps each all-gather dimension ID to how many tiles each peer owns along
  // that dimension.
  //
  // Example: `f32[4, 16] -> f32[16, 16] all-gather` across 4 GPUs on dim 0
  // (`d0`) with tile size [2, 16]:
  //   - `d0` has 16 / 2 = 8 tiles in total across all 4 GPUs.
  //   - Each GPU owns a 4-row slice, which is 4 / 2 = 2 tiles along `d0`.
  //   - So `tiles_per_peer_by_dim[d0] = 2`.
  llvm::DenseMap<int64_t, int64_t> tiles_per_peer_by_dim;
  // Number of peers across `tiles_per_peer_by_dim` and `kReplica` dimensions.
  int64_t num_peers = 1;
};

// Collects the index-wise variadic instructions of `region` and its nested
// regions.
void CollectIndexWiseVariadic(
    const TiledHloRegion& region,
    llvm::SmallVectorImpl<const TiledHloInstruction*>& result) {
  for (const TiledHloInstruction* instruction : region.instructions()) {
    if (IsIndexWiseVariadic(*instruction->hlo())) {
      result.push_back(instruction);
    }
    for (const TiledHloRegion& nested : instruction->hlo_regions()) {
      CollectIndexWiseVariadic(nested, result);
    }
  }
}

// Finds the index-wise variadic instructions of the computation and, for the
// operand tiles that carry a replica id, derives which parallel tile dimension
// carries the peer index and how many tiles per peer it spans.
absl::StatusOr<IndexWiseVariadicInfo> BuildIndexWiseVariadicInfo(
    const TiledHloComputation& tiled_computation) {
  IndexWiseVariadicInfo info;
  CollectIndexWiseVariadic(tiled_computation.tiled_root_region(),
                           info.instructions);
  if (info.instructions.empty()) {
    return info;
  }

  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  TF_RET_CHECK(!tiling_space.IsSymbolic())
      << "Expected tile sizes to be assigned at this point.";
  llvm::SmallVector<TilingSpace::DimensionInfo, 4> dims =
      tiling_space.dimensions();
  const int64_t num_dims = dims.size();

  llvm::SmallVector<int64_t, 4> tile_sizes;
  tile_sizes.reserve(num_dims);
  for (const TilingSpace::DimensionInfo& dim : dims) {
    tile_sizes.push_back(*dim.tile_size);
  }
  // Variables are laid out as [tile ids, tile sizes, runtime variables].
  const int64_t num_vars = 2 * num_dims + tiling_space.num_rt_vars();

  std::optional<int64_t> num_peers;
  for (const TiledHloInstruction* instruction : info.instructions) {
    const HloInstruction& hlo = *instruction->hlo();
    for (const auto& [output_index, operand] :
         llvm::enumerate(instruction->operands())) {
      const Tile& tile = operand->tile();
      if (tile.num_replica_ids() == 0) {
        // The output tile is not associated with a single peer; nothing to
        // factor out of its tile ids.
        continue;
      }
      TF_RET_CHECK(tile.num_replica_ids() == 1)
          << "Only one replica dimension is supported.";
      const DimTile& replica_dim_tile = tile.replica_ids().back();
      const int64_t num_replicas = replica_dim_tile.upper_bound.GetValue();
      SymbolicExpr peer = replica_dim_tile.offset;
      llvm::DenseSet<VariableID> used_vars;
      peer.GetUsedVariables(used_vars);
      if (used_vars.size() != 1 || *used_vars.begin() >= num_dims ||
          dims[*used_vars.begin()].type !=
              TilingSpace::DimensionSemantics::kParallel) {
        return absl::UnimplementedError(absl::StrFormat(
            "Peer index %s of output %d of %s must depend on exactly one "
            "parallel tile id.",
            peer.ToString(num_dims), output_index, hlo.name()));
      }
      const int64_t dim_id = *used_vars.begin();
      // Number of blocks across all replicas.
      const int64_t num_blocks =
          CeilOfRatio(dims[dim_id].dimension_size, tile_sizes[dim_id]);
      if (num_blocks < num_replicas || num_blocks % num_replicas != 0) {
        return absl::UnimplementedError(absl::StrFormat(
            "Dimension %d of output %d of %s has %d blocks, which is not a "
            "multiple of the number of peers %d.",
            dim_id, output_index, hlo.name(), num_blocks, num_replicas));
      }
      const int64_t tiles_per_peer = num_blocks / num_replicas;
      std::vector<int64_t> values(num_vars, 0);
      auto peer_at = [&](int64_t tile_id) {
        values[dim_id] = tile_id;
        return peer.Evaluate(values);
      };
      if (!llvm::all_of(llvm::seq<int64_t>(0, num_blocks), [&](int64_t t) {
            return peer_at(t) == t / tiles_per_peer;
          })) {
        return absl::UnimplementedError(absl::StrFormat(
            "Peer index %s of output %d of %s must be tile id %d divided by "
            "%d.",
            peer.ToString(num_dims), output_index, hlo.name(), dim_id,
            tiles_per_peer));
      }

      if (num_peers.has_value() && *num_peers != num_replicas) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output %d of %s reads from %d peers, but other outputs read "
            "from %d peers.",
            output_index, hlo.name(), num_replicas, *num_peers));
      }
      num_peers = num_replicas;
      info.tiles_per_peer_by_dim[dim_id] = tiles_per_peer;
    }
  }
  for (const TilingSpace::DimensionInfo& dim : dims) {
    if (dim.type != TilingSpace::DimensionSemantics::kReplica) {
      continue;
    }
    if (num_peers.has_value() && *num_peers != dim.dimension_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Replica dimension %d of %s has %d replicas, but other dimensions "
          "have %d peers.",
          dim.id.value(), dim.hlo->name(), dim.dimension_size, *num_peers));
    }
    num_peers = dim.dimension_size;
  }
  info.num_peers = num_peers.value_or(1);
  return info;
}

// Internal struct for holding the parallel dimensions of a root.
struct RootDims {
  // Parallel output dimension (all-gather) or `kReplica` dimension
  // (reduce-scatter) whose tile ID encodes the peer index.
  std::optional<int64_t> peer_dim_id;
  int64_t tiles_per_peer = 1;
  llvm::SmallVector<int64_t, 4> non_peer_dim_ids;
  llvm::SmallVector<int64_t, 4> non_peer_block_counts;
  int64_t num_tiles = 1;
};

absl::StatusOr<llvm::SmallVector<RootDims, 2>> BuildRootsForIndexWiseVariadic(
    const TiledHloComputation& tiled_computation,
    const IndexWiseVariadicInfo& info) {
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  // Group the parallel dimensions by the root (output) that defines them.
  llvm::SmallVector<RootDims, 2> roots;
  int64_t num_grouped_dims = 0;
  for (const auto& [root_index, tiled_root] :
       llvm::enumerate(tiled_computation.roots())) {
    const HloInstruction& hlo = *tiled_root->hlo();
    RootDims& root = roots.emplace_back();
    for (int64_t i = 0; i < GetFirstShape(&hlo).dimensions().size(); ++i) {
      const TilingSpace::DimensionInfo* dim =
          &tiling_space.GetDimensionInfo(hlo, i);
      if (dim->type != TilingSpace::DimensionSemantics::kParallel) {
        continue;
      }
      ++num_grouped_dims;
      const int64_t dim_id = dim->id.value();
      auto it = info.tiles_per_peer_by_dim.find(dim_id);
      if (dim->replica_dim.has_value() ||
          it != info.tiles_per_peer_by_dim.end()) {
        if (root.peer_dim_id.has_value() ||
            (dim->replica_dim.has_value() &&
             it != info.tiles_per_peer_by_dim.end())) {
          return absl::UnimplementedError(absl::StrFormat(
              "Output %d of %s has more than one dimension carrying the peer "
              "index.",
              root_index, hlo.name()));
        }
        if (dim->replica_dim.has_value()) {
          root.peer_dim_id = dim->replica_dim->value();
        } else {
          root.peer_dim_id = dim_id;
          root.tiles_per_peer = it->second;
          continue;
        }
      }
      root.non_peer_dim_ids.push_back(dim_id);
      root.non_peer_block_counts.push_back(tiling_space.GetNumTiles(*dim));
    }
    if (info.num_peers > 1 && !root.peer_dim_id.has_value()) {
      return absl::UnimplementedError(
          absl::StrFormat("Output %d of %s does not depend on the peer index.",
                          root_index, hlo.name()));
    }
    root.num_tiles = Product(root.non_peer_block_counts) * root.tiles_per_peer;
  }
  if (num_grouped_dims != tiling_space.num_parallel_dimensions()) {
    return absl::UnimplementedError(
        "Not all parallel dimensions are defined by the roots.");
  }
  return roots;
}

absl::StatusOr<Schedule> BuildIndexWiseVariadicSchedule(
    mlir::MLIRContext* ctx, const IndexWiseVariadicInfo& info,
    llvm::ArrayRef<RootDims> roots, int64_t num_tiles_per_pid,
    int64_t max_tiles, SymbolicExpr global_tile_id) {
  Schedule schedule;
  SymbolicExpr peer = global_tile_id.floorDiv(max_tiles);
  SymbolicExpr intra_peer_tile_id =
      info.num_peers > 1 ? global_tile_id % max_tiles : global_tile_id;
  for (const RootDims& root : roots) {
    llvm::SmallVector<int64_t, 4> block_counts = root.non_peer_block_counts;
    if (root.tiles_per_peer > 1) {
      block_counts.push_back(root.tiles_per_peer);
    }
    // `DelinearizeIndex` does not wrap its outermost component, so for a padded
    // tile (`intra_peer_tile_id >= num_tiles`) that component exceeds its
    // block count and the tile is masked by its upper bound.
    llvm::SmallVector<SymbolicExpr, 4> delinearized =
        DelinearizeIndex(block_counts, intra_peer_tile_id, ctx);
    for (int64_t i = 0; i < root.non_peer_dim_ids.size(); ++i) {
      schedule.dim_id_to_pid_expr[root.non_peer_dim_ids[i]] = delinearized[i];
    }
    if (!root.peer_dim_id.has_value()) {
      continue;
    }
    SymbolicExpr peer_tile_id = peer * root.tiles_per_peer;
    if (root.tiles_per_peer > 1) {
      peer_tile_id = peer_tile_id + delinearized.back();
    }
    if (root.num_tiles < max_tiles &&
        (root.non_peer_block_counts.empty() ||
         root.non_peer_block_counts.front() == 1)) {
      // No non-trivial non-peer dimension can absorb the padding: a non-peer
      // dimension covered by a single block folds its tile offset to 0, and
      // an unpadded `tiles_per_peer` offset would step into the next peer
      // instead of out of bounds. Push the peer tile id past the last peer.
      peer_tile_id =
          peer_tile_id + intra_peer_tile_id.floorDiv(root.num_tiles) *
                             (info.num_peers * root.tiles_per_peer);
    }
    schedule.dim_id_to_pid_expr[*root.peer_dim_id] = peer_tile_id;
  }
  schedule.num_tiles = info.num_peers * max_tiles;
  schedule.num_pids = CeilOfRatio(schedule.num_tiles, num_tiles_per_pid);
  return schedule;
}

// Schedules a computation that contains index-wise variadic instructions.
//
// All roots are "zipped": one program handles tile `i` of every root, so
// that each variadic instruction is executed once per program for all of
// its outputs. If the outputs are associated with a peer the peer index is
// the outermost tile index and is shared by all roots; below it every root
// gets its own intra-peer tile index over its remaining ("non-peer")
// parallel dimensions and any intra-peer tiles along the peer dimension
// itself. Roots with fewer tiles than the largest root are padded: their
// padded tiles are mapped out of bounds so that the emitter masks them.
absl::StatusOr<Schedule> GetIndexWiseVariadicSchedule(
    const TiledHloComputation& tiled_computation,
    const IndexWiseVariadicInfo& info, SymbolicExpr global_tile_id,
    int64_t num_tiles_per_pid) {
  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  using Roots = llvm::SmallVector<RootDims, 2>;
  ABSL_ASSIGN_OR_RETURN(Roots roots,
                   BuildRootsForIndexWiseVariadic(tiled_computation, info));

  // Find the maximum number of tiles across all roots.
  int64_t max_tiles = 1;
  for (const RootDims& root : roots) {
    max_tiles = std::max(max_tiles, root.num_tiles);
  }
  // Now build the scheduled rooted on the output with the max number of tiles.
  return BuildIndexWiseVariadicSchedule(ctx, info, roots, num_tiles_per_pid,
                                        max_tiles, global_tile_id);
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
  mlir::MLIRContext* ctx = tiled_computation.GetMLIRContext();
  SymbolicExpr program_id = CreateDimExpr(0, ctx);
  SymbolicExpr global_tile_id = program_id;
  if (num_tiles_per_pid > 1) {
    SymbolicExpr tile_id = CreateDimExpr(1, ctx);
    global_tile_id = program_id * num_tiles_per_pid + tile_id;
  }
  ABSL_ASSIGN_OR_RETURN(IndexWiseVariadicInfo variadic_info,
                   BuildIndexWiseVariadicInfo(tiled_computation));
  if (!variadic_info.instructions.empty()) {
    return GetIndexWiseVariadicSchedule(tiled_computation, variadic_info,
                                        global_tile_id, num_tiles_per_pid);
  }

  // Compute the block counts for each replica and parallel dimension.
  // Replica dimensions are placed outermost so each contiguous group of
  // parallel blocks targets the same peer.
  const TilingSpace& tiling_space = tiled_computation.tiling_space();
  llvm::SmallVector<int64_t, 4> block_counts;
  llvm::SmallVector<int64_t, 4> dim_ids;
  llvm::SmallVector<int64_t, 4> parallel_dim_block_counts;
  llvm::SmallVector<int64_t, 4> parallel_dim_ids;
  for (const auto& [dim_id, dimension] :
       llvm::enumerate(tiling_space.dimensions())) {
    if (dimension.type == TilingSpace::DimensionSemantics::kReplica) {
      block_counts.push_back(tiling_space.GetNumTiles(dimension));
      dim_ids.push_back(dim_id);
    } else if (dimension.type == TilingSpace::DimensionSemantics::kParallel) {
      parallel_dim_block_counts.push_back(tiling_space.GetNumTiles(dimension));
      parallel_dim_ids.push_back(dim_id);
    }
  }

  llvm::SmallVector<int64_t> permutation =
      GetParallelDimensionsPermutation(tiled_computation);
  if (!permutation.empty()) {
    parallel_dim_block_counts = llvm::to_vector<4>(
        xla::Permute(parallel_dim_block_counts, permutation));
    parallel_dim_ids =
        llvm::to_vector<4>(xla::Permute(parallel_dim_ids, permutation));
  }
  block_counts.append(parallel_dim_block_counts);
  dim_ids.append(parallel_dim_ids);

  llvm::SmallVector<SymbolicExpr, 4> delinearized_pid =
      DelinearizeIndex(block_counts, global_tile_id, ctx);
  Schedule schedule;
  for (const auto& [dim_id, expr] : llvm::zip(dim_ids, delinearized_pid)) {
    schedule.dim_id_to_pid_expr[dim_id] = expr;
  }
  schedule.num_tiles = Product(block_counts);
  schedule.num_pids = CeilOfRatio(schedule.num_tiles, num_tiles_per_pid);
  return schedule;
}

int64_t Schedule::GetNumTilesPerPid() const {
  return CeilOfRatio(num_tiles, num_pids);
}

}  // namespace xla::gpu::experimental
