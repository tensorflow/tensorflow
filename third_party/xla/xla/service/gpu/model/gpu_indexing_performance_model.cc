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

#include "xla/service/gpu/model/gpu_indexing_performance_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/codegen/tiling/tile.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/tiling_from_block_parameters.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {

using ::xla::xtile::BlockLevelParameters;
using ::xla::xtile::GetTilingSpaceConcreteSizes;

namespace {

namespace dot_model = ::xla::gpu::gpu_dot_fusion_cost_model::detail;

// Information about an operand read.
struct OperandReadInfo {
  // Total number of bytes read from the operand.
  int64_t total_bytes_read = 0;

  // Factor, between 0 and 1, determining how much of the chip's HBM bandwidth
  // is actually attained when reading this operand.
  double read_bandwidth_utilization_rate = 1.0;
};

using OperandReadMap =
    absl::flat_hash_map<const HloInstruction*, OperandReadInfo>;

// Accumulates `tile_bytes_read` and tracks the minimum
// `bandwidth_utilization_rate` for `hlo` in `read_map`.
void RecordOperandRead(const HloInstruction* hlo, int64_t tile_bytes_read,
                       double bandwidth_utilization_rate,
                       OperandReadMap& read_map) {
  OperandReadInfo& operand_read_info = read_map[hlo];
  operand_read_info.total_bytes_read += tile_bytes_read;
  // TODO(b/332714755): using std::min is more pessimistic than it needs
  // to be since it'll end up assuming that if one read is done with lower
  // bandwidth, all other reads of the same operand will also be done with
  // lower bandwidth. But it's a start. We should refactor this function
  // to properly track each read independently later.
  operand_read_info.read_bandwidth_utilization_rate =
      std::min(operand_read_info.read_bandwidth_utilization_rate,
               bandwidth_utilization_rate);
}

// Returns the unique bytes read from DRAM for an operand, capped at the
// operand's total size since re-reads across tiles are assumed to hit cache.
int64_t GetDramBytesRead(const HloInstruction* hlo, const OperandReadInfo& info,
                         const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  return std::min(shape_size(hlo->shape()), info.total_bytes_read);
}

// Computes total net HBM bytes read across all unique operands.
int64_t GetNetBytesRead(const OperandReadMap& read_map,
                        const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  int64_t net_bytes = 0;
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [hlo, info] : read_map) {
    net_bytes += GetDramBytesRead(hlo, info, shape_size);
  }
  return net_bytes;
}

// Computes the memory read time across all operands.
absl::Duration ComputeReadTime(
    const OperandReadMap& read_map, const se::DeviceDescription& device_info,
    int64_t num_blocks, const HloCostAnalysis::ShapeSizeFunction& shape_size,
    bool include_cache_rereads = true, double max_bandwidth_utilization = 1.0) {
  absl::Duration read_time = absl::ZeroDuration();
  // `absl::Duration` addition uses exact fixed-point integer arithmetic, so
  // map iteration order does not affect the accumulated result.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [hlo, operand_read_info] : read_map) {
    int64_t dram_bytes = GetDramBytesRead(hlo, operand_read_info, shape_size);
    int64_t total_bytes =
        include_cache_rereads ? operand_read_info.total_bytes_read : dram_bytes;

    read_time += GpuPerformanceModelBase::ReadTimeWithDRAMHeuristic(
        device_info, num_blocks, /*n_bytes_net=*/dram_bytes,
        /*n_bytes_total=*/total_bytes,
        /*element_type=*/hlo->shape().element_type(),
        /*hbm_bandwidth_utilization_rate=*/
        std::min(operand_read_info.read_bandwidth_utilization_rate,
                 max_bandwidth_utilization));
  }
  return read_time;
}

// Returns the number of elements in the tile after each dimension is padded to
// the next power of 2.
// TODO(b/353484968): Delete this function once we have constraints to only
// propagate tile sizes that are a power of 2.
int64_t GetPaddedTileSize(absl::Span<int64_t const> tile_sizes) {
  int64_t result = 1;
  for (int64_t tile_size : tile_sizes) {
    result *= llvm::PowerOf2Ceil(tile_size);
  }
  return result;
}

int64_t GetStaticPaddedTileSize(const TiledHloInstruction& tiled_hlo) {
  absl::StatusOr<llvm::SmallVector<int64_t>> tile_sizes =
      tiled_hlo.tile(0).GetStaticTileSizes();
  CHECK_OK(tile_sizes);
  return GetPaddedTileSize(*tile_sizes);
}

// Evaluates the symbolic tile sizes of `tiled_hlo` for `variable_values`, the
// values of all tiling space variables.
llvm::SmallVector<int64_t> EvaluateTileSizes(
    const TiledHloInstruction& tiled_hlo,
    absl::Span<const int64_t> variable_values) {
  return llvm::map_to_vector(tiled_hlo.tile().dim_tiles(),
                             [&](const DimTile& dim_tile) {
                               return dim_tile.size.Evaluate(variable_values);
                             });
}

// Constructs the values of all tiling space variables for `tiling_space` and
// concrete `tile_sizes`.
llvm::SmallVector<int64_t> GetTilingSpaceVariableValues(
    const TilingSpace& tiling_space, absl::Span<const int64_t> tile_sizes) {
  // The tiling space variables are tile IDs and tile sizes of all dimensions,
  // followed by runtime variables. The tile sizes of a concrete tiling are
  // constants, so any valid tile ID and runtime variable value gives the same
  // result. 0 is valid for both.
  const int64_t num_dimensions = tiling_space.num_dimensions();
  llvm::SmallVector<int64_t> variable_values(
      2 * num_dimensions + tiling_space.num_rt_vars(), 0);
  // `tile_sizes` has no entries for replica dimensions, which have tile size 1
  // (see `TilingSpace::AssignTileSizes`).
  int64_t tile_size_index = 0;
  for (const TilingSpace::DimensionInfo& dim : tiling_space.dimensions()) {
    variable_values[num_dimensions + dim.id.value()] =
        dim.type == TilingSpace::DimensionSemantics::kReplica
            ? 1
            : tile_sizes[tile_size_index++];
  }
  return variable_values;
}

// This is a conservative estimate to make sure that we don't get a tile that
// is too big and results in register spills for boundary instructions (roots
// and fusion operands) and iotas.
//
// We had the following reasoning for the value of this constant:
//  * Whenever a block needs to use a tile more than once, it needs to
//    either (1) load the tile from HBM several times, or (2) store the tile
//    in registers at the same time as some of the results. That is the case
//    for normalization diamonds for instance, where the input tile is used
//    twice.
//  * We expect kernels without reuse to benefit from smaller tile sizes
//    anyway.
//  * We use around 20% of the registers as working memory for indexing
//    computations and expensive instructions like exponential or cosine.
//
// This value was empirically determined in September 2024 and may change in
// the future.
constexpr double kFractionOfRegistersEstimatedForBoundaryAndIotaTiles = 0.4;

// Checks if the tile is too large to fit in registers and would result in
// spilling.
//
// Spilling almost always causes significant performance regressions, so this
// heuristic tries to be safe and increase recall at the cost of precision.
bool DoesTileFitInRegisters(int64_t tile_size,
                            const TiledHloInstruction& tiled_hlo,
                            const HloFusionAdaptor& fusion_adaptor,
                            absl::Span<const TiledHloInstruction* const> roots,
                            const se::DeviceDescription& device_info) {
  bool is_root = absl::c_linear_search(roots, &tiled_hlo);
  bool is_operand = !fusion_adaptor.ContainsInstruction(tiled_hlo.hlo());
  // Iota is not an operand, but usually needs to be materialized in
  // registers.
  bool is_iota = tiled_hlo.hlo()->opcode() == HloOpcode::kIota;
  // Roots, fusion operands, and iotas are likely to fully materialize in
  // registers. We use a more relaxed check on intermediate instructions
  // (e.g. needed for broadcasts that expand tensor sizes).
  double fraction_of_registers_available =
      (is_root || is_operand || is_iota)
          ? kFractionOfRegistersEstimatedForBoundaryAndIotaTiles
          : 1.0;

  // Number of 32-bit registers needed to store 1 element of the tile. 64-bit
  // types use two registers per element. Sub-32-bit types are unpacked to one
  // value per register.
  int64_t num_registers_per_element = CeilOfRatio(
      primitive_util::BitWidth(GetFirstShape(tiled_hlo.hlo()).element_type()),
      32);

  // Register allocation happens at PTX->SASS level, so we can't know the exact
  // number of registers used by a kernel. We make a few assumptions about the
  // kernel we will generate (this may not hold in the future):
  //
  //  * We'll need at least 1 register to store 1 element of the tile.
  //  * All values of the tile are live at the same time.
  //  * If all values don't need to be live at the same time (for example to
  //    compute a reduction), it will be modeled by an explicit loop with
  //    smaller tiles inside during tiling propagation.
  //
  // TODO(b/363194951): Check how many registers we need for scratch memory
  // for indexing computation and expensive instructions like exponential or
  // cosine.
  return num_registers_per_element * tile_size <=
         fraction_of_registers_available *
             device_info.registers_per_block_limit();
}

// How the instructions of a region are executed relative to the root of the
// computation.
struct RegionExecutionInfo {
  // Number of blocks, times the trip count of the enclosing sequential loops.
  int64_t num_blocks = 1;
  // Fraction of the read elements that are within the dimension bounds of the
  // enclosing sequential loops. Out-of-bound elements on the tail iteration are
  // masked off and not read from memory.
  double in_bound_read_fraction = 1.0;
};

// Calculates how the instructions of a region are executed, given how the
// tiled HLO that owns the region is executed.
//
// For Dot and Reduce, the block count is scaled UP by the trip count of the
// contracting / reduction loop.
// For Concat, the block count is scaled DOWN to the occupancy fraction of the
// operand.
RegionExecutionInfo GetRegionExecutionInfo(
    const TiledHloInstruction* tiled_hlo,
    const RegionExecutionInfo& info_cur_hlo, int64_t region_id) {
  // Avoids reading tile sizes, which symbolic tilings do not have.
  if (info_cur_hlo.num_blocks == 0) {
    return info_cur_hlo;
  }
  RegionExecutionInfo info_cur_region = info_cur_hlo;
  const HloInstruction* hlo = tiled_hlo->hlo();
  if (HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot,
                       HloOpcode::kReduce>(hlo) &&
      !tiled_hlo->operands().empty()) {
    absl::Span<const int64_t> sequential_dims =
        hlo->opcode() == HloOpcode::kReduce
            ? hlo->dimensions()
            : hlo->dot_dimension_numbers().lhs_contracting_dimensions();
    absl::StatusOr<llvm::SmallVector<int64_t>> operand_tile_sizes =
        tiled_hlo->operand(0)->tile(0).GetStaticTileSizes();
    CHECK_OK(operand_tile_sizes);
    for (int64_t dim : sequential_dims) {
      int64_t dim_size = hlo->operand(0)->shape().dimensions(dim);
      int64_t tile_size = (*operand_tile_sizes)[dim];
      int64_t trip_count = CeilOfRatio(dim_size, tile_size);
      info_cur_region.num_blocks *= trip_count;
      // Read element counts already clamp each tile dimension to `dim_size`, so
      // across `trip_count` iterations the loop counts
      // `trip_count * min(tile_size, dim_size)` elements instead of `dim_size`.
      // Scale down by their ratio to exclude out-of-bound elements masked off
      // on the tail iteration.
      info_cur_region.in_bound_read_fraction *=
          static_cast<double>(dim_size) /
          (trip_count * std::min(tile_size, dim_size));
    }
  } else if (HloPredicateIsOp<HloOpcode::kConcatenate>(hlo)) {
    CHECK_LT(region_id, hlo->operand_count());
    int64_t concat_dim = hlo->concatenate_dimension();
    info_cur_region.num_blocks =
        info_cur_region.num_blocks *
        hlo->operand(region_id)->shape().dimensions(concat_dim) /
        hlo->shape().dimensions(concat_dim);
  }
  return info_cur_region;
}

// Returns how often the root tile is computed: once per output tile, and
// for a root with a sequential dimension (the scan dimension of a tiled
// scan) once per iteration of the loop over that dimension.
int64_t GetNumRootTileExecutions(
    const TiledHloComputation& tiled_hlo_computation) {
  const TiledHloInstruction* root = tiled_hlo_computation.roots().front();
  const Shape& shape = root->hlo()->shape();
  if (shape.IsTuple()) {
    return tiled_hlo_computation.num_output_tiles();
  }
  absl::StatusOr<llvm::SmallVector<int64_t>> tile_sizes =
      root->tile(0).GetStaticTileSizes();
  CHECK_OK(tile_sizes);
  int64_t num_executions = 1;
  for (auto [dim_size, tile_size] :
       llvm::zip(shape.dimensions(), *tile_sizes)) {
    num_executions *= CeilOfRatio(dim_size, tile_size);
  }
  return num_executions;
}

// Visits all tiled HLOs in the computation (including those within regions)
// using a visitor that receives the tiled HLO, how it is executed relative to
// the root, and whether the instruction is within a dot region.
void ForEachInstructionInTiledHloComputation(
    const TiledHloComputation& tiled_hlo_computation,
    int64_t num_blocks_at_root,
    absl::FunctionRef<void(const TiledHloInstruction*,
                           const RegionExecutionInfo&, bool)>
        visitor) {
  struct WorkItem {
    const TiledHloInstruction* instruction;
    RegionExecutionInfo info_cur_hlo;
    bool in_dot_region;
  };
  std::vector<WorkItem> worklist;
  for (const auto* instruction : tiled_hlo_computation.instructions()) {
    worklist.push_back({instruction,
                        RegionExecutionInfo{/*num_blocks=*/num_blocks_at_root},
                        /*in_dot_region=*/false});
  }

  while (!worklist.empty()) {
    auto [instruction, info_cur_hlo, in_dot_region] = worklist.back();
    worklist.pop_back();

    visitor(instruction, info_cur_hlo, in_dot_region);

    bool child_in_dot_region =
        in_dot_region ||
        HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot>(
            instruction->hlo());
    for (const auto& [i, region] :
         llvm::enumerate(instruction->hlo_regions())) {
      RegionExecutionInfo info_cur_region =
          GetRegionExecutionInfo(instruction, info_cur_hlo, i);
      for (const auto& tiled_hlo : region.instructions()) {
        worklist.push_back({&*tiled_hlo, info_cur_region, child_in_dot_region});
      }
    }
  }
}

// Checks if all instructions in `tiled_hlo_computation` fit in registers,
// using `get_padded_tile_size` to compute the padded tile size of each
// instruction.
bool AllInstructionsFitInRegisters(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info,
    absl::FunctionRef<int64_t(const TiledHloInstruction&)>
        get_padded_tile_size) {
  const auto roots = tiled_hlo_computation.roots();
  bool all_fit = true;
  // We set num_blocks_at_root to zero because symbolic tilings have no static
  // tile sizes to count blocks with. Zero makes GetRegionExecutionInfo skip the
  // block count calculation.
  ForEachInstructionInTiledHloComputation(
      tiled_hlo_computation, /*num_blocks_at_root=*/0,
      [&](const TiledHloInstruction* tiled_hlo,
          const RegionExecutionInfo& /*info_cur_hlo*/, bool /*in_dot_region*/) {
        if (!all_fit) {
          return;
        }
        all_fit &=
            DoesTileFitInRegisters(get_padded_tile_size(*tiled_hlo), *tiled_hlo,
                                   fusion_adaptor, roots, device_info);
      });
  return all_fit;
}

// Returns the number of warps to use based on the largest tile size in the
// computation.
//
// This is a simple heuristic and we try to make minimal assumptions about the
// kernel that will be generated by a block-level emitter, but there are a few
// things we take into consideration.
//
// For smaller tile sizes, we pick less warps to make sure there is enough
// elements per thread to have vectorized loads and stores.
//
// For larger tiles, we don't know how many registers will be live at the same
// time and how much shared memory will be used, but there is a good chance that
// only one block will be able to reside on an SM at any given moment.
//
// Choosing 4 or less warps for a large tile will have the following problems:
//
//  * Not all register will be utilized. On H100, for example, there are 64K
//    registers available per SM in total, but there is also a limit of 255
//    registers per thread. To be able to use all available registers we
//    need at least 64K / 255 = 256 threads = 8 warps.
//  * Not enough parallelism to overlap compute and memory access.
//
// Choosing more than 8 warps can also cause performance regressions:
//   * If layout optimizations in a block-level emitter will decide to use
//     shared memory and insert barrier syncs to perform reduction or reduce
//     amount of HBM traffic.
//
// These values and thresholds were empirically determined in November 2024 and
// may change in the future.
// TODO(b/332714755): Make it smarter.
int64_t GetNumWarps(int64_t largest_live_tile_size) {
  if (largest_live_tile_size <= 256) {
    return 1;
  }
  if (largest_live_tile_size <= 1024) {
    return 2;
  }
  if (largest_live_tile_size <= 4096) {
    return 4;
  }
  return 8;
}

// Returns the size of the `tiled_hlo` tile that is used to pick the number of
// warps.
//
// Fusion operand and root tiles count in full. Non-root broadcasts and
// constants are skipped because they materialize as part of their users'
// tiles. A non-root iota counts only its extent along `iota_dimension`, from
// which it is broadcast. Other intermediates are scaled by
// `min(element_bytes, 4) / 4`, a discount for narrow types such as `pred`.
int64_t GetLiveTileSizeForNumWarps(const HloFusionAdaptor& fusion_adaptor,
                                   const TiledHloInstruction* tiled_hlo,
                                   bool is_root) {
  const HloInstruction* hlo = tiled_hlo->hlo();
  absl::StatusOr<llvm::SmallVector<int64_t>> tile_sizes =
      tiled_hlo->tile(0).GetStaticTileSizes();
  CHECK_OK(tile_sizes);
  if (is_root || !fusion_adaptor.ContainsInstruction(hlo)) {
    return GetPaddedTileSize(*tile_sizes);
  }
  if (HloPredicateIsOp<HloOpcode::kBroadcast, HloOpcode::kConstant>(hlo)) {
    return 1;
  }
  int64_t padded_tile_size =
      hlo->opcode() == HloOpcode::kIota
          ? llvm::PowerOf2Ceil(
                (*tile_sizes)[Cast<HloIotaInstruction>(hlo)->iota_dimension()])
          : GetPaddedTileSize(*tile_sizes);
  // Tuple-shaped instructions (e.g. variadic reduce) are not scaled.
  if (!hlo->shape().IsArray()) {
    return padded_tile_size;
  }
  constexpr int64_t kBytesPerRegister = 4;
  int64_t element_bytes = std::min<int64_t>(
      ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type()),
      kBytesPerRegister);
  return CeilOfRatio(padded_tile_size * element_bytes, kBytesPerRegister);
}

int64_t EstimateSharedMemoryUsagePerBlockImpl(
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info,
    absl::FunctionRef<int64_t(const TiledHloInstruction&)>
        get_padded_tile_size) {
  // For transposes, we consider the largest tile size that needs to be
  // materialized in shared memory.
  int64_t transpose_shmem_per_block = 0;

  // For reductions, we assume each warp will store its partial output tile in
  // shared memory.
  //
  // If the reduced tile size is no larger than the warp size, we assume the
  // reduction is performed entirely in registers and does not need shared
  // memory.
  const int64_t warp_size = WarpSize(device_info);
  int64_t largest_live_tile_size = 1;
  int64_t reduce_shmem_bytes_per_warp = 0;

  ForEachInstructionInTiledHloComputation(
      tiled_hlo_computation, /*num_blocks_at_root=*/0,
      [&](const TiledHloInstruction* instruction,
          const RegionExecutionInfo& /*info_cur_hlo*/, bool /*in_dot_region*/) {
        const int64_t padded_tile_size = get_padded_tile_size(*instruction);
        largest_live_tile_size =
            std::max(largest_live_tile_size, padded_tile_size);

        if (instruction->operands().empty()) {
          return;
        }

        const HloInstruction* hlo = instruction->hlo();
        const int64_t tile_bytes =
            padded_tile_size * ShapeUtil::ByteSizeOfPrimitiveType(
                                   GetFirstShape(hlo).element_type());

        switch (hlo->opcode()) {
          case HloOpcode::kTranspose: {
            transpose_shmem_per_block =
                std::max(transpose_shmem_per_block, tile_bytes);
            break;
          }
          case HloOpcode::kReduce: {
            const int64_t operand_tile_size =
                get_padded_tile_size(*instruction->operand(0));
            const int64_t reduced_tile_size =
                operand_tile_size / std::max<int64_t>(1, padded_tile_size);
            if (reduced_tile_size > warp_size) {
              reduce_shmem_bytes_per_warp =
                  std::max(reduce_shmem_bytes_per_warp, tile_bytes);
            }
            break;
          }
          default:
            return;
        }
      });

  const int64_t reduce_shmem_per_block =
      reduce_shmem_bytes_per_warp * GetNumWarps(largest_live_tile_size);

  return std::max(transpose_shmem_per_block, reduce_shmem_per_block);
}

}  // namespace

namespace internal {

bool DoesComputationFitInRegisters(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info) {
  return AllInstructionsFitInRegisters(fusion_adaptor, tiled_hlo_computation,
                                       device_info, GetStaticPaddedTileSize);
}

bool DoesTilingFitInRegisters(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& symbolic_tiled_computation,
    absl::Span<const int64_t> tile_sizes,
    const se::DeviceDescription& device_info) {
  llvm::SmallVector<int64_t> variable_values = GetTilingSpaceVariableValues(
      symbolic_tiled_computation.tiling_space(), tile_sizes);
  return AllInstructionsFitInRegisters(
      fusion_adaptor, symbolic_tiled_computation, device_info,
      [&](const TiledHloInstruction& tiled_hlo) {
        return GetPaddedTileSize(EvaluateTileSizes(tiled_hlo, variable_values));
      });
}

int64_t EstimateSharedMemoryUsagePerBlock(
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info) {
  return EstimateSharedMemoryUsagePerBlockImpl(
      tiled_hlo_computation, device_info, GetStaticPaddedTileSize);
}

bool DoesTilingFitInSharedMemory(
    const TiledHloComputation& symbolic_tiled_computation,
    absl::Span<const int64_t> tile_sizes,
    const se::DeviceDescription& device_info) {
  llvm::SmallVector<int64_t> variable_values = GetTilingSpaceVariableValues(
      symbolic_tiled_computation.tiling_space(), tile_sizes);
  const int64_t shared_memory_usage = EstimateSharedMemoryUsagePerBlockImpl(
      symbolic_tiled_computation, device_info,
      [&](const TiledHloInstruction& tiled_hlo) {
        return GetPaddedTileSize(EvaluateTileSizes(tiled_hlo, variable_values));
      });

  // Having at least 2 blocks per SM allows for hiding latency of block-level
  // synchronization barriers.
  constexpr double kFractionOfSharedMemoryAvailable = 0.5;

  return shared_memory_usage <= kFractionOfSharedMemoryAvailable *
                                    device_info.shared_memory_per_core();
}
}  // namespace internal

namespace {

// Returns the LHS + RHS tile bytes one block consumes over the K loop, at the
// dot's operand types. Clamps tiles larger than the problem.
int64_t CalculateOperandBytesPerBlockAtDotType(
    const HloDotInstruction& dot_instr,
    const dot_model::DotProblemInfo& dot_info,
    const dot_model::DotTileSize& dot_tile,
    absl::Span<const int64_t> batch_tile_sizes) {
  // Dot output dims are [batch..., m, n]; clamp each batch tile to its dim.
  int64_t clamped_b = 1;
  for (auto [tile_size, dim_size] : llvm::zip(
           batch_tile_sizes,
           dot_instr.shape().dimensions().first(batch_tile_sizes.size()))) {
    clamped_b *= std::min(tile_size, dim_size);
  }
  int64_t clamped_m = std::min(dot_tile.m, dot_info.m);
  int64_t clamped_n = std::min(dot_tile.n, dot_info.n);
  // The K loop masks its tail tile, so K is counted exactly once.
  int64_t lhs_bytes = CeilOfRatio<int64_t>(
      clamped_b * clamped_m * dot_info.k *
          primitive_util::BitWidth(dot_info.lhs_element_type),
      8);
  int64_t rhs_bytes = CeilOfRatio<int64_t>(
      clamped_b * dot_info.k * clamped_n *
          primitive_util::BitWidth(dot_info.rhs_element_type),
      8);
  return lhs_bytes + rhs_bytes;
}

// Holds dot resource footprints and mainloop work estimates.
struct DotEstimates {
  PrimitiveType output_element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  int64_t flops_with_wave_quant = 0;
  absl::Duration compute_time = absl::ZeroDuration();
  int64_t k_loop_iterations = 0;
  int64_t shared_memory_per_block_bytes = 0;
  int registers_per_thread = 0;
  // L2 operand bytes over all blocks, counted at the dot's operand types.
  int64_t l2_bytes_at_dot_type = 0;
};

// Combines two dot estimates within the same fusion by summing sequential
// mainloop work and taking the maximum of per-block resource footprints.
DotEstimates MergeDotEstimates(DotEstimates dst, const DotEstimates& src) {
  // TODO(b/495346904): Weight compute utilization by per-dot peak rate when
  // fusing dots with different output element types.
  dst.output_element_type = src.output_element_type;
  dst.flops_with_wave_quant += src.flops_with_wave_quant;
  dst.compute_time += src.compute_time;
  dst.k_loop_iterations += src.k_loop_iterations;
  dst.shared_memory_per_block_bytes = std::max(
      dst.shared_memory_per_block_bytes, src.shared_memory_per_block_bytes);
  dst.registers_per_thread =
      std::max(dst.registers_per_thread, src.registers_per_thread);
  dst.l2_bytes_at_dot_type += src.l2_bytes_at_dot_type;
  return dst;
}

// Computes block launch resource footprints and dot runtime estimates.
absl::StatusOr<DotEstimates> BuildDotEstimates(
    const TiledHloInstruction* tiled_hlo,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_params) {
  const HloDotInstruction* dot_instr =
      Cast<HloDotInstruction>(tiled_hlo->hlo());
  ABSL_RETURN_IF_ERROR(gpu_dot_fusion_cost_model::IsSupported(dot_instr));

  int64_t lhs_contracting_dim =
      dot_instr->dot_dimension_numbers().lhs_contracting_dimensions(0);
  ABSL_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> lhs_tile_sizes,
                   tiled_hlo->operand(0)->tile(0).GetStaticTileSizes());
  CHECK_LT(lhs_contracting_dim, lhs_tile_sizes.size());
  int64_t block_k = lhs_tile_sizes[lhs_contracting_dim];

  ABSL_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> tile_sizes,
                   tiled_hlo->tile(0).GetStaticTileSizes());
  absl::Span<const int64_t> batch_tile_sizes =
      absl::MakeConstSpan(tile_sizes).first(tile_sizes.size() - 2);
  dot_model::DotTileSize dot_tile{
      /*m=*/tile_sizes[tile_sizes.size() - 2],
      /*n=*/tile_sizes.back(),
      /*k=*/block_k,
      /*b=*/Product(batch_tile_sizes),
  };

  dot_model::DotProblemInfo dot_info(*dot_instr);

  ABSL_ASSIGN_OR_RETURN(dot_model::ComputeAndFlops compute_and_flops,
                   dot_model::CalculateComputeTimeWithTileAndWaveQuantization(
                       dot_info, dot_tile, device_info));

  return DotEstimates{
      /*output_element_type=*/dot_info.output_element_type,
      /*flops_with_wave_quant=*/compute_and_flops.flops_with_wave_quant,
      /*compute_time=*/compute_and_flops.compute_time,
      /*k_loop_iterations=*/CeilOfRatio<int64_t>(dot_info.k, block_k),
      /*shared_memory_per_block_bytes=*/
      dot_model::CalculateSharedMemoryPerBlockBytes(dot_info, dot_tile,
                                                    block_params.num_stages),
      /*registers_per_thread=*/
      dot_model::CalculateRegistersPerThread(dot_info, dot_tile, block_params,
                                             device_info),
      /*l2_bytes_at_dot_type=*/
      CalculateOperandBytesPerBlockAtDotType(*dot_instr, dot_info, dot_tile,
                                             batch_tile_sizes),
  };
}

int64_t GetShapeSizeRecursive(
    const Shape& shape, HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return shape_size_fn(shape);
  }

  int64_t total_size = 0;
  for (const auto& element_shape : shape.tuple_shapes()) {
    total_size += GetShapeSizeRecursive(element_shape, shape_size_fn);
  }
  return total_size;
}

// Returns the time to write every fusion output to HBM exactly once. Unlike
// `ComputeReadTime`, there is no cache or occupancy adjustment.
absl::Duration ComputeOutputWriteTime(
    absl::Span<const TiledHloInstruction* const> roots,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    double max_bandwidth_utilization = 1.0) {
  absl::Duration write_time = absl::ZeroDuration();
  for (const TiledHloInstruction* root : roots) {
    // Take the min, not the product, so a transfer-size cap and poor
    // coalescing don't penalize the same bytes twice.
    double utilization =
        std::min(BandwidthUtilizationRateHeuristicForTiledMemoryAccess(
                     *root, device_info),
                 max_bandwidth_utilization);
    double bytes_per_second = utilization * device_info.memory_bandwidth();
    int64_t root_bytes =
        GetShapeSizeRecursive(root->hlo()->shape(), shape_size);
    write_time += absl::Seconds(root_bytes / bytes_per_second);
  }
  return write_time;
}

absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTiledHloComputationImpl(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const BlockLevelParameters& block_level_parameters,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    absl::FunctionRef<int64_t(const HloInstruction*)> flops_per_element_fn) {
  OperandReadMap n_bytes_total_map;
  OperandReadMap dot_region_operand_read_map;
  OperandReadMap epilogue_operand_read_map;

  // Compute time for dot flops is counted separately.
  int64_t dot_flops = 0;
  int64_t flops = 0;
  int64_t bytes_read = 0;
  int64_t dot_l2_bytes_read = 0;
  int64_t num_blocks = tiled_hlo_computation.num_output_tiles();

  DotEstimates dot_estimates;

  // Check if the computation is too large to fit in registers and would result
  // in spilling.
  if (!internal::DoesComputationFitInRegisters(
          fusion_adaptor, tiled_hlo_computation, device_info)) {
    // TODO(b/363194951): Estimate performance regression due to spilling in
    // terms of memory bandwidth instead of returning infinite run time.
    return EstimateRunTimeData::Infinite();
  }

  ForEachInstructionInTiledHloComputation(
      tiled_hlo_computation, GetNumRootTileExecutions(tiled_hlo_computation),
      [&](const TiledHloInstruction* tiled_hlo,
          const RegionExecutionInfo& info_cur_hlo, bool in_dot_region) {
        const HloInstruction* hlo = tiled_hlo->hlo();
        int64_t num_blocks_cur_hlo = info_cur_hlo.num_blocks;
        absl::StatusOr<llvm::SmallVector<int64_t>> tile_sizes =
            tiled_hlo->tile(0).GetStaticTileSizes();
        CHECK_OK(tile_sizes);
        if (fusion_adaptor.ContainsInstruction(hlo)) {
          // Number of elements in the tile after padding.
          int64_t padded_tile_size = GetPaddedTileSize(*tile_sizes);

          // Total number of elements computed for this tile across all blocks.
          //
          // Even if real `tile_size` is smaller than `padded_tile_size`, SM
          // will still perform calculations on masked values, so they should
          // count towards FLOPs.
          int64_t num_elements = num_blocks_cur_hlo * padded_tile_size;

          if (hlo->opcode() == HloOpcode::kDot) {
            if (absl::StatusOr<DotEstimates> cur_dot_estimates =
                    BuildDotEstimates(tiled_hlo, device_info,
                                      block_level_parameters);
                cur_dot_estimates.ok()) {
              // BuildDotEstimates() counts one block; scale to all blocks.
              cur_dot_estimates->l2_bytes_at_dot_type *= num_blocks_cur_hlo;
              dot_estimates =
                  MergeDotEstimates(dot_estimates, *cur_dot_estimates);

              // The dot cost model operates on the tile- and wave- quantized
              // FLOPS which is more accurate for performance estimates but
              // the flops reported by the indexing model are naive algorithm
              // flops.
              // Since the compute time for these flops is already accounted
              // for in dot_estimates.compute_time, we're accumulating it
              // separately from `flops`.
              dot_flops += flops_per_element_fn(hlo) * num_elements;
              return;
            }
          }

          // Tiles inside the computation contribute to the total FLOPs count.
          flops += flops_per_element_fn(hlo) * num_elements;
          return;
        }
        // Calculate the total number of elements that are read from memory
        // across all blocks.
        //
        // Triton requires that all tiles have dimensions that are padded to the
        // next power of 2. However, the load masks the padded elements, so they
        // are not read from memory, but set directly in registers. Generally
        // it's hard to know the amount of masked elements from the tile, but in
        // some cases we can get an estimate. If the tile size is larger than
        // the dimension size, we know that the padded elements are masked. This
        // helps us pick better tile size for cases where we need to get a tile
        // of the full row, for example, when we tile a softmax computation.
        int64_t num_elements = num_blocks_cur_hlo;
        for (auto [tile_size, dimension_size] :
             llvm::zip(*tile_sizes, tiled_hlo->hlo()->shape().dimensions())) {
          num_elements *= std::min(tile_size, dimension_size);
        }

        // Tiles of the operands of the fusion contribute to the total memory
        // read time.
        int64_t element_type_size =
            ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());
        int64_t tile_bytes_read =
            std::llround(element_type_size * num_elements *
                         info_cur_hlo.in_bound_read_fraction);

        bytes_read += tile_bytes_read;
        if (in_dot_region) {
          dot_l2_bytes_read += tile_bytes_read;
        }

        double effective_bandwidth_utilization_rate =
            BandwidthUtilizationRateHeuristicForTiledMemoryAccess(*tiled_hlo,
                                                                  device_info);

        RecordOperandRead(hlo, tile_bytes_read,
                          effective_bandwidth_utilization_rate,
                          n_bytes_total_map);
        RecordOperandRead(hlo, tile_bytes_read,
                          effective_bandwidth_utilization_rate,
                          in_dot_region ? dot_region_operand_read_map
                                        : epilogue_operand_read_map);
      });

  auto roots = tiled_hlo_computation.roots();
  // Output tiles partition each root, so each output byte is written once.
  int64_t bytes_written = 0;
  for (auto* root : roots) {
    bytes_written += GetShapeSizeRecursive(root->hlo()->shape(), shape_size);
  }

  absl::Duration compute_time =
      GpuPerformanceModelBase::ComputeTime(
          device_info, flops, num_blocks,
          block_level_parameters.num_warps * WarpSize(device_info)) +
      dot_estimates.compute_time;

  absl::Duration read_time;
  absl::Duration write_time;
  absl::Duration exec_time;
  double compute_utilization = 0.0;
  double memory_utilization = 0.0;
  const bool has_dot = dot_estimates.k_loop_iterations > 0;
  if (has_dot) {
    int64_t net_bytes_read = GetNetBytesRead(n_bytes_total_map, shape_size);
    int64_t total_net_hbm_bytes = net_bytes_read + bytes_written;
    float effective_hbm_bandwidth =
        dot_model::GetEffectiveHbmBandwidth(total_net_hbm_bytes, device_info);
    // TODO(b/510667327): This fraction folds the table's fixed latency into
    // bandwidth, so reads and writes combine it with coalescing via min. Keep
    // the latency separate and apply coalescing to bandwidth only.
    double hbm_bandwidth_utilization =
        static_cast<double>(effective_hbm_bandwidth) /
        device_info.memory_bandwidth();
    // Exclude K-loop cache re-reads because `CalculateL2Time` models L2
    // traffic separately.
    absl::Duration dot_dram_read_time = ComputeReadTime(
        dot_region_operand_read_map, device_info, num_blocks, shape_size,
        /*include_cache_rereads=*/false, hbm_bandwidth_utilization);
    absl::Duration epilogue_read_time = ComputeReadTime(
        epilogue_operand_read_map, device_info, num_blocks, shape_size,
        /*include_cache_rereads=*/true, hbm_bandwidth_utilization);
    read_time = dot_dram_read_time + epilogue_read_time;
    write_time = ComputeOutputWriteTime(roots, device_info, shape_size,
                                        hbm_bandwidth_utilization);

    dot_model::LaunchConfig launch_config{
        /*num_stages=*/block_level_parameters.num_stages,
        /*num_warps=*/block_level_parameters.num_warps,
        /*k_loop_iterations=*/dot_estimates.k_loop_iterations,
        /*threadblock_count=*/num_blocks,
        /*shared_memory_per_block_bytes=*/
        dot_estimates.shared_memory_per_block_bytes,
        /*registers_per_thread=*/dot_estimates.registers_per_thread,
    };
    absl::Duration post_mainloop_time = epilogue_read_time + write_time;
    absl::Duration mainloop_time =
        dot_model::CalculatePipelinedLoopTimeWithLaunchWaves(
            launch_config, dot_estimates.compute_time,
            dot_model::HbmEstimates{dot_dram_read_time, absl::ZeroDuration()},
            device_info);
    // The epilogue (operand reads and output writes) runs after the K loop and
    // doesn't overlap with it.
    absl::Duration pipelined_loop_time = mainloop_time + post_mainloop_time;
    // Widening prologues (e.g. s8 -> bf16) also pay for an in-register convert
    // in the K loop, which we don't model yet. As a proxy, floor L2 traffic at
    // the dot's operand types, even though Triton loads the narrow type. The
    // floor also applies to other prologues (broadcasts, chained-dot operands).
    // TODO(b/581269931): Model the convert cost and drop this floor.
    absl::Duration l2_time = dot_model::CalculateL2Time(
        dot_estimates.k_loop_iterations, device_info,
        std::max(dot_l2_bytes_read, dot_estimates.l2_bytes_at_dot_type),
        block_level_parameters.is_tma_allowed);
    exec_time = std::max(pipelined_loop_time, l2_time);

    // Roofline utilization uses wave-quantized FLOPs and net DRAM traffic.
    EstimateRunTimeData roofline_estimates = EstimateRunTimeData::Zero();
    roofline_estimates.flops = dot_estimates.flops_with_wave_quant + flops;
    roofline_estimates.bytes_read = net_bytes_read;
    roofline_estimates.bytes_written = bytes_written;
    roofline_estimates.exec_time = exec_time;
    compute_utilization = dot_model::CalculateComputeUtilization(
        roofline_estimates, device_info, dot_estimates.output_element_type);
    memory_utilization =
        dot_model::CalculateMemoryUtilization(roofline_estimates, device_info);
  } else {
    read_time =
        ComputeReadTime(n_bytes_total_map, device_info, num_blocks, shape_size);
    write_time = ComputeOutputWriteTime(roots, device_info, shape_size);
    exec_time = GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
        compute_time, read_time + write_time);
  }

  const int64_t estimated_shared_memory_per_block =
      has_dot ? dot_estimates.shared_memory_per_block_bytes
              : internal::EstimateSharedMemoryUsagePerBlock(
                    tiled_hlo_computation, device_info);

  return EstimateRunTimeData{
      /*flops=*/flops + dot_flops,
      /*bytes_read=*/bytes_read,
      /*bytes_written=*/bytes_written,
      /*read_time=*/read_time,
      /*write_time=*/write_time,
      /*compute_time=*/compute_time,
      /*exec_time=*/exec_time,
      /*l2_bytes_read=*/dot_l2_bytes_read,
      /*shared_memory_per_block_bytes=*/estimated_shared_memory_per_block,
      /*registers_per_thread=*/dot_estimates.registers_per_thread,
      /*compute_utilization=*/compute_utilization,
      /*memory_utilization=*/memory_utilization};
}

// TODO(b/534676634): Support multi-reduction and multi-dimension reduction
// fusions. When supporting multi-reduction fusions, `sequential_tile_sizes`
// must be ordered in the reverse post-order that the merged fusion computation
// will have after `MergeFusionInstruction` (e.g., via a post-order DFS from
// `fusion_adaptor.GetRoots()` following
// `HloInstructionAdaptor::GetOperands()`), because
// `HloFusionAdaptor::MakeInstructionPostOrder()` on a producer-consumer adaptor
// concatenates the producer's post-order before the consumer's post-order,
// which can differ from the post-order of the merged computation.
std::vector<int64_t> GetSequentialTileSizes(
    const TiledHloComputation& tiled_hlo_computation) {
  const TilingSpace& tiling_space = tiled_hlo_computation.tiling_space();
  std::vector<int64_t> sequential_tile_sizes;
  bool all_dimensions_untiled = true;
  for (const auto& dim : tiling_space.dimensions()) {
    if (dim.type == TilingSpace::DimensionSemantics::kSequential &&
        dim.hlo->opcode() == HloOpcode::kReduce) {
      CHECK(dim.tile_size.has_value()) << dim.ToString();
      sequential_tile_sizes.push_back(*dim.tile_size);
      all_dimensions_untiled &= *dim.tile_size >= dim.dimension_size;
    }
  }
  // An empty result means that the sequential dimensions are not tiled. Keep it
  // empty when every sequential dimension is covered by a single tile, so that
  // the config of such tilings is the same as before reduction tiling.
  if (all_dimensions_untiled || sequential_tile_sizes.size() != 1) {
    return {};
  }
  return sequential_tile_sizes;
}

// Returns block-level parameters with the tile sizes that reproduce the tiling
// of `tiled_hlo_computation`.
BlockLevelParameters BlockLevelParametersForTiling(
    const TiledHloComputation& tiled_hlo_computation) {
  BlockLevelParameters block_level_parameters;
  // The scan dimension is part of the root shape but not a parallel
  // dimension, so its tile size goes to `sequential_tile_sizes`.
  const auto& dimensions = tiled_hlo_computation.tiling_space().dimensions();
  const bool contains_scan =
      absl::c_any_of(dimensions, [](const TilingSpace::DimensionInfo& dim) {
        return dim.type == TilingSpace::DimensionSemantics::kSequential &&
               dim.hlo != nullptr && dim.hlo->opcode() == HloOpcode::kScan;
      });
  if (contains_scan) {
    std::vector<int64_t> parallel_tile_sizes;
    std::vector<int64_t> sequential_tile_sizes;
    for (const TilingSpace::DimensionInfo& dim : dimensions) {
      if (dim.type == TilingSpace::DimensionSemantics::kParallel) {
        parallel_tile_sizes.push_back(*dim.tile_size);
      } else if (dim.type == TilingSpace::DimensionSemantics::kSequential) {
        sequential_tile_sizes.push_back(*dim.tile_size);
      }
    }
    block_level_parameters.output_tile_sizes.resize(
        tiled_hlo_computation.roots().size(), parallel_tile_sizes);
    block_level_parameters.sequential_tile_sizes =
        std::move(sequential_tile_sizes);
    return block_level_parameters;
  }

  block_level_parameters.output_tile_sizes.reserve(
      tiled_hlo_computation.roots().size());
  for (const TiledHloInstruction* tiled_root : tiled_hlo_computation.roots()) {
    absl::StatusOr<llvm::SmallVector<int64_t>> tile_sizes =
        tiled_root->tile(0).GetStaticTileSizes();
    CHECK_OK(tile_sizes);
    block_level_parameters.output_tile_sizes.emplace_back(tile_sizes->begin(),
                                                          tile_sizes->end());
  }
  block_level_parameters.sequential_tile_sizes =
      GetSequentialTileSizes(tiled_hlo_computation);
  return block_level_parameters;
}

absl::StatusOr<std::optional<TiledRunTimeData>> EstimateTiledRunTimeDataImpl(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    absl::FunctionRef<int64_t(const HloInstruction*)> flops_per_element_fn) {
  BlockLevelParameters block_level_parameters =
      BlockLevelParametersForTiling(tiled_hlo_computation);
  block_level_parameters.num_warps =
      GpuPerformanceModelWithIndexingAnalysis::EstimateNumWarps(
          fusion_adaptor, tiled_hlo_computation);

  ABSL_ASSIGN_OR_RETURN(
      EstimateRunTimeData estimate_run_time_data,
      EstimateRunTimeForTiledHloComputationImpl(
          fusion_adaptor, tiled_hlo_computation, block_level_parameters,
          device_info, shape_size, flops_per_element_fn));

  // Skip tilings with infinite runtime (e.g., due to register spilling).
  if (estimate_run_time_data.exec_time == absl::InfiniteDuration()) {
    return std::nullopt;
  }

  return TiledRunTimeData{estimate_run_time_data, block_level_parameters};
}

// Returns the candidate tile sizes for a single-dimension reduction `dim`,
// pruning sequential tile sizes that are unlikely to be profitable to reduce
// search time. This is an empirical heuristic rather than a strict
// guarantee—the pruned tilings are not deterministically worse in all cases,
// but the underlying intuition holds well across current benchmarks and model
// workloads.
absl::StatusOr<std::vector<int64_t>> PruneUnprofitableReductionTileSizes(
    const TilingSpace::DimensionInfo& dim) {
  const Shape& operand_shape = dim.hlo->operand(0)->shape();
  const int64_t reduction_size = dim.dimension_size;
  const int64_t minor_dim = LayoutUtil::Minor(operand_shape.layout(), 0);
  const bool is_contiguous = dim.hlo->dimensions(0) == minor_dim;

  if (is_contiguous) {
    // For a contiguous reduction, small power-of-two dimensions (<= 256)
    // already have coalesced reads and no power-of-two padding waste when
    // untiled. Non-power-of-two dimensions in (128, 256) pad to 256 when
    // untiled (wasting up to nearly half the tile), so tiling them remains
    // worthwhile. When tiled, tiles < 16 under-utilize a warp and add loop
    // overhead without improving coalescing.
    if (reduction_size > 256 ||
        (reduction_size > 128 &&
         !llvm::isPowerOf2_64(static_cast<uint64_t>(reduction_size)))) {
      return GetPowerOf2TileSizesInRange(reduction_size,
                                         /*min_tile_size=*/16);
    }
  } else {
    // For a strided reduction, tiling the reduction dimension lowers register
    // pressure so the minor (parallel) dimension can use a larger tile for
    // coalesced reads. Keep reduction tilings only when the reduction
    // dimension is large enough to constrain registers and the minor
    // dimension is large enough to benefit from a wider tile.
    const int64_t minor_dim_size = operand_shape.dimensions(minor_dim);
    if (reduction_size > 256 ||
        (reduction_size >= 64 && minor_dim_size >= 128)) {
      return GetPowerOf2TileSizesInRange(reduction_size);
    }
  }
  return std::vector<int64_t>{reduction_size};
}

// Whether the tiling search for `fusion_adaptor` should enumerate tile sizes
// for the reduction dimensions, i.e. consider emitting reductions as loops.
bool ShouldTileReductionDimensions(const HloFusionAdaptor& fusion_adaptor) {
  const HloModule* module =
      fusion_adaptor.GetRoots().front().instruction().GetModule();
  return module != nullptr &&
         module->config()
             .debug_options()
             .xla_gpu_experimental_enable_reduction_dimension_tiling();
}

std::string GetFusionRootName(const HloFusionAdaptor& fusion_adaptor) {
  return fusion_adaptor.GetRoots().front().instruction().ToString(
      HloPrintOptions::ShortParsable());
}

absl::Status DumpAndLogTopKCandidates(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const TiledRunTimeData> candidates) {
  const HloInstruction& root = fusion_adaptor.GetRoots().front().instruction();
  std::string dump_path;
  if (const HloModule* module = root.GetModule()) {
    dump_path = module->config()
                    .debug_options()
                    .xla_gpu_dump_cost_model_top_k_candidates_to();
  }

  if (dump_path.empty()) {
    return absl::OkStatus();
  }

  const std::string fusion_name = GetFusionRootName(fusion_adaptor);

  std::string dump_content;
  absl::StrAppend(&dump_content, "=== Cost model top-", candidates.size(),
                  " candidates for: ", fusion_name, " ===\n");
  for (int i = 0; i < candidates.size(); ++i) {
    absl::StrAppend(&dump_content, "Candidate #", i, ": ", candidates[i], "\n");
  }

  absl::StrAppend(&dump_content, "\n");
  std::string resolved_path;
  if (!tsl::io::ResolveTestPrefixes(dump_path, resolved_path)) {
    return FailedPrecondition("Failed to resolve cost model dump path: %s",
                              dump_path);
  }
  tsl::Env* env = tsl::Env::Default();
  ABSL_RETURN_IF_ERROR(tsl::AppendStringToFile(env, resolved_path, dump_content));

  return absl::OkStatus();
}

absl::StatusOr<absl::InlinedVector<TiledRunTimeData, 4>>
FilterAndSortCandidates(
    std::vector<std::optional<TiledRunTimeData>> raw_candidates, int top_k,
    const HloFusionAdaptor& fusion_adaptor) {
  absl::InlinedVector<TiledRunTimeData, 4> candidates;
  for (auto& res : raw_candidates) {
    if (res.has_value()) {
      candidates.push_back(std::move(*res));
    }
  }
  VLOG(1) << absl::StrCat("Found ", candidates.size(),
                          " valid tiling candidates.");

  absl::c_stable_sort(
      candidates, [](const TiledRunTimeData& a, const TiledRunTimeData& b) {
        if (a.runtime_data.exec_time != b.runtime_data.exec_time) {
          return a.runtime_data.exec_time < b.runtime_data.exec_time;
        }
        // Break exact ties towards larger contracting dimension tiles
        // (currently reductions and scans), if they exist, i.e. fewer
        // iterations of the sequential loop, whose per-iteration overhead the
        // model does not charge.
        return Product(a.block_level_parameters.sequential_tile_sizes) >
               Product(b.block_level_parameters.sequential_tile_sizes);
      });
  if (VLOG_IS_ON(2)) {
    const std::string fusion_name = GetFusionRootName(fusion_adaptor);
    for (int i = 0; i < candidates.size(); ++i) {
      if (i < top_k) {
        VLOG(2) << "[" << fusion_name << "] Candidate #" << i << ": "
                << candidates[i];
      } else {
        VLOG(3) << "[" << fusion_name << "] Candidate #" << i << ": "
                << candidates[i];
      }
    }
  }
  if (candidates.size() > top_k) {
    candidates.resize(top_k);
  }
  ABSL_RETURN_IF_ERROR(DumpAndLogTopKCandidates(fusion_adaptor, candidates));
  return candidates;
}

struct BaseTilingSpace {
  PooledOrFallbackMlirContext context;
  std::unique_ptr<TilingSpace> tiling_space;
};

// Evaluates a single tiling candidate: clones `base->tiling_space`, assigns
// `padded_tile_sizes` to the clone, tiles the fusion and estimates its run
// time. Returns std::nullopt if the candidate cannot be tiled or if it
// violates Triton constraints.
//
// If `mlir_context_pool` is not null, the clone is rebound to a context
// borrowed from the pool, so that concurrent evaluations do not contend on a
// single mlir::MLIRContext.
absl::StatusOr<std::optional<TiledRunTimeData>> EvaluateCandidate(
    const HloFusionAdaptor& fusion_adaptor,
    std::shared_ptr<const BaseTilingSpace> base,
    absl::Span<const int64_t> padded_tile_sizes,
    MlirContextPool* mlir_context_pool,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    absl::FunctionRef<int64_t(const HloInstruction*)> flops_fn) {
  // Cloning is faster than calling TilingSpace::Create() for each tiling
  // candidate.
  ABSL_ASSIGN_OR_RETURN(PooledOrFallbackMlirContext context,
                   BorrowMlirContextOr(mlir_context_pool,
                                       base->tiling_space->mlir_context()));
  std::unique_ptr<TilingSpace> tiling_space =
      base->tiling_space->Clone(context.get());
  ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(padded_tile_sizes));
  VLOG(3) << "Trying tile sizes " << absl::StrJoin(padded_tile_sizes, ",");

  const absl::StatusOr<TiledHloComputation> tiled_computation =
      TiledHloComputation::Tile(fusion_adaptor, std::move(tiling_space));
  if (!tiled_computation.ok()) {
    // TODO: b/511080616 - GetValidTilings() must return only tilings that can
    // be tiled and we should treat all errors here as a failure.
    VLOG(3) << "[" << GetFusionRootName(fusion_adaptor)
            << "] Tiling failed for " << absl::StrJoin(padded_tile_sizes, ",")
            << " with error: " << tiled_computation.status().message();
    return std::nullopt;
  }
  if (const Decision valid =
          VerifyTritonConstraints(*tiled_computation, device_info);
      !valid) {
    VLOG(3) << "[" << GetFusionRootName(fusion_adaptor)
            << "] Triton constraints violated for tiling "
            << absl::StrJoin(padded_tile_sizes, ",") << ": " << valid.Explain();
    return std::nullopt;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::optional<TiledRunTimeData> tiled_run_time_data,
      EstimateTiledRunTimeDataImpl(fusion_adaptor, *tiled_computation,
                                   device_info, shape_size, flops_fn));

  if (tiled_run_time_data.has_value()) {
    VLOG(2) << "Accepted tile sizes ["
            << absl::StrJoin(tiled_run_time_data->block_level_parameters
                                 .output_tile_sizes.front(),
                             ",")
            << "], sequential tile sizes ["
            << absl::StrJoin(tiled_run_time_data->block_level_parameters
                                 .sequential_tile_sizes,
                             ",")
            << "], exec_time " << tiled_run_time_data->runtime_data.exec_time;
  } else {
    VLOG(3) << "[" << GetFusionRootName(fusion_adaptor)
            << "] Rejected tile sizes " << absl::StrJoin(padded_tile_sizes, ",")
            << ": exceeds post-tiling register limit";
  }
  return tiled_run_time_data;
}

tsl::Future<TopKTiledRunTimeDataOrError> TryFindTopKBestTilingsWithTilingSpace(
    const HloFusionAdaptor& fusion_adaptor, int top_k, tsl::Executor* executor,
    std::function<int64_t(const HloInstruction*)> flops_fn,
    mlir::MLIRContext* mlir_context, MlirContextPool* mlir_context_pool,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size) {
  CHECK(executor == &tsl::InlineExecutor::Instance() ||
        mlir_context_pool != nullptr || mlir_context->isMultithreadingEnabled())
      << "Concurrent tiling evaluation requires either an MlirContextPool or "
         "an MLIRContext with multithreading enabled.";

  // When a context pool is provided, borrow a context for the base TilingSpace
  // as well, since this function may be called concurrently with the same
  // single-threaded `mlir_context`. Candidates clone from the base TilingSpace
  // asynchronously, so the context stays borrowed until the last candidate
  // releases `base`.
  ABSL_ASSIGN_OR_RETURN(PooledOrFallbackMlirContext context,
                   BorrowMlirContextOr(mlir_context_pool, mlir_context));
  auto base = std::make_shared<BaseTilingSpace>(
      BaseTilingSpace{std::move(context), nullptr});
  ABSL_ASSIGN_OR_RETURN(base->tiling_space,
                   TilingSpace::Create(fusion_adaptor, base->context.get()));

  // TODO(b/534676634): Lift `num_reduction_dims == 1` once multi-reduction and
  // multi-dimension reduction fusions are supported.
  const int64_t num_reduction_dims = absl::c_count_if(
      base->tiling_space->dimensions(),
      [](const TilingSpace::DimensionInfo& dim) {
        return dim.type == TilingSpace::DimensionSemantics::kSequential &&
               dim.hlo->opcode() == HloOpcode::kReduce;
      });
  const bool tile_reduction_dims =
      ShouldTileReductionDimensions(fusion_adaptor) && num_reduction_dims == 1;
  ABSL_ASSIGN_OR_RETURN(
      auto tilings,
      base->tiling_space->GetValidTilings(
          [tile_reduction_dims](const TilingSpace::DimensionInfo& dim)
              -> absl::StatusOr<std::vector<int64_t>> {
            if (dim.type == TilingSpace::DimensionSemantics::kSequential) {
              if (dim.hlo->opcode() == HloOpcode::kReduce) {
                if (!tile_reduction_dims) {
                  return std::vector<int64_t>{dim.dimension_size};
                }
                return PruneUnprofitableReductionTileSizes(dim);
              }
              if (dim.hlo->opcode() == HloOpcode::kScan &&
                  dim.dimension_size <= kMaxRowLocalScanDimSize) {
                return std::vector<int64_t>{dim.dimension_size};
              }
            }
            return GetPowerOf2TileSizesInRange(dim.dimension_size);
          }));
  VLOG(1) << absl::StrCat(
      "TryFindTopKBestTilingsForFusionAsync tiling_space evaluating ",
      tilings.size(), " tilings.");
  VLOG(2) << "TilingSpace for " << GetFusionRootName(fusion_adaptor) << ":\n"
          << *base->tiling_space;

  // Tile the fusion symbolically once, to reject candidates whose tiles do not
  // fit in registers before tiling them. If symbolic tiling fails, there are no
  // tiles to check and all candidates are tiled.
  const absl::StatusOr<TiledHloComputation> symbolic_tiled_computation =
      TiledHloComputation::Tile(fusion_adaptor, base->tiling_space->Clone());

  std::vector<tsl::Future<std::optional<TiledRunTimeData>>> candidate_futures;
  candidate_futures.reserve(tilings.size());
  for (const llvm::SmallVector<int64_t, 4>& tiling : tilings) {
    // Assign padded tile size as Triton emitter will require that.
    llvm::SmallVector<int64_t, 4> padded_tile_sizes =
        xla::xtile::GetPaddedTileSizes(tiling);

    // Early pruning invalid candidates as an optimization.
    if (Decision decision = VerifySubsetOfTritonConstraints(
            padded_tile_sizes, *base->tiling_space, device_info);
        !decision) {
      VLOG(3) << "[" << GetFusionRootName(fusion_adaptor)
              << "] Pre-filtering candidate violating Triton constraints: "
              << absl::StrJoin(padded_tile_sizes, ",") << ": "
              << decision.Explain();
      continue;
    }

    if (symbolic_tiled_computation.ok() &&
        !internal::DoesTilingFitInRegisters(fusion_adaptor,
                                            *symbolic_tiled_computation,
                                            padded_tile_sizes, device_info)) {
      VLOG(3) << "[" << GetFusionRootName(fusion_adaptor)
              << "] Pre-filtering candidate exceeding register limit: "
              << absl::StrJoin(padded_tile_sizes, ",");
      continue;
    }

    if (symbolic_tiled_computation.ok() &&
        !internal::DoesTilingFitInSharedMemory(
            *symbolic_tiled_computation, padded_tile_sizes, device_info)) {
      VLOG(3) << "Pre-filtering candidate exceeding shared memory limit: "
              << absl::StrJoin(padded_tile_sizes, ",");
      continue;
    }

    candidate_futures.push_back(tsl::MakeFutureOn(
        *executor,
        [base, &fusion_adaptor, device_info = &device_info, shape_size,
         flops_fn, mlir_context_pool,
         padded_tile_sizes = std::move(padded_tile_sizes)]() mutable {
          return EvaluateCandidate(fusion_adaptor, std::move(base),
                                   padded_tile_sizes, mlir_context_pool,
                                   *device_info, shape_size, flops_fn);
        }));
  }

  if (candidate_futures.empty()) {
    VLOG(1) << "Found 0 valid tiling candidates.";
    return TopKTiledRunTimeDataOrError(
        absl::InlinedVector<TiledRunTimeData, 4>{});
  }

  return tsl::JoinFutures(absl::MakeSpan(candidate_futures))
      .Map(*executor,
           [&fusion_adaptor,
            top_k](std::vector<std::optional<TiledRunTimeData>> results)
               -> absl::StatusOr<TopKTiledRunTimeDataOrError> {
             ABSL_ASSIGN_OR_RETURN(auto candidates,
                              FilterAndSortCandidates(std::move(results), top_k,
                                                      fusion_adaptor));
             return TopKTiledRunTimeDataOrError(std::move(candidates));
           });
}

}  // namespace

int64_t GpuPerformanceModelWithIndexingAnalysis::FlopsPerElement(
    const HloInstruction* instr) {
  // Instruction that are only used for indexing are not counted for FLOPs.
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConstant:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return 0;
    case HloOpcode::kConcatenate: {
      // Same as GpuHloCostAnalysis::HandleConcatenate.
      auto* concat = Cast<HloConcatenateInstruction>(instr);
      int64_t dim = concat->concatenate_dimension();
      if (dim > 0 && concat->operand(0)->shape().dimensions()[dim] & 31) {
        return 400;
      }
      return 6;
    }
    case HloOpcode::kScan: {
      int64_t flops_per_scan_computation = 0;
      for (const HloInstruction* scan_instr :
           instr->called_computations()[0]->instructions()) {
        flops_per_scan_computation += FlopsPerElement(scan_instr);
      }
      // A scan produces one element per combiner application.
      return flops_per_scan_computation;
    }
    default:
      break;
  };

  // Get the FLOPs per element for elementwise operations that only depend on
  // the element type.
  if (instr->IsElementwise()) {
    return cost_analysis_.GetFlopsPerElementwiseOpElement(
        instr->shape().element_type(), instr->opcode());
  }

  if (instr->opcode() == HloOpcode::kReduce) {
    int64_t flops_per_reduce_computation = 0;
    for (const HloInstruction* reducer_instr :
         instr->called_computations()[0]->instructions()) {
      flops_per_reduce_computation += FlopsPerElement(reducer_instr);
    }

    auto operand_shape = instr->operand(0)->shape();

    // Size of reduction dimensions padded to power of 2.
    int64_t padded_reduction_factor = 1;
    for (int64_t dim : instr->dimensions()) {
      padded_reduction_factor *=
          llvm::PowerOf2Ceil(operand_shape.dimensions(dim));
    }

    // The Cost Model assumes that the reduction computation is applied N-1
    // times to reduce N elements. This is not true, because emitters will
    // generate a loop with N iterations. Also note that this calculation now
    // differs from `GpuHloCostAnalysis` because it accounts for power-of-two
    // padding of the reduction dimensions.
    return (padded_reduction_factor - 1) * flops_per_reduce_computation;
  }

  // Encountered unexpected instruction, call into `GpuHloCostAnalysis`.
  CHECK_OK(cost_analysis_.RevisitInstruction(instr));

  return cost_analysis_.flop_count(*instr) /
         ShapeUtil::ElementsInRecursive(instr->shape());
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledHloComputation(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const BlockLevelParameters& block_level_parameters) {
  return EstimateRunTimeForTiledHloComputationImpl(
      fusion_adaptor, tiled_hlo_computation, block_level_parameters,
      *device_info_, shape_size_,
      [this](const HloInstruction* hlo) { return FlopsPerElement(hlo); });
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledFusion(
    const HloFusionAdaptor& fusion_adaptor,
    const BlockLevelParameters& block_level_parameters) {
  ABSL_ASSIGN_OR_RETURN(PooledOrFallbackMlirContext context,
                   BorrowMlirContextOr(mlir_context_pool_, mlir_context_));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<TilingSpace> tiling_space,
                   TilingSpace::Create(fusion_adaptor, context.get()));

  ABSL_ASSIGN_OR_RETURN(
      llvm::SmallVector<int64_t> tile_sizes,
      GetTilingSpaceConcreteSizes(*tiling_space, block_level_parameters,
                                  enable_same_shape_multi_output_fusion_));
  // Assign padded tile size as that what is a Triton emitter requirement.
  ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(
      xla::xtile::GetPaddedTileSizes(tile_sizes)));

  ABSL_ASSIGN_OR_RETURN(
      TiledHloComputation tiled_hlo_computation,
      TiledHloComputation::Tile(fusion_adaptor, std::move(tiling_space)));
  return EstimateRunTimeForTiledHloComputationImpl(
      fusion_adaptor, tiled_hlo_computation, block_level_parameters,
      *device_info_, shape_size_,
      [&](const HloInstruction* hlo) { return FlopsPerElement(hlo); });
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTriton(
    const HloInstruction* instr,
    const BlockLevelParameters* block_level_parameters) {
  const auto& fusion_analysis = fusion_analysis_cache_->Get(*instr);

  if (block_level_parameters != nullptr) {
    return EstimateRunTimeForTiledFusion(fusion_analysis.fusion(),
                                         *block_level_parameters);
  }

  auto launch_config = TritonFusion::GetLaunchConfig(&fusion_analysis);
  if (!launch_config.has_value()) {
    return absl::InvalidArgumentError(
        "Could not get launch config for Triton fusion.");
  }
  return EstimateRunTimeForTiledFusion(fusion_analysis.fusion(),
                                       launch_config->block_level_parameters);
}

/*static*/
int64_t GpuPerformanceModelWithIndexingAnalysis::EstimateNumWarps(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation) {
  // Decide on the number of warps to use based on the largest live tile size
  // at any given point within the computation.
  int64_t largest_live_tile_size = 1;
  const auto roots = tiled_hlo_computation.roots();
  ForEachInstructionInTiledHloComputation(
      tiled_hlo_computation, /*num_blocks_at_root=*/1,
      [&](const TiledHloInstruction* tiled_hlo,
          const RegionExecutionInfo& /*info_cur_hlo*/, bool /*in_dot_region*/) {
        bool is_root = absl::c_linear_search(roots, tiled_hlo);
        largest_live_tile_size = std::max(
            largest_live_tile_size,
            GetLiveTileSizeForNumWarps(fusion_adaptor, tiled_hlo, is_root));
      });
  return GetNumWarps(largest_live_tile_size);
}

namespace internal {

absl::flat_hash_map<const HloInstruction*, int64_t> PrecomputeFlopsMap(
    const HloFusionAdaptor& fusion_adaptor,
    absl::FunctionRef<int64_t(const HloInstruction*)> flops_per_element_fn) {
  auto post_order = fusion_adaptor.MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*, int64_t> flops_map;
  flops_map.reserve(post_order.size());
  for (const HloInstructionAdaptor& instruction_adaptor : post_order) {
    flops_map.emplace(&instruction_adaptor.instruction(),
                      flops_per_element_fn(&instruction_adaptor.instruction()));
  }
  return flops_map;
}

}  // namespace internal

tsl::Future<TopKTiledRunTimeDataOrError>
GpuPerformanceModelWithIndexingAnalysis::TryFindTopKBestTilingsForFusionAsync(
    const HloFusionAdaptor& fusion_adaptor, int top_k,
    tsl::Executor* executor) {
  XLA_SCOPED_LOGGING_TIMER(
      "GpuPerformanceModelWithIndexingAnalysis::"
      "TryFindTopKBestTilingsForFusionAsync");
  if (!fusion_adaptor.GetRoots().empty() &&
      fusion_adaptor.GetRoots()[0].instruction().parent() != nullptr) {
    const HloInstruction& root = fusion_adaptor.GetRoots()[0].instruction();
    VLOG(1) << "TryFindTopKBestTilingsForFusionAsync adaptor root "
            << root.ToString(HloPrintOptions::ShortParsable()) << " parent "
            << root.parent()->name();
  }

  if (executor == nullptr) {
    executor = &tsl::InlineExecutor::Instance();
  }

  auto flops_map = std::make_shared<
      const absl::flat_hash_map<const HloInstruction*, int64_t>>(
      internal::PrecomputeFlopsMap(
          fusion_adaptor,
          [this](const HloInstruction* hlo) { return FlopsPerElement(hlo); }));

  auto flops_fn = [flops_map](const HloInstruction* hlo) -> int64_t {
    if (hlo->opcode() == HloOpcode::kParameter ||
        hlo->opcode() == HloOpcode::kTuple ||
        hlo->opcode() == HloOpcode::kGetTupleElement) {
      return 0;
    }
    auto it = flops_map->find(hlo);
    CHECK(it != flops_map->end())
        << "Missing precomputed FLOPs for " << hlo->name();
    return it->second;
  };

  return TryFindTopKBestTilingsWithTilingSpace(
      fusion_adaptor, top_k, executor, flops_fn, mlir_context_,
      mlir_context_pool_, *device_info_, shape_size_);
}

tsl::Future<TiledRunTimeDataOrError>
GpuPerformanceModelWithIndexingAnalysis::TryFindBestTilingForFusionAsync(
    const HloFusionAdaptor& fusion_adaptor, tsl::Executor* executor) {
  if (executor == nullptr) {
    executor = &tsl::InlineExecutor::Instance();
  }
  return TryFindTopKBestTilingsForFusionAsync(fusion_adaptor, /*top_k=*/1,
                                              executor)
      .Map(*executor,
           [&fusion_adaptor](const TopKTiledRunTimeDataOrError& top_k_result)
               -> TiledRunTimeDataOrError {
             if (std::holds_alternative<FusionDecision>(top_k_result)) {
               return std::get<FusionDecision>(top_k_result);
             }
             const auto& tilings =
                 std::get<absl::InlinedVector<TiledRunTimeData, 4>>(
                     top_k_result);
             if (tilings.empty()) {
               return FusionDecision::Forbid("No valid tilings found.");
             }
             VLOG(1)
                 << "TryFindBestTilingForFusionAsync "
                 << fusion_adaptor.GetRoots().front().instruction().ToString(
                        HloPrintOptions::ShortParsable())
                 << " best output_tile_sizes ["
                 << absl::StrJoin(
                        tilings.front()
                            .block_level_parameters.output_tile_sizes.front(),
                        ",")
                 << "], exec_time " << tilings.front().runtime_data.exec_time;
             return tilings.front();
           });
}

}  // namespace gpu
}  // namespace xla
