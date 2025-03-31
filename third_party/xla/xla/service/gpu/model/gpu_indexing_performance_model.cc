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
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

// Information about an operand read.
struct OperandReadInfo {
  // Total number of bytes read from the operand.
  int64_t total_bytes_read = 0;

  // Factor, between 0 and 1, determining how much of the chip's HBM bandwidth
  // is actually attained when reading this operand.
  double read_bandwidth_utilization_rate = 1.0;
};

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

// Checks if the tile is too large to fit in registers and would result in
// spilling.
//
// Spilling almost always causes significant performance regressions, so this
// heuristic tries to be safe and increase recall at the cost of precision.
bool DoesTileFitsInRegisters(int64_t tile_size,
                             const se::DeviceDescription& device_info) {
  // This is a conservative estimate to make sure that we don't get a tile that
  // is too big and results in register spills.
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
  constexpr double kFractionOfRegistersAvailableToStoreTile = 0.4;

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
  //
  // TODO(b/363194951): Check how the number of registers used depends on the
  // data type. `registers_per_block_limit()` returns the number of 32-bit
  // registers. Check if 64-bit types need twice as many registers. Check if
  // smaller types can fit into one register.
  return tile_size <= kFractionOfRegistersAvailableToStoreTile *
                          device_info.registers_per_block_limit();
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
  if (largest_live_tile_size <= 256) return 1;
  if (largest_live_tile_size <= 1024) return 2;
  if (largest_live_tile_size <= 4096) return 4;
  return 8;
}

}  // namespace

int64_t GpuPerformanceModelWithIndexingAnalysis::FlopsPerElement(
    const HloInstruction* instr) {
  // Instruction that are only used for indexing are not counted for FLOPs.
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConstant:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return 0;
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
    auto output_shape = instr->shape().IsArray()
                            ? instr->shape()
                            : instr->shape().tuple_shapes(0);

    // Size of reduction dimensions.
    int64_t reduction_factor = ShapeUtil::ElementsIn(operand_shape) /
                               ShapeUtil::ElementsIn(output_shape);

    // The Cost Model assumes that the reduction computation is applied N-1
    // times to reduce N elements. This is not true, because emitters will
    // generate a loop with N iterations. We don't fix it here to keep this
    // estimate consistent with GpuHloCostAnalysis. This is like doesn't matter
    // much for the application of the Cost Model.
    return (reduction_factor - 1) * flops_per_reduce_computation;
  }

  // Encountered unexpected instruction, call to GpuHloCostAnalysis.
  TF_CHECK_OK(
      cost_analysis_.RevisitInstruction(const_cast<HloInstruction*>(instr)));

  return cost_analysis_.flop_count(*instr) /
         ShapeUtil::ElementsInRecursive(instr->shape());
}

int64_t GpuPerformanceModelWithIndexingAnalysis::GetShapeSizeRecursive(
    const Shape& shape) const {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return shape_size_(shape);
  }

  int64_t total_size = 0;
  for (const auto& element_shape : shape.tuple_shapes()) {
    total_size += GetShapeSizeRecursive(element_shape);
  }
  return total_size;
}

int64_t GetIterationSpaceSize(const IndexingMap& indexing_map,
                              const HloInstruction* instr) {
  if (indexing_map.IsUndefined()) {
    return ShapeUtil::ElementsInRecursive(instr->shape());
  }

  if (indexing_map.IsKnownEmpty()) {
    return 0;
  }

  auto get_ranges_iteration_space_size =
      [](const std::vector<Interval>& ranges) {
        int64_t num_iters = 1;
        for (const Interval& range : ranges) {
          num_iters *= range.upper - range.lower + 1;
        }
        return num_iters;
      };

  return get_ranges_iteration_space_size(indexing_map.GetSymbolBounds()) *
         get_ranges_iteration_space_size(indexing_map.GetDimensionBounds());
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForFusion(
    const HloFusionAnalysis& fusion_analysis, bool is_coalesced) {
  auto& fusion_adaptor = fusion_analysis.fusion();
  VLOG(5) << "EstimateRunTimeForFusion: " << fusion_adaptor.ToString();

  auto roots = fusion_adaptor.GetRoots();
  CHECK_EQ(roots.size(), 1)
      << "Indexing cost model doesn't support multi-output fusions.";
  auto root_shape = roots.front().shape();

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);

  int64_t num_blocks = launch_dimensions.num_blocks();

  // Compute indexing from root to each instruction in the fusion and fusion
  // operands. For each instruction, tells which elements of the instructions
  // result will be used to compute one result element of the fusion.
  auto grouped_fusion_indexing = ComputeGroupedOutputToInputIndexing(
      fusion_adaptor, roots[0], mlir_context_);

  int64_t flops = 0;
  int64_t bytes_read = 0;
  absl::Duration read_time = absl::ZeroDuration();

  for (const auto& [instr, indexing_maps] : grouped_fusion_indexing) {
    VLOG(10) << "instr: " << instr->name();

    // Instructions inside the fusion are computation and account for FLOPs
    // count. Instructions outside the fusion are operands of the fusion and
    // account for memory read time.
    bool is_operand = !fusion_adaptor.ContainsInstruction(instr);

    auto element_type = instr->shape().element_type();
    int64_t n_bytes_total = 0;
    for (const auto& indexing_map : indexing_maps) {
      VLOG(10) << indexing_map;

      int64_t num_iters = GetIterationSpaceSize(indexing_map, instr);

      if (is_operand) {
        int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
        n_bytes_total += type_size * num_iters;
      } else {
        int64_t flops_per_element = FlopsPerElement(instr);
        flops += flops_per_element * num_iters;
      }
    }

    if (is_operand) {
      int64_t operand_size = shape_size_(instr->shape());
      int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
      bytes_read += n_bytes_total;

      VLogOperandRead(instr, n_bytes_total, n_bytes_net, is_coalesced);

      read_time += ReadTimeWithDRAMHeuristic(
          *device_info_, num_blocks, n_bytes_net, n_bytes_total, element_type,
          GetCoalescingUtilizationRate(element_type, *device_info_,
                                       is_coalesced));
    }
  }

  int64_t bytes_written = GetShapeSizeRecursive(root_shape);

  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, num_blocks,
                  launch_dimensions.num_threads_per_block());
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time, GpuPerformanceModelOptions::Default());

  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO fusion: " << fusion_adaptor.ToString()
          << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();

  return runtime_data;
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForInstruction(
    const HloInstruction* producer) {
  // Stand-alone bitcast is always no-op during runtime.
  if (producer->opcode() == HloOpcode::kBitcast) {
    return EstimateRunTimeData::Zero();
  }

  auto fusion_analysis = HloFusionAnalysis::Create(*producer, *device_info_);

  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), *device_info_, producer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer) {
  auto fusion_analysis =
      HloFusionAnalysis::Create(*producer, *consumer, *device_info_);

  bool is_coalesced =
      IsReadCoalescedHeuristic(fusion_analysis.GetEmitterFusionKind(),
                               *device_info_, producer, consumer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

/*static*/
GpuPerformanceModelWithIndexingAnalysis::RunTimes
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto producer_runtime = EstimateRunTimeForInstruction(producer);

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (const auto& consumer : fused_consumers) {
    time_unfused += EstimateRunTimeForInstruction(consumer).exec_time;
    time_fused +=
        EstimateRunTimeForProducerConsumer(producer, consumer).exec_time;
  }

  return {time_unfused, time_fused};
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledHloComputation(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const LaunchDimensions& launch_dimensions) {
  absl::flat_hash_map<const HloInstruction*, OperandReadInfo> n_bytes_total_map;

  int64_t flops = 0;
  int64_t bytes_read = 0;
  int64_t num_blocks = launch_dimensions.num_blocks();

  for (const auto& tiled_hlo : tiled_hlo_computation.instructions()) {
    // Number of elements in the tile after padding.
    int64_t padded_tile_size = GetPaddedTileSize(tiled_hlo->tile_sizes());

    // Check if the tile is too large to fit in registers and would result in
    // spilling.
    if (!DoesTileFitsInRegisters(padded_tile_size, *device_info_)) {
      // TODO(b/363194951): Estimate performance regression due to spilling in
      // terms of memory bandwidth instead of returning infinite run time.
      return EstimateRunTimeData::Infinite();
    }

    const HloInstruction* hlo = tiled_hlo->hlo();

    if (fusion_adaptor.ContainsInstruction(hlo)) {
      if (hlo->opcode() == HloOpcode::kConcatenate) {
        // TODO(b/351342921): Add propagation of the number of blocks that read
        // or compute a tile. Concatenate is the only operation that may change
        // that.
        return absl::FailedPreconditionError(
            "Concatenate is not supported by the indexing cost model.");
      }

      // Total number of elements computed for this tile across all blocks.
      //
      // Even if real `tile_size` is smaller than `padded_tile_size`, SM will
      // still perform calculations on masked values, so they should count
      // towards FLOPs.
      int64_t num_elements = num_blocks * padded_tile_size;

      // Tiles inside the computation contribute to the total FLOPs count.
      flops += FlopsPerElement(hlo) * num_elements;
    } else {
      // Number of elements in the tile.
      int64_t tile_size = Product(tiled_hlo->tile_sizes());

      // Total number of elements that are read from memory across all blocks.
      //
      // Triton requires that all tiles have dimensions that are padded to the
      // next power of 2. However, the load masks the padded elements, so they
      // are not read from memory, but set directly in registers. As a result,
      // the number of elements read from memory is equal to the size of the
      // original tile.
      int64_t num_elements = num_blocks * tile_size;

      // Tiles of the operands of the fusion contribute to the total memory
      // read time.
      int64_t element_type_size =
          ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());
      int64_t tile_bytes_read = element_type_size * num_elements;

      bytes_read += tile_bytes_read;

      double effective_bandwidth_utilization_rate =
          BandwidthUtilizationRateHeuristicForTiledMemoryAccess(*tiled_hlo,
                                                                *device_info_);

      OperandReadInfo& operand_read_info = n_bytes_total_map[hlo];
      operand_read_info.total_bytes_read += tile_bytes_read;
      // TODO(b/332714755): using std::min is more pessimistic than it needs to
      // be since it'll end up assuming that if one read is done with lower
      // bandwidth, all other reads of the same operand will also be done with
      // lower bandwidth. But it's a start. We should refactor this function to
      // properly track each read independently later.
      operand_read_info.read_bandwidth_utilization_rate =
          std::min(operand_read_info.read_bandwidth_utilization_rate,
                   effective_bandwidth_utilization_rate);
    }
  }

  absl::Duration read_time = absl::ZeroDuration();
  for (const auto& [hlo, operand_read_info] : n_bytes_total_map) {
    int64_t operand_size = shape_size_(hlo->shape());
    int64_t n_bytes_net =
        std::min(operand_size, operand_read_info.total_bytes_read);

    // TODO(b/332714755): use
    // `BandwidthUtilizationRateHeuristicForTiledMemoryAccess` to compute read
    // time as well.
    read_time += ReadTimeWithDRAMHeuristic(
        *device_info_, num_blocks, n_bytes_net,
        operand_read_info.total_bytes_read,
        /*element_type=*/hlo->shape().element_type(),
        /*hbm_bandwidth_utilization_rate=*/
        operand_read_info.read_bandwidth_utilization_rate);
  }

  auto roots = tiled_hlo_computation.GetRoots();
  int64_t bytes_written = 0;
  absl::Duration write_time;
  for (auto* root : roots) {
    int64_t effective_bandwidth =
        BandwidthUtilizationRateHeuristicForTiledMemoryAccess(*root,
                                                              *device_info_) *
        device_info_->memory_bandwidth();
    int64_t bytes_written_for_root =
        GetShapeSizeRecursive(root->hlo()->shape());
    write_time +=
        absl::Seconds(1.0 * bytes_written_for_root / effective_bandwidth);
    bytes_written += bytes_written_for_root;
  }

  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, num_blocks,
                  launch_dimensions.num_threads_per_block());

  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time, GpuPerformanceModelOptions::Default());

  return EstimateRunTimeData{/*flops=*/flops,
                             /*bytes_read=*/bytes_read,
                             /*bytes_written=*/bytes_written,
                             /*read_time=*/read_time,
                             /*write_time=*/write_time,
                             /*compute_time=*/compute_time,
                             /*exec_time=*/exec_time};
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledFusion(
    const HloFusionAdaptor& fusion_adaptor,
    const LaunchDimensions& launch_dimensions,
    const std::vector<std::vector<int64_t>>& tile_sizes) {
  // TODO(b/332714755): Add caching for SymbolicTileAnalysis.
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(
          fusion_adaptor, mlir_context_,
          /*emitter_specific_constraints_builder=*/nullptr);
  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SymbolicTileAnalysis failed. ", fusion_decision->Explain()));
  }
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      analysis.ComputeTiledHloInstructions(
                          tile_sizes[analysis.real_root_index()]));

  return EstimateRunTimeForTiledHloComputation(
      fusion_adaptor, tiled_hlo_computation, launch_dimensions);
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTriton(
    const HloInstruction* producer, const HloInstruction* consumer) {
  const auto& fusion_analysis =
      (consumer == nullptr) ? fusion_analysis_cache_->Get(*producer)
                            : fusion_analysis_cache_->Get(*producer, *consumer);
  auto launch_config = TritonFusion(fusion_analysis).launch_config();

  if (!launch_config.has_value()) {
    return absl::InvalidArgumentError(
        "Could not get launch config for Triton fusion.");
  }

  return EstimateRunTimeForTiledFusion(
      fusion_analysis.fusion(), launch_config->launch_dimensions,
      launch_config->block_level_parameters.output_tile_sizes);
}

/*static*/
LaunchDimensions
GpuPerformanceModelWithIndexingAnalysis::GetLaunchDimensionsForTiledFusion(
    const TiledHloComputation& tiled_hlo_computation,
    const se::DeviceDescription& device_info) {
  int64_t num_blocks = tiled_hlo_computation.num_output_tiles();

  // Decide on the number of warps to use based on the largest live tile size
  // at any given point within the computation.
  int64_t largest_live_tile_size = 1;
  for (const auto& tiled_hlo : tiled_hlo_computation.instructions()) {
    largest_live_tile_size = std::max(
        largest_live_tile_size, GetPaddedTileSize(tiled_hlo->tile_sizes()));
  }
  int64_t num_warps = GetNumWarps(largest_live_tile_size);

  return {static_cast<uint64_t>(num_blocks),
          static_cast<uint64_t>(num_warps * WarpSize(device_info))};
}

absl::StatusOr<TiledRunTimeDataOrError>
GpuPerformanceModelWithIndexingAnalysis::TryFindBestTilingForFusion(
    const HloFusionAdaptor& fusion_adaptor) {
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(
          fusion_adaptor, mlir_context_,
          TritonEmitterConstraints::GetBuilder(*device_info_));

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return *fusion_decision;
  }

  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  TF_ASSIGN_OR_RETURN(auto tilings, analysis.GetGoodTilings());

  std::optional<TiledRunTimeData> best_tiled_run_time_data;

  for (const auto& tiling : tilings) {
    // TODO(b/372454662): This needs to be adjusted if we want to support more
    // than one "real root" (i.e. a root without users).
    // Currently ComputeTiledHloInstructions() may fail and return an
    // Unimplemented error for cases of multi-output fusion that we do not
    // support yet.
    auto maybe_tiled_hlo_computation =
        analysis.ComputeTiledHloInstructions(tiling);
    if (!maybe_tiled_hlo_computation.ok()) {
      if (maybe_tiled_hlo_computation.status().code() ==
              absl::StatusCode::kUnimplemented &&
          absl::StrContains(maybe_tiled_hlo_computation.status().message(),
                            "multi-output fusion")) {
        continue;
      }
      return maybe_tiled_hlo_computation.status();
    }

    auto tiled_hlo_computation = std::move(maybe_tiled_hlo_computation.value());
    LaunchDimensions launch_dimensions =
        GetLaunchDimensionsForTiledFusion(tiled_hlo_computation, *device_info_);

    TF_ASSIGN_OR_RETURN(
        EstimateRunTimeData estimate_run_time_data,
        EstimateRunTimeForTiledHloComputation(
            fusion_adaptor, tiled_hlo_computation, launch_dimensions));

    if (!best_tiled_run_time_data.has_value() ||
        estimate_run_time_data.exec_time <
            best_tiled_run_time_data->runtime_data.exec_time) {
      BlockLevelParameters block_level_parameters;
      auto tiled_roots = tiled_hlo_computation.GetRoots();
      block_level_parameters.output_tile_sizes.reserve(tiled_roots.size());
      for (auto tiled_root : tiled_roots) {
        block_level_parameters.output_tile_sizes.emplace_back(
            tiled_root->tile_sizes().begin(), tiled_root->tile_sizes().end());
      }
      block_level_parameters.num_warps =
          launch_dimensions.num_threads_per_block() / WarpSize(*device_info_);

      best_tiled_run_time_data =
          TiledRunTimeData{estimate_run_time_data, block_level_parameters};
    }
  }

  if (!best_tiled_run_time_data.has_value()) {
    return FusionDecision::Forbid("No valid tilings found.");
  }
  return *best_tiled_run_time_data;
}

}  // namespace gpu
}  // namespace xla
