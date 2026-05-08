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

#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::gpu_dot_fusion_cost_model {

namespace detail {

namespace {
using ::xla::primitive_util::BitWidth;

int64_t CalculateNumThreadblocks(const DotProblemInfo& dot,
                                 const DotTileSize& dot_tile) {
  // TODO(maniananth): Add special handling for grouped matmuls here.
  int64_t num_tiles_along_b_dimension = CeilOfRatio<int64_t>(dot.b, dot_tile.b);
  int64_t num_tiles_along_m_dimension = CeilOfRatio<int64_t>(dot.m, dot_tile.m);
  int64_t num_tiles_along_n_dimension = CeilOfRatio<int64_t>(dot.n, dot_tile.n);
  int64_t num_threadblocks = num_tiles_along_b_dimension *
                             num_tiles_along_m_dimension *
                             num_tiles_along_n_dimension;

  return num_threadblocks;
}

int64_t CalculateNumWaves(int64_t threadblock_count,
                          const se::DeviceDescription& device_info) {
  int64_t core_count = device_info.core_count();
  return CeilOfRatio<int64_t>(threadblock_count, core_count);
}

int64_t CalculateTileFlops(const DotTileSize& dot_tile, int64_t problem_k) {
  return /*2 FLOPs per MAC*/ 2 * dot_tile.b * dot_tile.m * dot_tile.n *
         problem_k;
}

// Calculates the effective flops for a GPU DOT operation as a function of the
// tile size (excludes clock throttling). Not all tile sizes are equally able to
// extract utilization on the same generation GPUs even if the workload is
// compute bound. GEMM performance is sensitive to the tensor core
// instruction throughputs that the programming model exposes.
double GetEffectiveFlopsPerNsForTileSize(
    const int64_t tile_m, const se::DeviceDescription& device_info,
    xla::PrimitiveType element_type) {
  se::CudaComputeCapability cuda_compute_capability =
      device_info.cuda_compute_capability();

  // Peak flops per ns for device.
  int64_t peak_flops_per_ns =
      GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(device_info,
                                                           element_type);

  // Final flops derate factor.
  double flops_derate = 1.0;

  if (cuda_compute_capability.IsBlackwell()) {
    if (tile_m < 128) {
      // TODO(maniananth): Update this derate once we have more data from
      // actual measurements on Blackwell. For now, we are applying a 50%
      // derate to account for smaller M shapes.
      flops_derate = 0.5;
    }
  } else if (cuda_compute_capability.IsHopper()) {
    if (tile_m < 64) {
      // Having a tile size M < 64 will lead to not being able to use the H100
      // tensor core instructions (wgmma). Defaulting to wmma instructions from
      // A100 can result in a 63% derate in flops as benchmarked by HazyResearch
      // as part of ThunderKittens work.
      // (https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
      flops_derate = 0.63;
    }
  } else if (cuda_compute_capability.IsAmpere()) {
    if (tile_m < 16) {
      // A100 tensor core instructions are effective at tile_m >= 16. We're
      // applying a 50% derate to account for this.
      flops_derate = 0.5;
    }
  }
  return peak_flops_per_ns * flops_derate;
}

int64_t CalculateL2Bytes(const DotProblemInfo& dot, const DotTileSize& out_tile,
                         int64_t threadblock_count) {
  // When tiling the GEMM problem on the outputs and mapping one tile per SM,
  // the problem of data replication (or extra loads of the same data) between
  // multiple SMs occurs. This leads to more data loads than what’s expected
  // algorithmically, and increases bandwidth needs on the L2 → SM paths.

  // Input data loaded by each tile is equal to (Tile_M + Tile_N) * problem_k
  // bytes (The threadblock iterates over the entire problem_k dimension).
  int64_t lhs_bytes = CeilOfRatio<int64_t>(
      out_tile.b * out_tile.m * dot.k * BitWidth(dot.lhs_element_type), 8);
  int64_t rhs_bytes = CeilOfRatio<int64_t>(
      out_tile.b * out_tile.n * dot.k * BitWidth(dot.rhs_element_type), 8);
  int64_t l2_data_per_tile = lhs_bytes + rhs_bytes;

  // Across all the tiles, data loads will be equal to: (l2_data_per_tile *
  // threadblock_count).

  // TODO(maniananth): Since H100, threadblocks within the same cluster will
  // avoid redundant loads by reading from L2 cache once and multicasting the
  // data to all threadblocks within the cluster. This is controlled
  // programmatically and most performant GEMM implementations will use this
  // feature. To model this, we scale the total data loads by the total number
  // of threadblocks in a cluster.

  // On A100 and older GPUs, we will not see this behavior and the total data
  // loads will be equal to (l2_data_per_tile * threadblock_count). Hence the
  // cluster shape can be set to (1x1).
  // TODO(maniananth): Account for Threadblock clusters here.
  int64_t total_l2_data = l2_data_per_tile * threadblock_count;
  return total_l2_data;
}

}  // namespace

DotProblemInfo::DotProblemInfo(const HloDotInstruction& dot) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  // We support 4D and higher rank GEMMs to handle multi-dimensional batching
  // (such as having independent head and batch dimensions in multi-head
  // attention workloads) without requiring explicit reshape or flattening ops.
  b = 1;
  for (int64_t batch_dim_idx : dim_numbers.lhs_batch_dimensions()) {
    b *= lhs_shape.dimensions(batch_dim_idx);
  }
  m = lhs_shape.dimensions(lhs_non_contracting_dims[0]);
  n = rhs_shape.dimensions(rhs_non_contracting_dims[0]);
  k = lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions()[0]);

  lhs_element_type = dot.operand(0)->shape().element_type();
  rhs_element_type = dot.operand(1)->shape().element_type();
  output_element_type = dot.shape().element_type();
}

absl::StatusOr<ComputeAndFlops> CalculateComputeTimeWithTileAndWaveQuantization(
    const DotProblemInfo& dot, const DotTileSize& dot_tile,
    const se::DeviceDescription& device_info) {
  int64_t threadblock_count = CalculateNumThreadblocks(dot, dot_tile);
  int64_t wave_count = CalculateNumWaves(threadblock_count, device_info);
  int64_t flops_per_tile = CalculateTileFlops(dot_tile, dot.k);
  // The following is not the actual number of threadblocks launched, but due to
  // how wave quantization works, we get the effect of running extra
  // threadblocks when adding to roofline projections.
  int64_t cta_count_with_wave_quant = wave_count * device_info.core_count();
  int64_t total_flops_with_wave_quant =
      flops_per_tile * cta_count_with_wave_quant;
  double effective_flops_rate = GetEffectiveFlopsPerNsForTileSize(
      dot_tile.m, device_info, dot.lhs_element_type);

  ComputeAndFlops result;
  result.flops_with_wave_quant = total_flops_with_wave_quant;
  // TODO(maniananth): Add a cap for power throttling here.
  result.compute_time = absl::Nanoseconds(1.0f * total_flops_with_wave_quant /
                                          effective_flops_rate);

  return result;
}

absl::StatusOr<absl::Duration> CalculateL2Time(
    const DotProblemInfo& dot, const DotTileSize& dot_tile,
    const se::DeviceDescription& device_info, bool is_tma_allowed) {
  // TODO(maniananth): L2 bandwidth has been hardcoded for H100 based on
  // microbenchmarking L2 bandwidth within a partition, but we should add this
  // to the device info and extend for more GPUs.

  int64_t threadblock_count = CalculateNumThreadblocks(dot, dot_tile);
  double device_l2_bandwidth = 6.65 * 1e12;  // Measured H100 L2 bandwidth.
  int64_t num_k_iters = CeilOfRatio<int64_t>(dot.k, dot_tile.k);

  // Empirical overheads per K-dimension iteration.
  // The overhead is dictated by the memory instruction pathway rather than
  // strictly the hardware generation.
  // Tuned via grid search to minimize MAPE.
  constexpr double kTmaLoopOverheadSeconds =
      150 * 1e-9;  // Fast path (cp.async.bulk)
  constexpr double kLegacyLoopOverheadSeconds =
      400 * 1e-9;  // Slow path (cp.async)
  double k_loop_overhead =
      is_tma_allowed ? kTmaLoopOverheadSeconds : kLegacyLoopOverheadSeconds;

  double base_time_seconds =
      1.0f * CalculateL2Bytes(dot, dot_tile, threadblock_count) /
      device_l2_bandwidth;
  return absl::Seconds(base_time_seconds + num_k_iters * k_loop_overhead);
}

// Returns the effective HBM bandwidth in bytes per second for a given dma_size.
// dma_size is the total amount of data transferred to/from HBM in bytes.
float GetEffectiveHbmBandwidth(const int64_t dma_size,
                               const se::DeviceDescription& device_info) {
  using HbmBandwidthLookupEntry =
      std::pair</*dma_size*/ int64_t, /*measured bandwidth*/ float>;
  std::array<HbmBandwidthLookupEntry, 18> hbm_bandwidth_GBps_lookup_h100 = {
      {{8192, 1.42f},
       {16384, 3.03f},
       {32768, 6.02f},
       {65536, 11.77f},
       {131072, 23.68f},
       {262144, 47.35f},
       {524288, 92.56f},
       {1048576, 179.06f},
       {2097152, 346.75f},
       {4194304, 639.38f},
       {8388608, 1069.98f},
       {16777216, 1583.95f},
       {33554432, 1974.72f},
       {67108864, 2343.19f},
       {134217728, 2632.96f},
       {268435456, 2766.69f},
       {536870912, 2968.89f},
       {1073741824, 3126.0f}}};

  if (dma_size <= hbm_bandwidth_GBps_lookup_h100.front().first) {
    return hbm_bandwidth_GBps_lookup_h100.front().second * (1 << 30);
  }
  if (dma_size >= hbm_bandwidth_GBps_lookup_h100.back().first) {
    return hbm_bandwidth_GBps_lookup_h100.back().second * (1 << 30);
  }

  auto it2 = std::lower_bound(hbm_bandwidth_GBps_lookup_h100.begin(),
                              hbm_bandwidth_GBps_lookup_h100.end(), dma_size,
                              [](const std::pair<int64_t, float>& a,
                                 const int64_t b) { return a.first < b; });
  auto it1 = it2 - 1;

  // Linear interpolation between the two entries in the lookup table. std::lerp
  // is not used as it is only available since C++20.
  auto a = it1->second;
  auto b = it2->second;
  auto t =
      (dma_size - it1->first) / static_cast<float>(it2->first - it1->first);
  return (a + t * (b - a)) * (1 << 30);
}

HbmEstimates CalculateHbmTime(const DotProblemInfo& dot,
                              const se::DeviceDescription& device_info) {
  // Calculate the number of bytes for input reads and output writes to HBM.
  int64_t lhs_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.m * dot.k * BitWidth(dot.lhs_element_type), 8);
  int64_t rhs_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.k * dot.n * BitWidth(dot.rhs_element_type), 8);
  int64_t output_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.m * dot.n * BitWidth(dot.output_element_type), 8);

  // Main loop loads the input matrices from HBM using SW pipelining and updates
  // accumulators stored in register files (within the SM/compute unit). The
  // epilogue loop writes the output matrices from register files to HBM. Main
  // loop and epilogue loop are executed sequentially.
  int64_t main_loop_bytes = lhs_tile_bytes + rhs_tile_bytes;
  int64_t epilogue_bytes = output_tile_bytes;

  HbmEstimates result;
  result.bytes_read = main_loop_bytes;
  result.bytes_written = epilogue_bytes;

  // Calculate the effective HBM bandwidth for the input and output bytes using
  // the derate lookup table.
  float dram_bandwidth =
      GetEffectiveHbmBandwidth(main_loop_bytes + epilogue_bytes, device_info);

  // Calculate the HBM time using the effective bandwidth for each transfer
  // size. In the current implementation, we are assuming that the main loop and
  // epilogue loop have the same effective DRAM bandwidth. This could change in
  // the future, if we choose to model it based on their respective transfer
  // sizes.
  result.read_time = absl::Seconds(1.0f * (main_loop_bytes) / dram_bandwidth);
  result.write_time = absl::Seconds(1.0f * (epilogue_bytes) / dram_bandwidth);

  return result;
}

}  // namespace detail

absl::Status IsSupported(const HloDotInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  if (lhs_non_contracting_dims.size() > 1 ||
      rhs_non_contracting_dims.size() > 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Multiple non-contracting dimensions are not supported, got LHS: [",
        absl::StrJoin(lhs_non_contracting_dims, ","), "], RHS: [",
        absl::StrJoin(rhs_non_contracting_dims, ","), "]"));
  }
  if (dim_numbers.lhs_contracting_dimensions_size() != 1 ||
      dim_numbers.rhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Exactly one contracting dimension is supported, got LHS: [",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ","),
        "], RHS: [",
        absl::StrJoin(dim_numbers.rhs_contracting_dimensions(), ","), "]"));
  }

  // TODO: b/501002656 - Support downstream transposes by fixing dimension
  // mapping.
  std::vector<const HloInstruction*> stack;
  absl::flat_hash_set<const HloInstruction*> visited;
  stack.push_back(dot);
  visited.insert(dot);
  while (!stack.empty()) {
    const HloInstruction* current = stack.back();
    stack.pop_back();
    if (current != dot && current->opcode() == HloOpcode::kTranspose) {
      return absl::UnimplementedError(
          "Dot with a downstream transpose is not supported.");
    }
    for (const HloInstruction* user : current->users()) {
      if (visited.insert(user).second) {
        stack.push_back(user);
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<int64_t> ExtractBlockK(const HloDotInstruction* dot) {
  if (!dot->has_backend_config()) {
    return absl::FailedPreconditionError(
        "Dot instruction must have a backend config with tiling sizes.");
  }
  TF_ASSIGN_OR_RETURN(auto tile_config, dot->backend_config<xla::gpu::Tile>());
  TF_RET_CHECK(tile_config.sizes_size() > 0)
      << "Tile backend config must have sizes.";
  return tile_config.sizes(0);
}

absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForDotOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info, std::optional<int64_t> block_k) {
  TF_RETURN_IF_ERROR(IsSupported(dot));
  if (block_params.output_tile_sizes.size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only single tile size is supported, got ",
                     block_params.output_tile_sizes.size()));
  }

  int64_t block_k_val;
  if (block_k.has_value()) {
    block_k_val = *block_k;
  } else {
    TF_ASSIGN_OR_RETURN(block_k_val, ExtractBlockK(dot));
  }

  detail::DotProblemInfo dot_info(*dot);

  const std::vector<int64_t>& tile_shape = block_params.output_tile_sizes[0];
  if (tile_shape.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tile shape must be of size at least 2, got ", tile_shape.size()));
  }
  int64_t tile_b = 1;
  for (size_t i = 0; i < tile_shape.size() - 2; ++i) {
    tile_b *= tile_shape[i];
  }
  int64_t tile_m = tile_shape[tile_shape.size() - 2];
  int64_t tile_n = tile_shape[tile_shape.size() - 1];
  detail::DotTileSize dot_tile{/*m=*/tile_m,
                               /*n=*/tile_n,
                               /*k=*/block_k_val,
                               /*b=*/tile_b};

  EstimateRunTimeData estimates;

  // Calculate compute roofline with tile and wave quantization.
  TF_ASSIGN_OR_RETURN(detail::ComputeAndFlops compute_and_flops,
                      detail::CalculateComputeTimeWithTileAndWaveQuantization(
                          dot_info, dot_tile, device_info));
  estimates.compute_time = compute_and_flops.compute_time;
  estimates.flops = compute_and_flops.flops_with_wave_quant;

  // Calculate HBM roofline.
  detail::HbmEstimates hbm_timing =
      detail::CalculateHbmTime(dot_info, device_info);

  estimates.read_time = hbm_timing.read_time;
  estimates.write_time = hbm_timing.write_time;
  estimates.bytes_read = hbm_timing.bytes_read;
  estimates.bytes_written = hbm_timing.bytes_written;

  // Calculate L2 time.
  TF_ASSIGN_OR_RETURN(absl::Duration l2_time,
                      detail::CalculateL2Time(dot_info, dot_tile, device_info,
                                              block_params.is_tma_allowed));

  // Assuming perfect overlap between compute and memory.
  estimates.exec_time = std::max(
      {compute_and_flops.compute_time, hbm_timing.total_time(), l2_time});

  return estimates;
}

}  // namespace xla::gpu::gpu_dot_fusion_cost_model
