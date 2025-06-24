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
#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using primitive_util::BitWidth;

namespace {

bool TileFitsInRegisters(int64_t block_m, int64_t block_n,
                         const PrimitiveType& element_type,
                         const se::DeviceDescription& device_info) {
  int bits_per_output_elem = BitWidth(element_type);
  int registers_per_block = device_info.registers_per_block_limit();
  int64_t block_size = block_m * block_n;
  int64_t bytes_per_block =
      CeilOfRatio<int64_t>(block_size * bits_per_output_elem, 8);
  constexpr double kFractionOfRegistersAvailableForAccumulators = 0.8;
  return bytes_per_block <=
         (registers_per_block * kFractionOfRegistersAvailableForAccumulators);
}

absl::StatusOr<absl::InlinedVector<BlockLevelParameters, 4>>
GetDotAlgorithmValidConfigs(const HloDotInstruction* dot,
                            const se::DeviceDescription& device_info) {
  absl::InlinedVector<BlockLevelParameters, 4> valid_configs;

  for (int64_t block_m = detail::kMinBlockDim; block_m <= detail::kMaxBlockDim;
       block_m *= 2) {
    for (int64_t block_n = detail::kMinBlockDim;
         block_n <= detail::kMaxBlockDim; block_n *= 2) {
      if (!TileFitsInRegisters(block_m, block_n, dot->shape().element_type(),
                               device_info)) {
        continue;
      }

      // TODO(maniananth): Add the logic to find valid kBlock stages.
      BlockLevelParameters block_level_parameters;
      block_level_parameters.output_tile_sizes.push_back(
          std::vector<int64_t>{block_m, block_n});
      // TODO(maniananth): Add the logic to sweep num warps per block.
      block_level_parameters.num_warps = detail::kNumWarpsPerBlock;
      valid_configs.push_back(block_level_parameters);
    }
  }

  return valid_configs;
}

int64_t CalculateNumThreadblocks(const HloDotInstruction* dot, int64_t tile_m,
                                 int64_t tile_n) {
  GpuDotFusionCostModel::DotProblemDimensions dims(*dot);
  int64_t tile_k = dims.k;
  // TODO(maniananth): Add special handling for grouped matmuls here.
  int64_t num_tiles_along_m_dimension = CeilOfRatio<int64_t>(dims.m, tile_m);
  int64_t num_tiles_along_n_dimension = CeilOfRatio<int64_t>(dims.n, tile_n);
  int64_t num_tiles_along_k_dimension = CeilOfRatio<int64_t>(dims.k, tile_k);
  int64_t num_threadblocks = dims.b * num_tiles_along_m_dimension *
                             num_tiles_along_n_dimension *
                             num_tiles_along_k_dimension;

  return num_threadblocks;
}

int64_t CalculateNumWaves(int64_t threadblock_count,
                          const se::DeviceDescription& device_info) {
  int64_t core_count = device_info.core_count();
  return CeilOfRatio<int64_t>(threadblock_count, core_count);
}

int64_t CalculateTileFlops(int64_t tile_m, int64_t tile_n, int64_t problem_k) {
  return /*flops per MAC*/ 2 * tile_m * tile_n * problem_k;
}

// Calculates the effective flops for a GPU DOT operation as a function of the
// tile size (excludes clock throttling). Not all tile sizes are equally able to
// extract utilization on the same generation GPUs even if the workload is
// compute bound. GEMM performance is sensitive to the tensor core
// instruction throughputs that the programming model exposes.
double GetEffectiveFlopsPerNsForTileSize(
    const int64_t tile_m, const se::DeviceDescription& device_info) {
  se::CudaComputeCapability cuda_compute_capability =
      device_info.cuda_compute_capability();

  // Peak flops per ns for device.
  int64_t peak_flops_per_ns =
      GpuPerformanceModelBase::CalculateEffectiveFlopsPerNs(
          device_info, device_info.fpus_per_core(), device_info.core_count());

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

int64_t CalculateL2Bytes(absl::Span<const int64_t> tile_shape,
                         int64_t problem_k, int64_t threadblock_count) {
  // When tiling the GEMM problem on the outputs and mapping one tile per SM,
  // the problem of data replication (or extra loads of the same data) between
  // multiple SMs occurs. This leads to more data loads than what’s expected
  // algorithmically, and increases bandwidth needs on the L2 → SM paths.

  // Input data loaded by each tile is equal to (Tile_M + Tile_N) * Tile_K
  // bytes.
  int64_t l2_data_per_tile = (tile_shape[0] + tile_shape[1]) * problem_k;

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

namespace detail {

absl::StatusOr<absl::Duration> CalculateComputeTimeWithTileAndWaveQuantization(
    const HloDotInstruction* dot, absl::Span<const int64_t> tile_shape,
    const se::DeviceDescription& device_info) {
  if (tile_shape.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tile shape must be of size 2, got ", tile_shape.size()));
  }

  GpuDotFusionCostModel::DotProblemDimensions dims(*dot);
  int64_t tile_m = tile_shape[0], tile_n = tile_shape[1];
  int64_t threadblock_count = CalculateNumThreadblocks(dot, tile_m, tile_n);
  int64_t wave_count = CalculateNumWaves(threadblock_count, device_info);
  int64_t flops_per_tile = CalculateTileFlops(tile_m, tile_n, dims.k);
  // The following is not the actual number of threadblocks launched, but due to
  // how wave quantization works, we get the effect of running extra
  // threadblocks when adding to roofline projections.
  int64_t cta_count_with_wave_quant = wave_count * device_info.core_count();
  int64_t total_flops_with_wave_quant =
      flops_per_tile * cta_count_with_wave_quant;
  double effective_flops =
      GetEffectiveFlopsPerNsForTileSize(tile_m, device_info);
  // TODO(maniananth): Add a cap for power throttling here.
  return absl::Nanoseconds(1.0f * total_flops_with_wave_quant /
                           effective_flops);
}

absl::StatusOr<absl::Duration> CalculateL2Time(
    const HloDotInstruction* dot, absl::Span<const int64_t> tile_shape,
    const se::DeviceDescription& device_info) {
  if (tile_shape.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tile shape must be of size 2, got ", tile_shape.size()));
  }
  // TODO(maniananth): L2 bandwidth has been hardcoded for H100 based on
  // microbenchmarking L2 bandwidth within a partition, but we should add this
  // to the device info and extend for more GPUs.
  // TODO(maniananth): Enforcing this check will cause unit tests written for
  // RTX A6000 device descriptions to fail. We should enable this check once we
  // have the L2 bandwidth for RTX A6000 or move unit tests to use H100
  // device description.
  // if (device_info.cuda_compute_capability() !=
  //     se::CudaComputeCapability(9, 0)) {
  //   return absl::InvalidArgumentError(
  //       "L2 time calculation is only supported for H100 GPUs.");
  // }

  GpuDotFusionCostModel::DotProblemDimensions dims(*dot);
  int64_t tile_m = tile_shape[0], tile_n = tile_shape[1];
  int64_t threadblock_count = CalculateNumThreadblocks(dot, tile_m, tile_n);
  double device_l2_bandwidth = 6.65 * 1e12;  // Measured H100 L2 bandwidth.

  return absl::Seconds(1.0f *
                       CalculateL2Bytes(tile_shape, dims.k, threadblock_count) /
                       device_l2_bandwidth);
}

absl::Duration CalculateHbmTime(const HloDotInstruction* dot,
                                const se::DeviceDescription& device_info) {
  // TODO(maniananth): Implement HBM derate lookup using profiled tables.
  float hbm_bandwidth_utilization_rate = 0.8;
  float dram_bandwidth =
      device_info.memory_bandwidth() * hbm_bandwidth_utilization_rate;

  GpuDotFusionCostModel::DotProblemDimensions dims(*dot);
  PrimitiveType lhs_element_type = dot->operand(0)->shape().element_type();
  PrimitiveType rhs_element_type = dot->operand(1)->shape().element_type();
  PrimitiveType output_element_type = dot->shape().element_type();

  // Calculate the number of bytes for input reads and output writes to HBM.
  int64_t lhs_tile_bytes = CeilOfRatio<int64_t>(
      dims.b * dims.m * dims.k * BitWidth(lhs_element_type), 8);
  int64_t rhs_tile_bytes = CeilOfRatio<int64_t>(
      dims.b * dims.k * dims.n * BitWidth(rhs_element_type), 8);
  int64_t output_tile_bytes = CeilOfRatio<int64_t>(
      dims.b * dims.m * dims.n * BitWidth(output_element_type), 8);

  // Main loop loads the input matrices from HBM using SW pipelining and updates
  // accumulators stored in register files (within the SM/compute unit). The
  // epilogue loop writes the output matrices from register files to HBM. Main
  // loop and epilogue loop are executed sequentially.
  int64_t main_loop_bytes = lhs_tile_bytes + rhs_tile_bytes;
  int64_t epilogue_bytes = output_tile_bytes;

  // Calculate the HBM time using the effective bandwidth for each transfer
  // size. In the current implementation, we are assuming that the main loop and
  // epilogue loop have the same effective DRAM bandwidth. This could change in
  // the future, if we choose to model it based on their respective transfer
  // sizes.
  absl::Duration hbm_time =
      absl::Seconds(1.0f * (main_loop_bytes + epilogue_bytes) / dram_bandwidth);

  return hbm_time;
}

}  // namespace detail

namespace GpuDotFusionCostModel {

absl::Status IsSupported(const HloDotInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_batch_dimensions(),
      dim_numbers.lhs_contracting_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_batch_dimensions(),
      dim_numbers.rhs_contracting_dimensions());

  if (lhs_non_contracting_dims.size() > 1 ||
      rhs_non_contracting_dims.size() > 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Multiple non-contracting dimensions are not supported, got LHS: [",
        absl::StrJoin(lhs_non_contracting_dims, ","), "], RHS: [",
        absl::StrJoin(rhs_non_contracting_dims, ","), "]"));
  }
  // Only checking one side of batch and contracting dimensions, since they must
  // be the same for left and right.
  if (dim_numbers.lhs_batch_dimensions_size() > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Batch dimension > 1 is not supported, got ",
                     absl::StrJoin(dim_numbers.lhs_batch_dimensions(), ",")));
  }
  if (dim_numbers.lhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Exactly one contracting dimension is supported, got ",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ",")));
  }
  if (dim_numbers.lhs_contracting_dimensions(0) != 1 ||
      dim_numbers.rhs_contracting_dimensions(0) != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Only lhs_contracting_dimensions=1 (got ",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ","),
        ") and  rhs_contracting_dimensions=0 (got ",
        absl::StrJoin(dim_numbers.rhs_contracting_dimensions(), ","),
        ") are supported."));
  }

  return absl::OkStatus();
}

DotProblemDimensions::DotProblemDimensions(const HloDotInstruction& dot) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  b = dim_numbers.lhs_batch_dimensions_size() > 0
          ? dim_numbers.lhs_batch_dimensions(0)
          : 1;
  m = lhs_shape.dimensions(lhs_non_contracting_dims[0]);
  n = rhs_shape.dimensions(rhs_non_contracting_dims[0]);
  k = lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions()[0]);
}

absl::StatusOr<absl::Duration> EstimateRunTimeForDotOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info) {
  TF_RETURN_IF_ERROR(IsSupported(dot));
  if (block_params.output_tile_sizes.size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only single tile size is supported, got ",
                     block_params.output_tile_sizes.size()));
  }

  // Calculate compute roofline with tile and wave quantization.
  TF_ASSIGN_OR_RETURN(absl::Duration compute_time,
                      detail::CalculateComputeTimeWithTileAndWaveQuantization(
                          dot, block_params.output_tile_sizes[0], device_info));
  // Calculate HBM roofline.
  absl::Duration hbm_time = detail::CalculateHbmTime(dot, device_info);
  // Calculate L2 time.
  TF_ASSIGN_OR_RETURN(absl::Duration l2_time,
                      detail::CalculateL2Time(
                          dot, block_params.output_tile_sizes[0], device_info));

  // Assuming perfect overlap between compute and memory.
  return std::max({compute_time, hbm_time, l2_time});
}

absl::StatusOr<absl::Duration> EstimateRunTimeForDotOp(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  TF_RETURN_IF_ERROR(IsSupported(dot));

  // TODO(maniananth): Implement this.
  return absl::UnimplementedError("Not implemented yet");
}

absl::StatusOr<BlockLevelParameters> FindBestBlockLevelParameters(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  TF_RETURN_IF_ERROR(IsSupported(dot));

  // TODO(maniananth): Implement this.
  return absl::UnimplementedError("Not implemented yet");
}

}  // namespace GpuDotFusionCostModel

}  // namespace gpu
}  // namespace xla
