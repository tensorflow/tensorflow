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

#include "xla/service/gpu/model/gpu_gemm_fusion_cost_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Returns OkStatus if the dot operation is supported by the cost model.
absl::Status GpuGemmFusionCostModel::CheckSupportedCheckDotDimensions(
    const HloDotInstruction* dot) {
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

std::tuple<int64_t, int64_t, int64_t, int64_t> GpuGemmFusionCostModel::get_bmnk(
    const HloDotInstruction& dot) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  int64_t b = dim_numbers.lhs_batch_dimensions_size() > 0
                  ? dim_numbers.lhs_batch_dimensions(0)
                  : 1;
  int64_t m = lhs_shape.dimensions(lhs_non_contracting_dims[0]);
  int64_t n = rhs_shape.dimensions(rhs_non_contracting_dims[0]);
  int64_t k = lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions()[0]);

  return std::make_tuple(b, m, n, k);
}

int GpuGemmFusionCostModel::GetInputBytesPerElement(
    const PrecisionConfig& precision_config) {
  int bytes_per_input_elem = 1;
  switch (precision_config.algorithm()) {
    case PrecisionConfig::ALG_UNSET:
      assert(false);
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      bytes_per_input_elem = 1;
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      bytes_per_input_elem = 2;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      bytes_per_input_elem = 4;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      bytes_per_input_elem = 2 * 3;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      bytes_per_input_elem = 4 * 3;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      bytes_per_input_elem = 2 * 6;
      break;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      bytes_per_input_elem = 8;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      bytes_per_input_elem = 2 * 9;
      break;
    default:
      bytes_per_input_elem = 0;
      break;
  }

  return bytes_per_input_elem;
}

int GpuGemmFusionCostModel::GetOutputBytesPerElement(
    const PrecisionConfig& precision_config) {
  int bytes_per_output_elem = 1;
  switch (precision_config.algorithm()) {
    case PrecisionConfig::ALG_UNSET:
      return 0;
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
      bytes_per_output_elem = 2;
      break;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      bytes_per_output_elem = 4;
      break;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      bytes_per_output_elem = 8;
      break;
    default:
      bytes_per_output_elem = 0;
      break;
  }

  return bytes_per_output_elem;
}

bool GpuGemmFusionCostModel::tileFitsInRegisters(
    int64_t block_m, int64_t block_n, const PrecisionConfig& precision_config,
    const se::DeviceDescription& device_info) {
  int bytes_per_output_elem = GetOutputBytesPerElement(precision_config);
  int registers_per_block = device_info.registers_per_block_limit();
  int64_t block_size = block_m * block_n;
  int64_t bytes_per_block = block_size * bytes_per_output_elem;
  return bytes_per_block <= registers_per_block;
}

llvm::SmallVector<BlockLevelParameters>
GpuGemmFusionCostModel::GetGemmAlgorithmValidConfigs(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  llvm::SmallVector<BlockLevelParameters> valid_configs;

  for (int64_t block_m = GpuGemmFusionCostModel::kMinBlockDim;
       block_m <= GpuGemmFusionCostModel::kMaxBlockDim; block_m *= 2) {
    for (int64_t block_n = GpuGemmFusionCostModel::kMinBlockDim;
         block_n <= GpuGemmFusionCostModel::kMaxBlockDim; block_n *= 2) {
      bool is_valid_config = tileFitsInRegisters(
          block_m, block_n, dot->precision_config(), device_info);
      if (!is_valid_config) {
        continue;
      }

      // TODO(maniananth): Add the logic to find valid kBlock stages.
      BlockLevelParameters block_level_parameters;
      block_level_parameters.output_tile_sizes.push_back(
          std::vector<int64_t>{block_m, block_n});
      // TODO(maniananth): Add the logic to sweep num warps per block.
      block_level_parameters.num_warps =
          GpuGemmFusionCostModel::kNumWarpsPerBlock;
      valid_configs.push_back(block_level_parameters);
    }
  }

  return valid_configs;
}

absl::Duration GpuGemmFusionCostModel::CalculateComputeTimeWithTileAndWaveQuant(
    const HloDotInstruction* dot, const std::vector<int64_t>& tile_size,
    const se::DeviceDescription& device_info) {
  // Wave Quantization effects occur when the number of threadblocks is
  // quantized to the number of SMs per GPU. More details in
  // go/detailed-gpu-gemm-modeling.
  int64_t problem_k;
  std::tie(std::ignore, std::ignore, std::ignore, problem_k) = get_bmnk(*dot);
  int64_t threadblock_count =
      GpuGemmFusionCostModel::CalcNumThreadblocks(dot, tile_size);
  int64_t wave_count =
      GpuGemmFusionCostModel::CalcNumWaves(threadblock_count, device_info);
  int64_t flops_per_tile =
      GpuGemmFusionCostModel::CalcTileFlops(tile_size, problem_k);
  // The following is not the actual number of threadblocks launched, but due to
  // how wave quantization works, we get the effect of running extra
  // threadblocks when adding to roofline projections.
  int64_t cta_count_with_wave_quant = wave_count * device_info.core_count();
  int64_t total_flops_with_wave_quant =
      flops_per_tile * cta_count_with_wave_quant;
  double effective_flops =
      GetEffectiveFlopsPerNsForTileSize(tile_size, device_info);
  // TODO(maniananth): Add a cap for power throttling here.
  return absl::Nanoseconds(1.0f * total_flops_with_wave_quant /
                           effective_flops);
}

absl::Duration
GpuGemmFusionCostModel::EstimateRunTimeForGemmOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info) {
  // Calculate compute roofline with tile quantization.
  absl::Duration compute_time =
      GpuGemmFusionCostModel::CalculateComputeTimeWithTileAndWaveQuant(
          dot, block_params.output_tile_sizes[0], device_info);
  // Calculate HBM roofline.
  absl::Duration hbm_time =
      GpuGemmFusionCostModel::CalculateHbmTime(dot, device_info);
  // Calculate L2 time.
  absl::Duration l2_time = GpuGemmFusionCostModel::CalculateL2Time(
      dot, block_params.output_tile_sizes[0], device_info);

  // Assuming perfect overlap between compute and memory.
  return std::max({compute_time, hbm_time, l2_time});
}

absl::Duration GpuGemmFusionCostModel::EstimateRunTimeForGemmOp(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  // TODO(maniananth): Implement this.
  assert(false);
  return absl::ZeroDuration();
}

BlockLevelParameters GpuGemmFusionCostModel::FindBestBlockLevelParameters(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  // TODO(maniananth): Implement this.
  assert(false);
  return absl::ZeroDuration();
}

int64_t GpuGemmFusionCostModel::CalcNumThreadblocks(
    const HloDotInstruction* dot, const std::vector<int64_t>& tile_size) {
  int64_t b, m, n, k;
  std::tie(b, m, n, k) = GpuGemmFusionCostModel::get_bmnk(*dot);
  int64_t tile_m = tile_size[0], tile_n = tile_size[1], tile_k = k;
  int64_t num_tiles_along_m_dimension = (m + tile_m - 1) / tile_m;
  int64_t num_tiles_along_n_dimension = (n + tile_n - 1) / tile_n;
  int64_t num_tiles_along_k_dimension = (k + tile_k - 1) / tile_k;
  // TODO(maniananth): Add special handling for grouped matmuls here.
  int64_t num_threadblocks = b * num_tiles_along_m_dimension *
                             num_tiles_along_n_dimension *
                             num_tiles_along_k_dimension;

  return num_threadblocks;
}

int64_t GpuGemmFusionCostModel::CalcNumWaves(
    int64_t threadblock_count, const se::DeviceDescription& device_info) {
  int core_count = device_info.core_count();
  return (threadblock_count + (core_count - 1)) / core_count;
}

int64_t GpuGemmFusionCostModel::CalcTileFlops(
    const std::vector<int64_t>& tile_size, int64_t problem_k) {
  return 2 * tile_size[0] * tile_size[1] * problem_k;
}

double GpuGemmFusionCostModel::GetEffectiveFlopsPerNsForTileSize(
    const std::vector<int64_t>& tile_size,
    const se::DeviceDescription& device_info) {
  int sm_major_version = device_info.cuda_compute_capability().major;
  assert(sm_major_version > 0);

  // Not all tile sizes are equally able to extract utilization on the same
  // generation GPUs even if the workload is compute bound. GEMM performance
  // is sensitive to the tensor core instruction throughputs that the
  // programming model exposes. More details in go/detailed-gpu-gemm-modeling.
  int64_t tile_m = tile_size[0];

  // peak flops per ns for device.
  int64_t peak_flops_per_ns =
      GpuPerformanceModelBase::CalculateEffectiveFlopsPerNs(
          device_info, device_info.fpus_per_core(), device_info.core_count());

  // Final flops derate factor.
  double flops_derate = 1.0;

  if (sm_major_version == 10) {
    if (tile_m < 128) {
      // TODO(maniananth): Update this derate once we have more data from
      // actual measurements on Blackwell. For now, we are applying a 50%
      // derate to account for smaller M shapes.
      flops_derate = 0.5;
    }
  } else if (sm_major_version == 9) {
    if (tile_m < 64) {
      // Having a tile size M < 64 will lead to not being able to use the H100
      // tensor core instructions (wgmma). Defaulting to wmma instructions from
      // A100 can result in a 63% derate in flops as benchmarked by HazyResearch
      // as part of ThunderKittens work.
      // (https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
      flops_derate = 0.63;
    }
  } else if (sm_major_version == 8) {
    if (tile_m < 16) {
      // A100 tensor core instructions are effective at tile_m >= 16. We're
      // applying a 50% derate to account for this.
      flops_derate = 0.5;
    }
  }
  return peak_flops_per_ns * flops_derate;
}

absl::Duration GpuGemmFusionCostModel::CalculateHbmTime(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info) {
  // TODO(maniananth): Implement HBM derate lookup using profiled tables.
  float hbm_bandwidth_utilization_rate = 0.8;
  float dram_bandwidth =
      device_info.memory_bandwidth() * hbm_bandwidth_utilization_rate;

  int64_t b, m, n, k;
  std::tie(b, m, n, k) = GpuGemmFusionCostModel::get_bmnk(*dot);

  PrecisionConfig precision_config = dot->precision_config();

  // Calculate the number of bytes for input reads and output writes to HBM.
  int64_t lhs_tile_bytes =
      b * m * k *
      GpuGemmFusionCostModel::GetInputBytesPerElement(precision_config);
  int64_t rhs_tile_bytes =
      b * k * n *
      GpuGemmFusionCostModel::GetInputBytesPerElement(precision_config);
  int64_t output_tile_bytes =
      b * m * n *
      GpuGemmFusionCostModel::GetOutputBytesPerElement(precision_config);

  // Main loop loads the input matrices from HBM using SW pipelining and updates
  // accumulators stored in register files (within the SM/compute unit). The
  // epilogue loop writes the output matrices from register files to HBM. Main
  // loop and epilogue loop are executed sequentially.
  int64_t main_loop_bytes = lhs_tile_bytes + rhs_tile_bytes;
  int64_t epilogue_bytes = output_tile_bytes;

  // Calculate the HBM time using the effective bandwidth for each transfer
  // size.
  absl::Duration hbm_time = absl::ZeroDuration();
  for (int64_t transfer_bytes :
       std::vector<int64_t>{main_loop_bytes, epilogue_bytes}) {
    hbm_time += absl::Seconds(1.0f * transfer_bytes / dram_bandwidth);
  }

  return hbm_time;
}

int64_t GpuGemmFusionCostModel::CalculateL2Bytes(
    const std::vector<int64_t>& tile_size, int64_t problem_k,
    int64_t threadblock_count) {
  // When tiling the GEMM problem on the outputs and mapping one tile per SM,
  // the problem of data replication (or extra loads of the same data) between
  // multiple SMs occurs. This leads to more data loads than what’s expected
  // algorithmically, and increases bandwidth needs on the L2 → SM paths.

  // Input data loaded by each tile is equal to (Tile_M + Tile_N) * Tile_K
  // bytes.
  int64_t l2_data_per_tile = (tile_size[0] + tile_size[1]) * problem_k;

  // Across all the tiles, data loads will be equal to: (l2_data_per_tile *
  // threadblock_count).

  // Since H100, threadblocks within the same cluster will avoid redundant loads
  // by reading from L2 cache once and multicasting the data to all threadblocks
  // within the cluster. This is controlled programmatically and most performant
  // GEMM implementations will use this feature. To model this, we scale the
  // total data loads by the total number of threadblocks in a cluster.

  // On A100 and older GPUs, we will not see this behavior and the total data
  // loads will be equal to (l2_data_per_tile * threadblock_count). Hence the
  // cluster shape can be set to (1x1).
  // TODO(maniananth): Account for Threadblock clusters here.
  int64_t total_l2_data = ceil(l2_data_per_tile * threadblock_count);
  return total_l2_data;
}

absl::Duration gpu::GpuGemmFusionCostModel::CalculateL2Time(
    const HloDotInstruction* dot, const std::vector<int64_t>& tile_size,
    const se::DeviceDescription& device_info) {
  int64_t problem_k;
  std::tie(std::ignore, std::ignore, std::ignore, problem_k) = get_bmnk(*dot);
  int64_t threadblock_count =
      GpuGemmFusionCostModel::CalcNumThreadblocks(dot, tile_size);
  // TODO(maniananth): This has been hardcoded for H100 based on
  // microbenchmarking L2 bandwidth within a partition, but we should add this
  // to the device info and extend for more GPUs.
  double device_l2_bandwidth = 6.65 * 1e12;

  return absl::Seconds(
      1.0f * CalculateL2Bytes(tile_size, problem_k, threadblock_count) /
      device_l2_bandwidth);
}

}  // namespace gpu
}  // namespace xla
