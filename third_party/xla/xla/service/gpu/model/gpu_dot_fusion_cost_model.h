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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_DOT_FUSION_COST_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_DOT_FUSION_COST_MODEL_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::gpu_dot_fusion_cost_model {

// Returns OkStatus if the dot operation is supported by the cost model.
absl::Status IsSupported(const HloDotInstruction* dot);

// Extracts the contracting dimension size (block_k) from the backend config.
absl::StatusOr<int64_t> ExtractBlockK(const HloDotInstruction* dot);

// Estimates the run time for a GPU DOT operation with the given set of block
// parameters.
// Flops with tile and wave quant.
absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForDotOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info,
    std::optional<int64_t> block_k = std::nullopt);

namespace detail {

struct DotProblemInfo {
  int64_t b = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  xla::PrimitiveType lhs_element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  xla::PrimitiveType rhs_element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  xla::PrimitiveType output_element_type =
      PrimitiveType::PRIMITIVE_TYPE_INVALID;

  explicit DotProblemInfo(const HloDotInstruction& dot);
};

struct DotTileSize {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t b = 1;
};

// Returns the effective HBM bandwidth in bytes per second for a given dma_size.
// dma_size is the total amount of data transferred to/from HBM in bytes.
float GetEffectiveHbmBandwidth(int64_t dma_size,
                               const se::DeviceDescription& device_info);

// Calculates the HBM time for a GPU DOT operation. Current implementation
// uses a flat derate on top of the spec bandwidth. A HBM bandwidth model based
// derate lookup from profiled data will be added in the future.
struct HbmEstimates {
  absl::Duration read_time;
  absl::Duration write_time;
  int64_t bytes_read = 0;
  int64_t bytes_written = 0;

  absl::Duration total_time() { return read_time + write_time; }
};
HbmEstimates CalculateHbmTime(const DotProblemInfo& dot,
                              const se::DeviceDescription& device_info);

// Calculates the L2 time for a GPU DOT operation.
absl::StatusOr<absl::Duration> CalculateL2Time(
    const DotProblemInfo& dot, const DotTileSize& dot_tile,
    const se::DeviceDescription& device_info, bool is_tma_allowed);

// Calculates the compute time for a GPU DOT operation with tile and wave
// quantization effects taken into account.
// (1) Tile Quantization effects occur when the input problem dimensions are
//     quantized to the tile shape.
// (2) Wave Quantization effects occur when the number of threadblocks is
//     quantized to the number of SMs per GPU.
struct ComputeAndFlops {
  absl::Duration compute_time = absl::ZeroDuration();
  int64_t flops_with_wave_quant = 0;
};

absl::StatusOr<ComputeAndFlops> CalculateComputeTimeWithTileAndWaveQuantization(
    const DotProblemInfo& dot, const DotTileSize& dot_tile,
    const se::DeviceDescription& device_info);

}  // namespace detail

}  // namespace xla::gpu::gpu_dot_fusion_cost_model

#endif  // XLA_SERVICE_GPU_MODEL_GPU_DOT_FUSION_COST_MODEL_H_
