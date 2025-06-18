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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace GpuDotFusionCostModel {

struct DotProblemDimensions {
  int64_t b;
  int64_t m;
  int64_t n;
  int64_t k;

  explicit DotProblemDimensions(const HloDotInstruction& dot);
};

// Returns OkStatus if the dot operation is supported by the cost model.
absl::Status IsSupported(const HloDotInstruction* dot);

// Estimates the run time for a GPU DOT operation with the given set of block
// parameters.
absl::StatusOr<absl::Duration> EstimateRunTimeForDotOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info);

// Estimates the run time for a GPU DOT operation.
absl::StatusOr<absl::Duration> EstimateRunTimeForDotOp(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info);

absl::StatusOr<BlockLevelParameters> FindBestBlockLevelParameters(
    const HloDotInstruction* dot, const se::DeviceDescription& device_info);
}  // namespace GpuDotFusionCostModel

namespace detail {

// Calculates the HBM time for a GPU DOT operation. Current implementation
// uses a flat derate on top of the spec bandwidth. A HBM bandwidth model based
// derate lookup from profiled data will be added in the future.
absl::Duration CalculateHbmTime(const HloDotInstruction* dot,
                                const se::DeviceDescription& device_info);

// Calculates the L2 time for a GPU DOT operation.
absl::StatusOr<absl::Duration> CalculateL2Time(
    const HloDotInstruction* dot, absl::Span<const int64_t> tile_shape,
    const se::DeviceDescription& device_info);

// Calculates the compute time for a GPU DOT operation with tile and wave
// quantization effects taken into account.
// (1) Tile Quantization effects occur when the input problem dimensions are
//     quantized to the tile shape.
// (2) Wave Quantization effects occur when the number of threadblocks is
//     quantized to the number of SMs per GPU.
absl::StatusOr<absl::Duration> CalculateComputeTimeWithTileAndWaveQuantization(
    const HloDotInstruction* dot, absl::Span<const int64_t> tile_shape,
    const se::DeviceDescription& device_info);

const int kMinBlockDim = 32;
const int kMaxBlockDim = 256;
const int kMaxSplitK = 128;
const int kNumWarpsPerBlock = 4;
}  // namespace detail

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_DOT_FUSION_COST_MODEL_H_
