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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_GEMM_FUSION_COST_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_GEMM_FUSION_COST_MODEL_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuGemmFusionCostModel {
 public:
  absl::Status CheckSupportedCheckDotDimensions(const HloDotInstruction* dot);

  static absl::Duration EstimateRunTimeForGemmOpWithBlockParameters(
      const HloDotInstruction* dot, const BlockLevelParameters& block_params,
      const se::DeviceDescription& device_info);

  static absl::Duration EstimateRunTimeForGemmOp(
      const HloDotInstruction* dot, const se::DeviceDescription& device_info);

  static BlockLevelParameters FindBestBlockLevelParameters(
      const HloDotInstruction* dot, const se::DeviceDescription& device_info);

  static absl::Duration CalculateComputeTimeWithTileAndWaveQuant(
      const HloDotInstruction* dot, const std::vector<int64_t>& tile_size,
      const se::DeviceDescription& device_info);

 protected:
  static std::tuple<int64_t, int64_t, int64_t, int64_t> get_bmnk(
      const HloDotInstruction& dot);
  llvm::SmallVector<BlockLevelParameters> GetGemmAlgorithmValidConfigs(
      const HloDotInstruction* dot, const se::DeviceDescription& device_info);
  bool tileFitsInRegisters(int64_t block_m, int64_t block_n,
                           const PrecisionConfig& precision_config,
                           const se::DeviceDescription& device_info);
  static int GetInputBytesPerElement(const PrecisionConfig& precision_config);
  static int GetOutputBytesPerElement(const PrecisionConfig& precision_config);
  static int64_t CalcNumThreadblocks(const HloDotInstruction* dot,
                                     const std::vector<int64_t>& tile_size);
  static int64_t CalcNumWaves(int64_t threadblock_count,
                              const se::DeviceDescription& device_info);
  static int64_t CalcTileFlops(const std::vector<int64_t>& tile_size,
                               int64_t problem_k);
  static double GetEffectiveFlopsPerNsForTileSize(
      const std::vector<int64_t>& tile_size,
      const se::DeviceDescription& device_info);
  static absl::Duration CalculateHbmTime(
      const HloDotInstruction* dot, const se::DeviceDescription& device_info);
  static int64_t CalculateL2Bytes(const std::vector<int64_t>& tile_size,
                                  int64_t problem_k, int64_t threadblock_count);
  static absl::Duration CalculateL2Time(
      const HloDotInstruction* dot, const std::vector<int64_t>& tile_size,
      const se::DeviceDescription& device_info);

 private:
  static const int kMinBlockDim = 32;
  static const int kMaxBlockDim = 256;
  static const int kMaxSplitK = 128;
  static const int kNumWarpsPerBlock = 4;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_GEMM_FUSION_COST_MODEL_H_
