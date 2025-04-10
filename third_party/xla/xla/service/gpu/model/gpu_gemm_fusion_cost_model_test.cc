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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuGemmFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo()};
};

TEST_F(GpuGemmFusionCostModelTest, GpuGemmComputeBoundBf16) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[8192,8192] parameter(0)
p1 = bf16[8192,8192] parameter(1)
ROOT r = bf16[8192,8192] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16
})"));

  GpuGemmFusionCostModel cost_model;
  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{64, 64}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  ASSERT_OK(cost_model.CheckSupportedCheckDotDimensions(Cast<HloDotInstruction>(
      module->entry_computation()->root_instruction())));
  absl::Duration runtime =
      cost_model.EstimateRunTimeForGemmOpWithBlockParameters(
          Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          block_params, device_description_);
  absl::Duration expected_runtime_compute_bound =
      GpuGemmFusionCostModel::CalculateComputeTimeWithTileAndWaveQuant(
          Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          block_params.output_tile_sizes[0], device_description_);
  ASSERT_EQ(runtime, expected_runtime_compute_bound);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
