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

#include <memory>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuDotFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription dda6000_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  se::DeviceDescription ddh100_{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()};
};

TEST_F(GpuDotFusionCostModelTest, GpuDotComputeBoundBf16) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[8192,8192] parameter(0)
p1 = bf16[8192,8192] parameter(1)
ROOT r = bf16[8192,8192] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{16, 16}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(GpuDotFusionCostModel::IsSupported(dot));
  absl::Duration runtime_a6000 =
      GpuDotFusionCostModel::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, dda6000_)
          .value();
  absl::Duration expected_runtime_compute_bound_a6000 =
      detail::CalculateComputeTimeWithTileAndWaveQuantization(
          dot, block_params.output_tile_sizes[0], dda6000_)
          .value();
  ASSERT_EQ(runtime_a6000, expected_runtime_compute_bound_a6000);

  block_params.output_tile_sizes = {{64, 64}};
  absl::Duration runtime_h100 =
      GpuDotFusionCostModel::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_)
          .value();
  absl::Duration expected_runtime_compute_bound_h100 =
      detail::CalculateComputeTimeWithTileAndWaveQuantization(
          dot, block_params.output_tile_sizes[0], ddh100_)
          .value();
  ASSERT_EQ(runtime_h100, expected_runtime_compute_bound_h100);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
