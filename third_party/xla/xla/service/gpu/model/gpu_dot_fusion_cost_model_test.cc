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

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuDotFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription ddh100_{TestGpuDeviceInfo::H100SXMDeviceInfo()};
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
  block_params.output_tile_sizes = {{128, 256}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  TF_ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_));
  TF_ASSERT_OK_AND_ASSIGN(
      auto expected_compute_and_flops_h100,
      gpu_dot_fusion_cost_model::detail::
          CalculateComputeTimeWithTileAndWaveQuantization(
              gpu_dot_fusion_cost_model::detail::DotProblemInfo(*dot),
              gpu_dot_fusion_cost_model::detail::OutputTileSize{
                  block_params.output_tile_sizes[0][0],
                  block_params.output_tile_sizes[0][1]},
              ddh100_));
  ASSERT_EQ(runtime_h100.exec_time,
            expected_compute_and_flops_h100.compute_time);
}

TEST_F(GpuDotFusionCostModelTest, GpuDotMemoryBoundBf16) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[4,4096] parameter(0)
p1 = bf16[4096,4096] parameter(1)
ROOT r = bf16[4,4096] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{4, 32}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  EstimateRunTimeData runtime_h100 =
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_)
          .value();
  int64_t approx_total_bytes = 2 /*BF16*/ * (4096 + 4 * 2) * 4096;
  float approx_hbm_bandwidth =
      gpu_dot_fusion_cost_model::detail::GetEffectiveHbmBandwidth(
          approx_total_bytes, ddh100_);
  absl::Duration approx_hbm_time =
      absl::Seconds(1.0f * approx_total_bytes / approx_hbm_bandwidth);
  ASSERT_EQ(runtime_h100.exec_time, approx_hbm_time);
}

TEST_F(GpuDotFusionCostModelTest, DifferentContractingDimsHaveSameRuntime) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_1_0,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[8192,1024] parameter(0)
p1 = bf16[1024,4096] parameter(1)
ROOT r = bf16[8192,4096] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16
})"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_0_1,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[1024,8192] parameter(0)
p1 = bf16[4096,1024] parameter(1)
ROOT r = bf16[8192,4096] dot(p0, p1),
lhs_contracting_dims={0}, rhs_contracting_dims={1}, algorithm=dot_bf16_bf16_bf16
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{128, 256}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;

  auto* dot_1_0 = Cast<HloDotInstruction>(
      module_1_0->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot_1_0));
  TF_ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100_1_0,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot_1_0, block_params, ddh100_));

  auto* dot_0_1 = Cast<HloDotInstruction>(
      module_0_1->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot_0_1));
  TF_ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100_0_1,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot_0_1, block_params, ddh100_));

  ASSERT_GT(absl::ToInt64Microseconds(runtime_h100_1_0.exec_time), 0);
  ASSERT_EQ(runtime_h100_1_0.exec_time, runtime_h100_0_1.exec_time);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
