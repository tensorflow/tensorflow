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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[8192,8192] parameter(0)
p1 = bf16[8192,8192] parameter(1)
ROOT r = bf16[8192,8192] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  BlockLevelParameters block_params;
  // TODO: b/510666436 - Tile sizes are intentionally kept large to reduce
  // L2 cache replication overhead modeled by threadblock_count, keeping
  // the operation compute bound.
  block_params.output_tile_sizes = {{256, 512}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_));
  ASSERT_OK_AND_ASSIGN(
      auto expected_compute_and_flops_h100,
      gpu_dot_fusion_cost_model::detail::
          CalculateComputeTimeWithTileAndWaveQuantization(
              gpu_dot_fusion_cost_model::detail::DotProblemInfo(*dot),
              gpu_dot_fusion_cost_model::detail::DotTileSize{
                  block_params.output_tile_sizes[0][0],
                  block_params.output_tile_sizes[0][1]},
              ddh100_));
  ASSERT_EQ(runtime_h100.exec_time,
            expected_compute_and_flops_h100.compute_time);
}

TEST_F(GpuDotFusionCostModelTest, GpuDotMemoryBoundBf16) {
  // TODO: b/510666436 - Backend config tuned to minimize L2 loads replication
  // so the operation remains strictly HBM bounded.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[4,4096] parameter(0)
p1 = bf16[4096,4096] parameter(1)
ROOT r = bf16[4,4096] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["512"]}
})"));

  BlockLevelParameters block_params;
  // TODO: b/510666436 - Output tile sizes tuned to minimize L2 loads
  // replication so the operation remains strictly HBM bounded.
  block_params.output_tile_sizes = {{4, 128}};
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
  EXPECT_NEAR(absl::ToDoubleMicroseconds(runtime_h100.exec_time),
              absl::ToDoubleMicroseconds(approx_hbm_time), 0.25);
}

TEST_F(GpuDotFusionCostModelTest, SkinnyBf16GemmWriteAmplification) {
  // Skinny GEMM: LHS = [16, 32] (bf16), RHS = [32, 64] (bf16), Output = [16,
  // 64] Output size mathematically = 16 * 64 * 2 bytes = 2048 bytes. But NCU
  // profiles show 2,476,800 bytes written to DRAM.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,64] parameter(1)
  ROOT r = bf16[16,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    backend_config={"sizes":["32"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{16, 64}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 4;

  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));

  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_, /*block_k=*/32));

  // With M = 16 and tile_n = 64 (bf16), the contiguous write width is 128
  // bytes. Since 128 >= 128 (transaction size), there is no write coalescing
  // penalty. The DRAM write volume remains exactly the mathematical output size
  // (2048 bytes).
  EXPECT_EQ(runtime_h100.bytes_written, 2048);
}

TEST_F(GpuDotFusionCostModelTest, SkinnyGemmDramWriteVolumeMismatch) {
  // Skinny GEMM: LHS = [64, 894, 2], RHS = [48, 64, 894], Output = [64, 2, 48]
  // Mathematical Output size = 64 * 2 * 48 * 8 bytes (f64) = 49,152 bytes.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f64[64,894,2] parameter(0)
  p1 = f64[48,64,894] parameter(1)
  ROOT r = f64[64,2,48] dot(p0, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={1},
    rhs_batch_dims={1}, rhs_contracting_dims={2},
    backend_config={"sizes":["64"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{16, 32}};  // m:16 n:32
  block_params.num_warps = 2;
  block_params.num_ctas = 1;
  block_params.num_stages = 3;

  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));

  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_, /*block_k=*/64));

  // Contiguous tile write width is 32 * 8 bytes = 256 bytes.
  // Since 256 >= 128 (transaction size), there is no coalescing penalty.
  // Estimated DRAM writes remains the mathematical output size (49152 bytes).
  EXPECT_EQ(runtime_h100.bytes_written, 49152);
}

TEST_F(GpuDotFusionCostModelTest, NarrowTileWriteAmplification) {
  // RowMajor Output = [128, 128] (bf16).
  // If we use a tile size with a very narrow N (e.g. tile_n = 8),
  // the continuous write width is 8 * 2 = 16 bytes.
  // With transaction size = 128 bytes, write coalesce penalty is 128/16 = 8.0x.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = bf16[128,128] parameter(0)
  p1 = bf16[128,128] parameter(1)
  ROOT r = bf16[128,128] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    backend_config={"sizes":["128"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {
      {16, 8}};  // tile_m = 16, tile_n = 8 (narrow N)
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 4;

  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));

  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_, /*block_k=*/128));

  // Mathematical Output size = 128 * 128 * 2 bytes = 32768 bytes.
  // With write coalesce penalty 8.0x, expected bytes_written = 32768 * 8.0 =
  // 262144 bytes.
  EXPECT_EQ(runtime_h100.bytes_written, 262144);
}

TEST_F(GpuDotFusionCostModelTest, DifferentContractingDimsHaveSameRuntime) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_1_0,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[8192,1024] parameter(0)
p1 = bf16[1024,4096] parameter(1)
ROOT r = bf16[8192,4096] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_0_1,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[1024,8192] parameter(0)
p1 = bf16[4096,1024] parameter(1)
ROOT r = bf16[8192,4096] dot(p0, p1),
lhs_contracting_dims={0}, rhs_contracting_dims={1}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{128, 256}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;

  auto* dot_1_0 = Cast<HloDotInstruction>(
      module_1_0->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot_1_0));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100_1_0,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot_1_0, block_params, ddh100_));

  auto* dot_0_1 = Cast<HloDotInstruction>(
      module_0_1->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot_0_1));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100_0_1,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot_0_1, block_params, ddh100_));

  ASSERT_GT(absl::ToInt64Microseconds(runtime_h100_1_0.exec_time), 0);
  ASSERT_EQ(runtime_h100_1_0.exec_time, runtime_h100_0_1.exec_time);
}

TEST_F(GpuDotFusionCostModelTest, ExtractBlockKFromTileConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[1024,2048] parameter(0)
p1 = bf16[2048,1024] parameter(1)
ROOT r = bf16[1024,1024] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_OK_AND_ASSIGN(int64_t block_k,
                       gpu_dot_fusion_cost_model::ExtractBlockK(dot));
  EXPECT_EQ(block_k, 32);
}

TEST_F(GpuDotFusionCostModelTest, ExtractBlockKNoBackendConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[1024,2048] parameter(0)
p1 = bf16[2048,1024] parameter(1)
ROOT r = bf16[1024,1024] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16
})"));

  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  EXPECT_THAT(gpu_dot_fusion_cost_model::ExtractBlockK(dot),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GpuDotFusionCostModelTest, GpuDot3DGemmIsSupported) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[16,1024,2048] parameter(0)
p1 = bf16[16,2048,1024] parameter(1)
ROOT r = bf16[16,1024,1024] dot(p0, p1),
lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{1, 128, 256}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_));
  ASSERT_GT(absl::ToInt64Microseconds(runtime_h100.exec_time), 0);
}

// We support 4D and higher rank GEMMs to handle multi-dimensional batching
// (such as having independent head and batch dimensions in multi-head
// attention workloads) without requiring explicit reshape or flattening ops.
TEST_F(GpuDotFusionCostModelTest, GpuDot4DGemm) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[2,8,1024,2048] parameter(0)
p1 = bf16[2,8,2048,1024] parameter(1)
ROOT r = bf16[2,8,1024,1024] dot(p0, p1),
lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
})"));

  BlockLevelParameters block_params;
  block_params.output_tile_sizes = {{1, 1, 128, 256}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 1;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_));
  ASSERT_GT(absl::ToInt64Microseconds(runtime_h100.exec_time), 0);
}

// TODO: b/501002656 - Remove this test once we support transposes in the dot
// fusion cost model.
TEST_F(GpuDotFusionCostModelTest, GpuDotWithDownstreamTransposeIsRejected) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ENTRY e {
p0 = bf16[1024,2048] parameter(0)
p1 = bf16[2048,1024] parameter(1)
d = bf16[1024,1024] dot(p0, p1),
lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_bf16,
backend_config={"sizes":["32"]}
ROOT r = bf16[1024,1024] transpose(d), dimensions={1,0}
})"));

  auto* root = module->entry_computation()->root_instruction();
  auto* dot = Cast<HloDotInstruction>(root->operand(0));
  EXPECT_THAT(gpu_dot_fusion_cost_model::IsSupported(dot),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
