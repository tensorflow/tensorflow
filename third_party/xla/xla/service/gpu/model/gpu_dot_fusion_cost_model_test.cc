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

using gpu_dot_fusion_cost_model::detail::
    CalculateComputeTimeWithTileAndWaveQuantization;
using gpu_dot_fusion_cost_model::detail::CalculateHardwareLaunchWaves;
using gpu_dot_fusion_cost_model::detail::CalculateLoopIterBytes;
using gpu_dot_fusion_cost_model::detail::
    CalculatePipelinedLoopTimeWithLaunchWaves;
using gpu_dot_fusion_cost_model::detail::CalculateSharedMemoryPerBlockBytes;
using gpu_dot_fusion_cost_model::detail::CalculateSmOccupancy;
using gpu_dot_fusion_cost_model::detail::DotProblemInfo;
using gpu_dot_fusion_cost_model::detail::DotTileSize;
using gpu_dot_fusion_cost_model::detail::GetEffectiveHbmBandwidth;
using gpu_dot_fusion_cost_model::detail::HbmEstimates;
using gpu_dot_fusion_cost_model::detail::kLoopLatencyTax;
using gpu_dot_fusion_cost_model::detail::SmOccupancy;

class GpuDotFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription ddh100_{TestGpuDeviceInfo::H100SXMDeviceInfo()};
};

TEST_F(GpuDotFusionCostModelTest, GpuDotComputeBoundBf16NumStages1) {
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
  // TODO(b/510666436): Tile sizes are intentionally kept large to reduce
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
  ASSERT_OK_AND_ASSIGN(auto expected_compute_and_flops_h100,
                       CalculateComputeTimeWithTileAndWaveQuantization(
                           DotProblemInfo(*dot),
                           DotTileSize{block_params.output_tile_sizes[0][0],
                                       block_params.output_tile_sizes[0][1]},
                           ddh100_));

  // For num_stages=1, exec_time is sequentially added: compute + mem + write.
  // We expect it to be significantly larger than just compute_time.
  EXPECT_GT(runtime_h100.exec_time,
            expected_compute_and_flops_h100.compute_time * 1.2);
}

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
  // TODO(b/510666436): Tile sizes are intentionally kept large to reduce
  // L2 cache replication overhead modeled by threadblock_count, keeping
  // the operation compute bound.
  block_params.output_tile_sizes = {{256, 512}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 3;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_h100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_));
  ASSERT_OK_AND_ASSIGN(auto expected_compute_and_flops_h100,
                       CalculateComputeTimeWithTileAndWaveQuantization(
                           DotProblemInfo(*dot),
                           DotTileSize{block_params.output_tile_sizes[0][0],
                                       block_params.output_tile_sizes[0][1]},
                           ddh100_));
  absl::Duration expected_time =
      expected_compute_and_flops_h100.compute_time + kLoopLatencyTax;
  // For pipelined loops, execution time is bounded by the dominant cost
  // (compute in this case), but imperfect overlap or pipeline setup/teardown
  // costs may slightly increase it. We allow up to 10% overhead.
  EXPECT_GE(runtime_h100.exec_time, expected_time);
  EXPECT_LE(runtime_h100.exec_time, expected_time * 1.1);
}

TEST_F(GpuDotFusionCostModelTest, GpuDotMemoryBoundBf16) {
  // TODO(b/510666436): Backend config tuned to minimize L2 loads replication
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
  // TODO(b/510666436): Output tile sizes tuned to minimize L2 loads
  // replication so the operation remains strictly HBM bounded.
  block_params.output_tile_sizes = {{4, 128}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 3;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_IS_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  EstimateRunTimeData runtime_h100 =
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, ddh100_)
          .value();
  int64_t approx_total_bytes = 2 /*BF16*/ * (4096 + 4 * 2) * 4096;
  float approx_hbm_bandwidth =
      GetEffectiveHbmBandwidth(approx_total_bytes, ddh100_);
  absl::Duration approx_hbm_time =
      absl::Seconds(1.0f * approx_total_bytes / approx_hbm_bandwidth) +
      kLoopLatencyTax;
  // For pipelined loops, execution time is bounded by the dominant cost (memory
  // in this case), but imperfect overlap or pipeline setup/teardown costs may
  // slightly increase it. We allow up to 10% overhead.
  EXPECT_GE(runtime_h100.exec_time, approx_hbm_time);
  EXPECT_LE(runtime_h100.exec_time, approx_hbm_time * 1.1);
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

  EXPECT_GT(absl::ToInt64Microseconds(runtime_h100_1_0.exec_time), 0);
  EXPECT_EQ(runtime_h100_1_0.exec_time, runtime_h100_0_1.exec_time);
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
  EXPECT_GT(absl::ToInt64Microseconds(runtime_h100.exec_time), 0);
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
  EXPECT_GT(absl::ToInt64Microseconds(runtime_h100.exec_time), 0);
}

// TODO(b/501002656): Remove this test once we support transposes in the dot
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

TEST_F(GpuDotFusionCostModelTest, CalculateIterBytes) {
  DotProblemInfo dot_info;
  dot_info.b = 1;
  dot_info.m = 1024;
  dot_info.n = 1024;
  dot_info.k = 2048;
  dot_info.lhs_element_type = PrimitiveType::BF16;
  dot_info.rhs_element_type = PrimitiveType::BF16;

  DotTileSize dot_tile{/*m=*/128, /*n=*/256, /*k=*/32, /*b=*/1};

  // lhs_iter_bytes = ceil(1 * 128 * 32 * 2 (bf16 - 2 bytes)) = 8192
  // rhs_iter_bytes = ceil(1 * 32 * 256 * 2 (bf16 - 2 bytes)) = 16384
  // total = 8192 + 16384 = 24576
  int64_t iter_bytes = CalculateLoopIterBytes(dot_info, dot_tile);
  EXPECT_EQ(iter_bytes, 24576);
}

TEST_F(GpuDotFusionCostModelTest, CalculateSharedMemoryPerBlockBytes) {
  DotProblemInfo dot_info_f32;
  dot_info_f32.lhs_element_type = PrimitiveType::F32;
  dot_info_f32.rhs_element_type = PrimitiveType::F32;

  // Tile size: (64*16*4) + (64*16*4) = 8192 bytes.
  // stages=3 -> 8192 * 3 = 24576 bytes.
  DotTileSize dot_tile_16{/*m=*/64, /*n=*/64, /*k=*/16, /*b=*/1};
  EXPECT_EQ(24576, CalculateSharedMemoryPerBlockBytes(dot_info_f32, dot_tile_16,
                                                      /*num_stages=*/3));

  // Tile size: (64*64*4) + (16*64*4) = 20480 bytes.
  // stages=4 -> 20480 * 4 = 81920 bytes.
  DotTileSize dot_tile_64{/*m=*/64, /*n=*/16, /*k=*/64, /*b=*/1};
  EXPECT_EQ(81920, CalculateSharedMemoryPerBlockBytes(dot_info_f32, dot_tile_64,
                                                      /*num_stages=*/4));

  DotProblemInfo dot_info_f64;
  dot_info_f64.lhs_element_type = PrimitiveType::F64;
  dot_info_f64.rhs_element_type = PrimitiveType::F64;

  // Tile size: (64*16*8) + (64*16*8) = 16384 bytes.
  // stages=1 -> 16384 * 1 = 16384 bytes.
  DotTileSize dot_tile_f64_16{/*m=*/64, /*n=*/64, /*k=*/16, /*b=*/1};
  EXPECT_EQ(16384, CalculateSharedMemoryPerBlockBytes(
                       dot_info_f64, dot_tile_f64_16, /*num_stages=*/1));
}

TEST_F(GpuDotFusionCostModelTest, CalculateSmOccupancy_ShmemLimited) {
  // Large shared memory should limit the occupancy to 1 block per SM.
  const SmOccupancy occupancy = CalculateSmOccupancy(
      /*shared_memory_per_block_bytes=*/200000,
      /*num_warps=*/4, ddh100_);
  EXPECT_EQ(occupancy.active_blocks_per_sm, 1);
  EXPECT_EQ(occupancy.active_warps_per_sm, 4);
}

TEST_F(GpuDotFusionCostModelTest, CalculateSmOccupancy_ThreadLimited) {
  const SmOccupancy occupancy = CalculateSmOccupancy(
      /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_);
  // H100 has 2048 threads per SM. 4 warps * 32 threads/warp = 128
  // threads/block. 2048 / 128 = 16 blocks per SM maximum.
  EXPECT_EQ(occupancy.active_blocks_per_sm, 16);
  EXPECT_EQ(occupancy.active_warps_per_sm, 64);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_ZeroBlocks) {
  // Zero threadblocks should require zero waves.
  EXPECT_EQ(0,
            CalculateHardwareLaunchWaves(/*threadblock_count=*/0,
                                         /*shared_memory_per_block_bytes=*/1024,
                                         /*num_warps=*/4, ddh100_));
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateHardwareLaunchWaves_SmallShmemFewBlocks) {
  // Small shared memory with few threadblocks should require 1 wave.
  int64_t small_shmem_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/1000, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_);
  EXPECT_EQ(1, small_shmem_waves);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_LargeShmem) {
  // Large shared memory should require more waves to execute the same
  // number of blocks.
  int64_t large_shmem_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/1000, /*shared_memory_per_block_bytes=*/200000,
      /*num_warps=*/4, ddh100_);
  EXPECT_GE(large_shmem_waves, 4);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_MoreBlocks) {
  // More threadblocks requires more waves.
  int64_t more_blocks_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/5000, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_);
  EXPECT_GT(more_blocks_waves, 1);
}

TEST_F(GpuDotFusionCostModelTest, CalculatePipelinedLoopTime) {
  HbmEstimates hbm_timing;
  hbm_timing.read_time = absl::Microseconds(100);
  hbm_timing.write_time = absl::Microseconds(50);
  absl::Duration compute_time = absl::Microseconds(200);
  const int64_t k_loop_iterations = 10;

  // Serial Execution: num_stages = 1
  absl::Duration serial_time = CalculatePipelinedLoopTime(
      /*num_stages=*/1, k_loop_iterations, compute_time, hbm_timing);

  // Pipelined Execution: num_stages = 3
  absl::Duration pipelined_time = CalculatePipelinedLoopTime(
      /*num_stages=*/3, k_loop_iterations, compute_time, hbm_timing);

  // Serial time should be roughly comparable to the sum of independent work.
  absl::Duration independent_work_time =
      hbm_timing.read_time + hbm_timing.write_time + compute_time;
  EXPECT_GE(serial_time, independent_work_time);
  EXPECT_LE(serial_time, independent_work_time * 1.1);

  // Pipelined time should be significantly faster than independent work.
  EXPECT_LT(pipelined_time, independent_work_time * 0.9);
}

TEST_F(GpuDotFusionCostModelTest,
       CalculatePipelinedLoopTimeWithLaunchWaves_ZeroBlocksHazard) {
  HbmEstimates hbm_timing;
  hbm_timing.read_time = absl::Microseconds(100);
  hbm_timing.write_time = absl::Microseconds(50);
  absl::Duration compute_time = absl::Microseconds(200);
  const int64_t k_loop_iterations = 10;

  // A configuration with no threadblocks should result in zero execution time.
  EXPECT_EQ(
      absl::ZeroDuration(),
      CalculatePipelinedLoopTimeWithLaunchWaves(
          /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/0,
          compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
          /*num_warps=*/4, ddh100_));
}

TEST_F(GpuDotFusionCostModelTest,
       CalculatePipelinedLoopTimeWithLaunchWaves_WaveBoundaryOverhead) {
  HbmEstimates hbm_timing;
  hbm_timing.read_time = absl::Microseconds(100);
  hbm_timing.write_time = absl::Microseconds(50);
  absl::Duration compute_time = absl::Microseconds(200);
  const int64_t k_loop_iterations = 10;

  // Wave boundary execution overhead. Many waves should be slower than
  // a perfectly scheduled 1-wave pipeline.
  absl::Duration result_one_wave = CalculatePipelinedLoopTimeWithLaunchWaves(
      /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/1,
      compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_);

  absl::Duration result_more_blocks_still_one_wave =
      CalculatePipelinedLoopTimeWithLaunchWaves(
          /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/1000,
          compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
          /*num_warps=*/4, ddh100_);

  absl::Duration result_many_waves = CalculatePipelinedLoopTimeWithLaunchWaves(
      /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/5000,
      compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_);

  EXPECT_EQ(result_one_wave, result_more_blocks_still_one_wave);
  EXPECT_GT(result_many_waves, result_one_wave);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
