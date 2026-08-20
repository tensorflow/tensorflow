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
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
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
using gpu_dot_fusion_cost_model::detail::CalculateRegistersPerThread;
using gpu_dot_fusion_cost_model::detail::CalculateSharedMemoryPerBlockBytes;
using gpu_dot_fusion_cost_model::detail::CalculateSmOccupancy;
using gpu_dot_fusion_cost_model::detail::ComputeAndFlops;
using gpu_dot_fusion_cost_model::detail::DotProblemInfo;
using gpu_dot_fusion_cost_model::detail::DotTileSize;
using gpu_dot_fusion_cost_model::detail::GetEffectiveFlopsPerNsForTileSize;
using gpu_dot_fusion_cost_model::detail::GetEffectiveHbmBandwidth;
using gpu_dot_fusion_cost_model::detail::HbmEstimates;
using gpu_dot_fusion_cost_model::detail::kLoopLatencyTax;
using gpu_dot_fusion_cost_model::detail::SmOccupancy;

class GpuDotFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription dda100_{TestGpuDeviceInfo::A100SXMDeviceInfo()};
  se::DeviceDescription ddh100_{TestGpuDeviceInfo::H100SXMDeviceInfo()};
  se::DeviceDescription ddb200_{TestGpuDeviceInfo::B200SXMDeviceInfo()};
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
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);
  EXPECT_EQ(occupancy.active_blocks_per_sm, 1);
  EXPECT_EQ(occupancy.active_warps_per_sm, 4);
}

TEST_F(GpuDotFusionCostModelTest, CalculateSmOccupancy_ThreadLimited) {
  const SmOccupancy occupancy = CalculateSmOccupancy(
      /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);
  // H100 has 2048 threads per SM. 4 warps * 32 threads/warp = 128
  // threads/block. 2048 / 128 = 16 blocks per SM maximum.
  EXPECT_EQ(occupancy.active_blocks_per_sm, 16);
  EXPECT_EQ(occupancy.active_warps_per_sm, 64);
}

TEST_F(GpuDotFusionCostModelTest, CalculateSmOccupancy_RegsLimited) {
  // H100 has 65536 registers per SM. 4 warps * 32 threads/warp = 128
  // threads/block.
  // At 200 registers/thread, 1 block requires 25600 registers.
  // 65536 / 25600 = 2 blocks per SM.
  const SmOccupancy occupancy = CalculateSmOccupancy(
      /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/200);
  EXPECT_EQ(occupancy.active_blocks_per_sm, 2);
  EXPECT_EQ(occupancy.active_warps_per_sm, 8);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_ZeroBlocks) {
  // Zero threadblocks should require zero waves.
  EXPECT_EQ(0,
            CalculateHardwareLaunchWaves(/*threadblock_count=*/0,
                                         /*shared_memory_per_block_bytes=*/1024,
                                         /*num_warps=*/4, ddh100_,
                                         /*registers_per_thread=*/0));
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateHardwareLaunchWaves_SmallShmemFewBlocks) {
  // Small shared memory with few threadblocks should require 1 wave.
  int64_t small_shmem_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/1000, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);
  EXPECT_EQ(1, small_shmem_waves);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_LargeShmem) {
  // Large shared memory should require more waves to execute the same
  // number of blocks.
  int64_t large_shmem_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/1000, /*shared_memory_per_block_bytes=*/200000,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);
  EXPECT_GE(large_shmem_waves, 4);
}

TEST_F(GpuDotFusionCostModelTest, CalculateHardwareLaunchWaves_MoreBlocks) {
  // More threadblocks requires more waves.
  int64_t more_blocks_waves = CalculateHardwareLaunchWaves(
      /*threadblock_count=*/5000, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);
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
          /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0));
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
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);

  absl::Duration result_more_blocks_still_one_wave =
      CalculatePipelinedLoopTimeWithLaunchWaves(
          /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/1000,
          compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
          /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);

  absl::Duration result_many_waves = CalculatePipelinedLoopTimeWithLaunchWaves(
      /*num_stages=*/3, k_loop_iterations, /*threadblock_count=*/5000,
      compute_time, hbm_timing, /*shared_memory_per_block_bytes=*/1024,
      /*num_warps=*/4, ddh100_, /*registers_per_thread=*/0);

  EXPECT_EQ(result_one_wave, result_more_blocks_still_one_wave);
  EXPECT_GT(result_many_waves, result_one_wave);
}

TEST_F(GpuDotFusionCostModelTest, CalculateComputeUtilization) {
  EstimateRunTimeData estimates = {};
  estimates.exec_time = absl::Seconds(4);

  int64_t theoretical_ops_per_second =
      GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(ddh100_,
                                                           PrimitiveType::F32) *
      1e9;
  // Set flops such that Compute Utilization is 0.5.
  // 0.5 = compute_utilization = flops / (theoretical * exec_time)  =>
  // flops = (theoretical * 4s) * 0.5  => flops = 2.0 * theoretical
  estimates.flops = static_cast<int64_t>(2.0 * theoretical_ops_per_second);

  // Compute Utilization: flops / (theoretical * exec_time(4s)) = 0.5
  EXPECT_DOUBLE_EQ(
      gpu_dot_fusion_cost_model::detail::CalculateComputeUtilization(
          estimates, ddh100_, PrimitiveType::F32),
      0.5);

  estimates.exec_time = absl::ZeroDuration();
  // We default to 0.0 compute utilization if the execution time is zero.
  EXPECT_DOUBLE_EQ(
      gpu_dot_fusion_cost_model::detail::CalculateComputeUtilization(
          estimates, ddh100_, PrimitiveType::F32),
      0.0);
}

TEST_F(GpuDotFusionCostModelTest, CalculateMemoryUtilization) {
  EstimateRunTimeData estimates = {};
  estimates.bytes_read = 1000;
  estimates.bytes_written = 2000;
  estimates.exec_time = absl::Seconds(4);

  ddh100_.set_memory_bandwidth(3000);

  // Memory roofline: (1000 + 2000) / 3000 B/s = 1.0s
  // Memory Utilization: roofline (1s) / exec_time (4s) = 0.25
  EXPECT_DOUBLE_EQ(
      gpu_dot_fusion_cost_model::detail::CalculateMemoryUtilization(estimates,
                                                                    ddh100_),
      0.25);

  estimates.exec_time = absl::ZeroDuration();
  // We default to 0.0 memory utilization if the execution time is zero.
  EXPECT_DOUBLE_EQ(
      gpu_dot_fusion_cost_model::detail::CalculateMemoryUtilization(estimates,
                                                                    ddh100_),
      0.0);

  estimates.exec_time = absl::Seconds(4);
  ddh100_.set_memory_bandwidth(0);
  // We default to 0.0 memory utilization if peak memory bandwidth is zero.
  EXPECT_DOUBLE_EQ(
      gpu_dot_fusion_cost_model::detail::CalculateMemoryUtilization(estimates,
                                                                    ddh100_),
      0.0);
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthMonotonicallyIncreasesWithTransferSize) {
  constexpr int64_t k8GiB = 8LL * (1LL << 30);
  for (const se::DeviceDescription* dev : {&dda100_, &ddh100_, &ddb200_}) {
    float prev_bw = 0.0f;
    for (int64_t dma_size = 8192; dma_size <= k8GiB; dma_size *= 2) {
      float bw = GetEffectiveHbmBandwidth(dma_size, *dev);
      EXPECT_GT(bw, prev_bw);
      prev_bw = bw;
    }
  }
}

// Verifies that normalized fractional bandwidth scales Ampere > Hopper >
// Blackwell. Narrower memory buses with fewer pseudo-channels require less
// in-flight concurrency to saturate memory pipelines, achieving higher
// fractions of peak bandwidth at smaller transfer sizes.
TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthFractionOrderingAcrossArchitectures) {
  constexpr int64_t kTransferSizes[] = {
      16 * 1024,         // 16 KiB
      64 * 1024,         // 64 KiB
      2 * 1024 * 1024,   // 2 MiB
      8 * 1024 * 1024,   // 8 MiB
      32 * 1024 * 1024,  // 32 MiB
      128 * 1024 * 1024  // 128 MiB
  };
  for (int64_t dma_size : kTransferSizes) {
    float frac_a100 = GetEffectiveHbmBandwidth(dma_size, dda100_) /
                      dda100_.memory_bandwidth();
    float frac_h100 = GetEffectiveHbmBandwidth(dma_size, ddh100_) /
                      ddh100_.memory_bandwidth();
    float frac_b200 = GetEffectiveHbmBandwidth(dma_size, ddb200_) /
                      ddb200_.memory_bandwidth();

    EXPECT_GT(frac_a100, frac_h100);
    EXPECT_GT(frac_h100, frac_b200);
  }
}

// Verifies that minimum transfer sizes are latency-bound and achieve a small
// fraction (< 10%) of peak bandwidth across architectures.
TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthMinTransferSizeIsLatencyBound) {
  for (const se::DeviceDescription* dev : {&dda100_, &ddh100_, &ddb200_}) {
    float first_frac =
        GetEffectiveHbmBandwidth(8192, *dev) / dev->memory_bandwidth();
    EXPECT_GT(first_frac, 0.0f);
    EXPECT_LT(first_frac, 0.10f);
  }
}

// Verifies that large asymptotic transfers approach full hardware saturation (>
// 90% peak bandwidth) across architectures.
TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthAsymptoticTransferApproachesHardwareSaturation) {
  constexpr int64_t k8GiB = 8LL * (1LL << 30);
  for (const se::DeviceDescription* dev : {&dda100_, &ddh100_, &ddb200_}) {
    float last_frac =
        GetEffectiveHbmBandwidth(k8GiB, *dev) / dev->memory_bandwidth();
    EXPECT_GT(last_frac, 0.90f);
    EXPECT_LE(last_frac, 1.0f);
  }
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthClampsBelowMinimumTransferSize) {
  // DMA sizes below minimum table entry (8 KiB) clamp to the minimum entry.
  EXPECT_FLOAT_EQ(GetEffectiveHbmBandwidth(4096, ddh100_),
                  GetEffectiveHbmBandwidth(8192, ddh100_));
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthClampsAboveMaximumTransferSize) {
  constexpr int64_t k8GiB = 8LL * (1LL << 30);
  constexpr int64_t k16GiB = 16LL * (1LL << 30);
  // DMA sizes above maximum table entry (8 GiB) clamp to the maximum entry.
  EXPECT_FLOAT_EQ(GetEffectiveHbmBandwidth(k16GiB, ddh100_),
                  GetEffectiveHbmBandwidth(k8GiB, ddh100_));
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthLinearlyInterpolatesBetweenTableEntries) {
  float bw_8k = GetEffectiveHbmBandwidth(8192, ddh100_);
  float bw_16k = GetEffectiveHbmBandwidth(16384, ddh100_);
  EXPECT_FLOAT_EQ(GetEffectiveHbmBandwidth(12288, ddh100_),
                  (bw_8k + bw_16k) / 2.0f);
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthFallsBackToAmpereForOlderArchitectures) {
  constexpr int64_t k128MiB = 128LL * (1LL << 20);
  se::DeviceDescription dd_volta = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::GpuComputeCapability{se::CudaComputeCapability(7, 0)});
  dd_volta.set_memory_bandwidth(900ULL * 1000 * 1000 * 1000);  // 900 GB/s
  EXPECT_FLOAT_EQ(
      GetEffectiveHbmBandwidth(k128MiB, dd_volta) / dd_volta.memory_bandwidth(),
      GetEffectiveHbmBandwidth(k128MiB, dda100_) / dda100_.memory_bandwidth());
}

TEST_F(GpuDotFusionCostModelTest,
       EffectiveHbmBandwidthFallsBackToBlackwellForFutureArchitectures) {
  constexpr int64_t k128MiB = 128LL * (1LL << 20);
  se::DeviceDescription dd_future = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::GpuComputeCapability{se::CudaComputeCapability(11, 0)});
  dd_future.set_memory_bandwidth(10000ULL * 1000 * 1000 * 1000);  // 10 TB/s
  EXPECT_FLOAT_EQ(
      GetEffectiveHbmBandwidth(k128MiB, dd_future) /
          dd_future.memory_bandwidth(),
      GetEffectiveHbmBandwidth(k128MiB, ddb200_) / ddb200_.memory_bandwidth());
}

TEST_F(GpuDotFusionCostModelTest, GpuDotComputeBoundA100Bf16) {
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
  block_params.output_tile_sizes = {{256, 512}};
  block_params.num_warps = 4;
  block_params.num_ctas = 1;
  block_params.num_stages = 3;
  auto* dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());
  ASSERT_OK(gpu_dot_fusion_cost_model::IsSupported(dot));
  ASSERT_OK_AND_ASSIGN(
      EstimateRunTimeData runtime_a100,
      gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, dda100_));
  ASSERT_OK_AND_ASSIGN(
      ComputeAndFlops expected_compute_and_flops_a100,
      CalculateComputeTimeWithTileAndWaveQuantization(
          DotProblemInfo(*dot),
          DotTileSize{/*m=*/block_params.output_tile_sizes[0][0],
                      /*n=*/block_params.output_tile_sizes[0][1]},
          dda100_));
  absl::Duration expected_time =
      expected_compute_and_flops_a100.compute_time + kLoopLatencyTax;
  EXPECT_GE(runtime_a100.exec_time, expected_time);
  EXPECT_LE(runtime_a100.exec_time, expected_time * 1.1);
}

TEST_F(GpuDotFusionCostModelTest, AmpereTileMDerate) {
  double flops_small = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/31, dda100_, PrimitiveType::F16);
  double flops_full = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/32, dda100_, PrimitiveType::F16);

  ASSERT_GT(flops_full, 0);
  // tile_m < 32: 50% derate.
  EXPECT_NEAR(flops_small / flops_full, 0.50, 1e-4);
}

TEST_F(GpuDotFusionCostModelTest, HopperTileMDerate) {
  double flops_small = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/63, ddh100_, PrimitiveType::F16);
  double flops_full = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/64, ddh100_, PrimitiveType::F16);

  ASSERT_GT(flops_full, 0);
  // tile_m < 64: 63% derate.
  EXPECT_NEAR(flops_small / flops_full, 0.63, 1e-4);
}

TEST_F(GpuDotFusionCostModelTest, BlackwellTileMDerate) {
  double flops_small = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/127, ddb200_, PrimitiveType::BF16);
  double flops_full = GetEffectiveFlopsPerNsForTileSize(
      /*tile_m=*/128, ddb200_, PrimitiveType::BF16);

  ASSERT_GT(flops_full, 0);
  // tile_m < 128: 50% derate.
  EXPECT_NEAR(flops_small / flops_full, 0.50, 1e-4);
}

DotProblemInfo CreateDotInfo(PrimitiveType output_type) {
  DotProblemInfo dot_info;
  dot_info.lhs_element_type = output_type;
  dot_info.rhs_element_type = output_type;
  dot_info.output_element_type = output_type;
  return dot_info;
}

BlockLevelParameters CreateBlockParams(int64_t num_warps) {
  BlockLevelParameters params;
  params.num_warps = num_warps;
  return params;
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateRegistersPerThreadIncreasesWithTileSize) {
  const DotProblemInfo dot_info = CreateDotInfo(PrimitiveType::F32);
  const BlockLevelParameters block_params = CreateBlockParams(/*num_warps=*/4);

  const int regs_small_tile = CalculateRegistersPerThread(
      dot_info, DotTileSize{/*m=*/64, /*n=*/64, /*k=*/32, /*b=*/1},
      block_params, ddh100_);
  const int regs_large_tile = CalculateRegistersPerThread(
      dot_info, DotTileSize{/*m=*/128, /*n=*/128, /*k=*/32, /*b=*/1},
      block_params, ddh100_);

  EXPECT_GT(regs_large_tile, regs_small_tile);
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateRegistersPerThreadDecreasesWithMoreWarps) {
  const DotProblemInfo dot_info = CreateDotInfo(PrimitiveType::F32);
  const DotTileSize tile{/*m=*/128, /*n=*/128, /*k=*/32, /*b=*/1};

  const int regs_4_warps = CalculateRegistersPerThread(
      dot_info, tile, CreateBlockParams(/*num_warps=*/4), ddh100_);
  const int regs_8_warps = CalculateRegistersPerThread(
      dot_info, tile, CreateBlockParams(/*num_warps=*/8), ddh100_);

  EXPECT_LT(regs_8_warps, regs_4_warps);
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateRegistersPerThreadIncreasesWithOutputBitWidth) {
  const DotTileSize tile{/*m=*/128, /*n=*/128, /*k=*/32, /*b=*/1};
  const BlockLevelParameters block_params = CreateBlockParams(/*num_warps=*/4);

  const int regs_f16 = CalculateRegistersPerThread(
      CreateDotInfo(PrimitiveType::F16), tile, block_params, ddh100_);
  const int regs_f32 = CalculateRegistersPerThread(
      CreateDotInfo(PrimitiveType::F32), tile, block_params, ddh100_);
  const int regs_f64 = CalculateRegistersPerThread(
      CreateDotInfo(PrimitiveType::F64), tile, block_params, ddh100_);

  EXPECT_GT(regs_f32, regs_f16);
  EXPECT_GT(regs_f64, regs_f32);
}

TEST_F(GpuDotFusionCostModelTest,
       CalculateRegistersPerThreadWithinRealisticBounds) {
  const DotProblemInfo dot_info = CreateDotInfo(PrimitiveType::F32);
  const BlockLevelParameters block_params = CreateBlockParams(/*num_warps=*/4);
  const DotTileSize tile{/*m=*/128, /*n=*/128, /*k=*/64, /*b=*/1};

  const int regs =
      CalculateRegistersPerThread(dot_info, tile, block_params, ddh100_);

  // Estimates must be strictly above base overhead (24) and within GPU hardware
  // max (255).
  EXPECT_GT(regs, 24);
  EXPECT_LE(regs, 255);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
