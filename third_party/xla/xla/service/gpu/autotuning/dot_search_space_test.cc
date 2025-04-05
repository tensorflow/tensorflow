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

#include "xla/service/gpu/autotuning/dot_search_space.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Eq;
using ::testing::Field;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::SizeIs;

#define DEFINE_FIELD_MATCHER(matcher_name, field_name)                 \
  auto matcher_name(auto matcher) {                                    \
    return Field(#field_name, &TritonGemmConfig::field_name, matcher); \
  }
// We technically don't need the semicolons below, but it throws off the VSCode
// formatter if we have what looks like a statement without a semicolon, so
// adding it makes editing the file a lot easier.
DEFINE_FIELD_MATCHER(BlockMIs, block_m);
DEFINE_FIELD_MATCHER(BlockNIs, block_n);
DEFINE_FIELD_MATCHER(BlockKIs, block_k);
DEFINE_FIELD_MATCHER(SplitKIs, split_k);
DEFINE_FIELD_MATCHER(NumStagesIs, num_stages);
DEFINE_FIELD_MATCHER(NumWarpsIs, num_warps);
DEFINE_FIELD_MATCHER(NumCtasIs, num_ctas);
#undef DEFINE_FIELD_MATCHER

auto IsValidConfig() {
  return AllOf(BlockMIs(Ge(1)), BlockNIs(Ge(1)), BlockKIs(Ge(1)),
               SplitKIs(Ge(1)), NumStagesIs(Ge(1)), NumWarpsIs(Ge(1)),
               NumCtasIs(Ge(1)));
};

class DotSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_;

  DotSearchSpaceTest()
      : device_description_(se::GpuDeviceInfoProto::default_instance()) {
    // Using H100 numbers as the most relevant example here.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
    // https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth
    device_description_.set_registers_per_block_limit(64 * 1024);
    device_description_.set_core_count(132);
    device_description_.set_threads_per_block_limit(1024);
    device_description_.set_threads_per_warp(32);
    device_description_.set_shared_memory_per_block_optin(227 * 1024);
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> GetDefaultDotModule(
      int lhs_parallel_dim = 1024, int rhs_parallel_dim = 1024,
      int contracting_dim = 1024) {
    constexpr const char* kModuleTextFormat = R"(
ENTRY e {
  p0 = f16[%d,%d] parameter(0)
  p1 = f16[%d,%d] parameter(1)
  ROOT r = f16[%d,%d] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
    return ParseAndReturnVerifiedModule(absl::StrFormat(
        kModuleTextFormat, lhs_parallel_dim, contracting_dim, contracting_dim,
        rhs_parallel_dim, lhs_parallel_dim, rhs_parallel_dim));
  }

  HloDotInstruction* GetDot(VerifiedHloModule* module) {
    return Cast<HloDotInstruction>(
        module->entry_computation()->root_instruction());
  }

  TritonDotFusionSearchSpace MakeSearchSpace(VerifiedHloModule* module) {
    return TritonDotFusionSearchSpace(device_description_, GetDot(module));
  }
};

TEST_F(DotSearchSpaceTest, SerializesSearchSpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_EQ(search_space.Serialize(),
            "problem_size_BxMxNxKxE: 1x1024x1024x1024x(16->16) "
            "tile_range_SxMxNxK: [1-128]x[16-256]x[16-512]x[8-?] "
            "desired_total_warps: 2640 warps_per_block: [4-?]");
}

TEST_F(DotSearchSpaceTest, ReturnsValidConfigList) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(IsValidConfig())));
}

TEST_F(DotSearchSpaceTest, HonorsForcedContractingSplit) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(/*force_contracting_split=*/2),
      AllOf(Not(IsEmpty()), Each(IsValidConfig()), Each(SplitKIs(Eq(2)))));
}

TEST_F(DotSearchSpaceTest, ConsidersContractingSplitForSmallOutputSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/1024));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(SplitKIs(Ge(2))));
}

TEST_F(DotSearchSpaceTest, LimitsContractingSplitForSmallerContractingSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/32));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(SplitKIs(Le(4)))));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseOutputTiles) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Ge(32)), BlockNIs(Ge(32)))).Times(Ge(2)));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseTilesForLowOccupancyProblem) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/4096, /*rhs_parallel_dim=*/16,
                          /*contracting_dim=*/4096));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Ge(32)), SplitKIs(Ge(2)))));
}

TEST_F(DotSearchSpaceTest,
       FindsUniqueOccupancyMaximizingTilingForSmallProblem) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/32, /*rhs_parallel_dim=*/32,
                          /*contracting_dim=*/32));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(SizeIs(1), Each(AllOf(BlockMIs(Eq(16)), BlockNIs(Eq(16)),
                                          SplitKIs(Eq(4))))));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseTilesForForcedHugeSplit) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(/*force_contracting_split=*/128),
      Contains(AllOf(BlockMIs(Ge(32)), BlockNIs(Ge(32)), SplitKIs(Eq(128)))));
}

TEST_F(DotSearchSpaceTest, PadsTilesForSmallParallelDimension) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024,
                                              /*rhs_parallel_dim=*/15,
                                              /*contracting_dim=*/1024));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(BlockNIs(Eq(16))));
}

TEST_F(DotSearchSpaceTest, HonorsMinimumOutputTileSizeForTinyProblem) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/12,
                                              /*rhs_parallel_dim=*/8,
                                              /*contracting_dim=*/16));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(),
      AllOf(Not(IsEmpty()), Each(BlockMIs(Ge(16))), Each(BlockNIs(Ge(16)))));
}

TEST_F(DotSearchSpaceTest, AssignsEnoughWarpsPerScheduler) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/512,
                          /*contracting_dim=*/1024));
  auto search_space = MakeSearchSpace(module.get());

  // 1024x512 elements / 32x32 elements/block = 32x16 blocks = 512 blocks.
  // 512 blocks * 4 warps/block = 2048 warps.
  // 132 cores * 4 schedulers/core * 5 desired warps/scheduler = 2640 desired
  // warps.
  // ceil(2640 desired warps / 2048 warps) = ceil(1.3) = 2 desired split
  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Eq(32)), BlockNIs(Eq(32)),
                             NumWarpsIs(Eq(4)), SplitKIs(Eq(2)))));
}

TEST_F(DotSearchSpaceTest, DoesNotBreakCTASizeLimits) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024 * 16,
                                              /*rhs_parallel_dim=*/1024 * 16));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(NumWarpsIs(Le(32)))));
}

TEST_F(DotSearchSpaceTest, ConsidersAppropriateCTASizeForTileSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/4096,
                                              /*rhs_parallel_dim=*/4096));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Contains(AllOf(BlockMIs(Eq(64)), BlockNIs(Eq(32)),
                                   NumWarpsIs(Eq(4)))),
                    Contains(AllOf(BlockMIs(Eq(128)), BlockNIs(Eq(32)),
                                   NumWarpsIs(Eq(8))))));
}

TEST_F(DotSearchSpaceTest, FindsFullCacheLineContractingTileSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(BlockKIs(Ge(64))));
}

TEST_F(DotSearchSpaceTest, HonorsSharedMemoryLimit) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/4096, /*rhs_parallel_dim=*/4096,
                          /*contracting_dim=*/4096));
  auto search_space = MakeSearchSpace(module.get());

  auto good_output_tile =
      AllOf(BlockMIs(Eq(128)), BlockNIs(Eq(128)), SplitKIs(Eq(1)));
  // 2B * (128 + 128) * block_k < 227 KB =>
  // block_k <= 227 KB / (2B * (128 + 128)) = 454
  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Contains(good_output_tile),
                    Not(Contains(AllOf(good_output_tile, BlockKIs(Ge(512)))))));
}

TEST_F(DotSearchSpaceTest, HonorsContractingSizeLimit) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/256));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(/*force_contracting_split=*/4),
              AllOf(Not(IsEmpty()), Each(BlockKIs(Le(64)))));
}

TEST_F(DotSearchSpaceTest, EnsuresContractingTileSizeFitsInstructonShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/4));
  auto search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(BlockKIs(Ge(8)))));
}

}  // namespace
}  // namespace xla::gpu
