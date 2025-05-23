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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::SizeIs;

// Returns a matcher that verifies that each container element that matches
// `filter` also matches `matcher`, and that there is at least one such element.
template <typename FilterMatcher, typename Matcher>
auto WhenFilteredBy(FilterMatcher filter, Matcher matcher) {
  // We check the negation: there is no element that matches `filter` and does
  // not match `matcher`.
  return AllOf(Contains(filter), Not(Contains(AllOf(filter, Not(matcher)))));
}

template <typename MatcherType>
auto BlockMIs(MatcherType matcher) {
  return Field("block_m", &TritonGemmConfig::block_m, matcher);
}
template <typename MatcherType>
auto BlockNIs(MatcherType matcher) {
  return Field("block_n", &TritonGemmConfig::block_n, matcher);
}
template <typename MatcherType>
auto BlockKIs(MatcherType matcher) {
  return Field("block_k", &TritonGemmConfig::block_k, matcher);
}
template <typename MatcherType>
auto SplitKIs(MatcherType matcher) {
  return Field("split_k", &TritonGemmConfig::split_k, matcher);
}
template <typename MatcherType>
auto NumStagesIs(MatcherType matcher) {
  return Field("num_stages", &TritonGemmConfig::num_stages, matcher);
}
template <typename MatcherType>
auto NumWarpsIs(MatcherType matcher) {
  return Field("num_warps", &TritonGemmConfig::num_warps, matcher);
}
template <typename MatcherType>
auto NumCtasIs(MatcherType matcher) {
  return Field("num_ctas", &TritonGemmConfig::num_ctas, matcher);
}

auto IsValidConfig() {
  return AllOf(BlockMIs(Ge(1)), BlockNIs(Ge(1)), BlockKIs(Ge(1)),
               SplitKIs(Ge(1)), NumStagesIs(Ge(1)), NumWarpsIs(Ge(1)),
               NumCtasIs(Ge(1)));
};

class DefaultDeviceDotSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_{
      se::GpuDeviceInfoProto::default_instance()};

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

TEST_F(DefaultDeviceDotSearchSpaceTest, ReturnsValidConfigList) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Not(IsEmpty()));
}

class DotSearchSpaceTest : public DefaultDeviceDotSearchSpaceTest {
 protected:
  DotSearchSpaceTest() {
    // Using H100 numbers as the most relevant example here.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
    // https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth
    device_description_.set_registers_per_block_limit(64 * 1024);
    device_description_.set_core_count(132);
    device_description_.set_threads_per_block_limit(1024);
    device_description_.set_threads_per_warp(32);
    device_description_.set_shared_memory_per_block_optin(227 * 1024);
    device_description_.set_gpu_compute_capability(
        se::CudaComputeCapability::Hopper());
  }
};

TEST_F(DotSearchSpaceTest, SerializesSearchSpace) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_EQ(search_space.ToString(),
            "problem_size_BxMxNxKxE: 1x1024x1024x1024x(16->16) "
            "tile_range_SxMxNxK: [1-64]x[16-256]x[16-512]x[16-?] "
            "desired_total_warps: 2640 occupancy_optimization: 1 "
            "warps_per_cta: [2-?]");
}

TEST_F(DotSearchSpaceTest, ReturnsValidConfigList) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(IsValidConfig())));
}

TEST_F(DotSearchSpaceTest, HonorsForcedContractingSplit) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(/*force_contracting_split=*/2),
      AllOf(Not(IsEmpty()), Each(IsValidConfig()), Each(SplitKIs(Eq(2)))));
}

TEST_F(DotSearchSpaceTest, ConsidersContractingSplitForSmallOutputSize) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(SplitKIs(Ge(2))));
}

TEST_F(DotSearchSpaceTest, LimitsContractingSplitForSmallerContractingSize) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/32));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(SplitKIs(Le(4)))));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseOutputTiles) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024,
                                              /*rhs_parallel_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Ge(32)), BlockNIs(Ge(32)))).Times(Ge(2)));
}

TEST_F(DotSearchSpaceTest, RestrictsOutputToSquareishTiles) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024,
                                              /*rhs_parallel_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(),
      WhenFilteredBy(BlockMIs(Eq(64)), BlockNIs(AllOf(Ge(32), Le(128)))));
}

TEST_F(DotSearchSpaceTest, AllowsLargerRhsForExpensiveLhs) {
  constexpr const char* kModuleText = R"(
ENTRY e {
  p0 = f16[4096,4096] parameter(0)
  e0 = f16[4096,4096] exponential(p0)
  p1 = f16[4096,4096] parameter(1)
  ROOT r = f16[4096,4096] dot(e0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kModuleText));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Eq(32)), BlockNIs(Ge(128)))));
}

TEST_F(DotSearchSpaceTest, AllowsLargerLhsForExpensiveRhs) {
  constexpr const char* kModuleText = R"(
ENTRY e {
  p0 = f16[4096,4096] parameter(0)
  p1 = f16[4096,4096] parameter(1)
  e1 = f16[4096,4096] exponential(p1)
  ROOT r = f16[4096,4096] dot(p0, e1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kModuleText));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Ge(128)), BlockNIs(Eq(32)))));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseTilesForLowOccupancyProblem) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/4096, /*rhs_parallel_dim=*/16,
                          /*contracting_dim=*/4096));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Ge(32)), SplitKIs(Ge(2)))));
}

TEST_F(DotSearchSpaceTest,
       FindsUniqueOccupancyMaximizingTilingForSmallProblem) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/32, /*rhs_parallel_dim=*/32,
                          /*contracting_dim=*/32));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(SizeIs(1), Each(AllOf(BlockMIs(Eq(16)), BlockNIs(Eq(16)),
                                          SplitKIs(Eq(2))))));
}

TEST_F(DotSearchSpaceTest, FindsGoodDataReuseTilesForForcedHugeSplit) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024,
                                              /*rhs_parallel_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(/*force_contracting_split=*/128),
      Contains(AllOf(BlockMIs(Ge(32)), BlockNIs(Ge(32)), SplitKIs(Eq(128)))));
}

TEST_F(DotSearchSpaceTest, PadsTilesForSmallParallelDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024,
                                              /*rhs_parallel_dim=*/15,
                                              /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(BlockNIs(Eq(16))));
}

TEST_F(DotSearchSpaceTest, HonorsMinimumOutputTileSizeForTinyProblem) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/12,
                                              /*rhs_parallel_dim=*/8,
                                              /*contracting_dim=*/16));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(),
      AllOf(Not(IsEmpty()), Each(BlockMIs(Ge(16))), Each(BlockNIs(Ge(16)))));
}

TEST_F(DotSearchSpaceTest, AssignsEnoughWarpsPerScheduler) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/512,
                          /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  // 1024x512 elements / 32x32 elements/CTA = 32x16 blocks = 512 CTAs.
  // 512 CTAs * 4 warps/CTA = 2048 warps.
  // 132 cores * 4 schedulers/core * 5 desired warps/scheduler = 2640 desired
  // warps.
  // ceil(2640 desired warps / 2048 warps) = ceil(1.3) = 2 desired split
  EXPECT_THAT(search_space.GenerateConfigs(),
              Contains(AllOf(BlockMIs(Eq(32)), BlockNIs(Eq(32)),
                             NumWarpsIs(Eq(4)), SplitKIs(Eq(2)))));
}

TEST_F(DotSearchSpaceTest, DoesNotBreakCtaSizeLimits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/1024 * 16,
                                              /*rhs_parallel_dim=*/1024 * 16));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(NumWarpsIs(Le(32)))));
}

TEST_F(DotSearchSpaceTest, ConsidersAppropriateCtaSizeForTileSize) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/4096,
                                              /*rhs_parallel_dim=*/4096));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Contains(AllOf(BlockMIs(Eq(64)), BlockNIs(Eq(32)),
                                   NumWarpsIs(Eq(4)))),
                    Contains(AllOf(BlockMIs(Eq(64)), BlockNIs(Eq(64)),
                                   NumWarpsIs(Eq(8))))));
}

TEST_F(DotSearchSpaceTest, FindsFullCacheLineContractingTileSize) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(BlockKIs(Ge(64))));
}

TEST_F(DotSearchSpaceTest, HonorsSharedMemoryLimit) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/4096, /*rhs_parallel_dim=*/4096,
                          /*contracting_dim=*/4096));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  // We pick the 128x128 output tiling and only verify that configs with these
  // properties honor the memory limit. This simplifies the test logic and makes
  // the calculation easier to verify by hand, while not reducing the coverage
  // of the test.
  // 2B * (128 + 128) * block_k < 227 KB =>
  // block_k <= 227 KB / (2B * (128 + 128)) = 454
  EXPECT_THAT(search_space.GenerateConfigs(/*force_contracting_split=*/1),
              WhenFilteredBy(AllOf(BlockMIs(Eq(128)), BlockNIs(Eq(128))),
                             BlockKIs(Le(256))));
}

TEST_F(DotSearchSpaceTest, HonorsContractingSizeLimit) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/256));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(/*force_contracting_split=*/4),
              AllOf(Not(IsEmpty()), Each(BlockKIs(Le(64)))));
}

TEST_F(DotSearchSpaceTest, EnsuresContractingTileSizeFitsInstructonShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/4));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(BlockKIs(Ge(8)))));
}

TEST_F(DotSearchSpaceTest, FindReasonablePipeliningStageCount) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Contains(NumStagesIs(Ge(2))).Times(Ge(2)),
                    Contains(NumStagesIs(Eq(1))), Each(NumStagesIs(Le(5)))));
}

TEST_F(DotSearchSpaceTest, LimitsStagesToAvailableTileSize) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/1024, /*rhs_parallel_dim=*/1024,
                          /*contracting_dim=*/128));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  // We pick the 64x32x32 tiling and only verify that configs with these
  // properties choose the right number of stages. This simplifies the test
  // logic and makes the calculation easier to verify by hand, while not
  // reducing the coverage of the test.
  EXPECT_THAT(search_space.GenerateConfigs(/*force_contracting_split=*/2),
              WhenFilteredBy(
                  AllOf(BlockMIs(Eq(64)), BlockNIs(Eq(32)), BlockKIs(Eq(32))),
                  NumStagesIs(Le(2))));
}

TEST_F(DotSearchSpaceTest, ConsidersFewWarpsPerCtaAndMmaForSmallProblem) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/128, /*rhs_parallel_dim=*/128,
                          /*contracting_dim=*/128));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(),
      Contains(AllOf(NumWarpsIs(Eq(2)), BlockMIs(Eq(16)), BlockNIs(Eq(16)))));
}

TEST_F(DotSearchSpaceTest, EnsuresWgmmaShapeForLargeProblem) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16 * 1024,
                                              /*rhs_parallel_dim=*/16 * 1024,
                                              /*contracting_dim=*/4096));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  EXPECT_THAT(
      search_space.GenerateConfigs(),
      AllOf(Not(IsEmpty()), Each(AllOf(NumWarpsIs(Ge(4)), BlockMIs(Ge(64)),
                                       BlockNIs(Ge(16))))));
}

TEST_F(DotSearchSpaceTest, ReturnsAllConfigsIfNoHints) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());
  std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs();

  EXPECT_THAT(search_space.OptimizeConfigSet(configs, {}),
              ElementsAreArray(configs));
}

TEST_F(DotSearchSpaceTest, OptimizesEmptyConfigSet) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());
  TritonGemmConfig hint = {/*block_m=*/32,   /*block_n=*/32,
                           /*block_k=*/32,   /*split_k=*/1,
                           /*num_stages=*/1, /*num_warps=*/4,
                           /*num_ctas=*/1};

  EXPECT_THAT(search_space.OptimizeConfigSet({}, {hint}), IsEmpty());
}

TEST_F(DotSearchSpaceTest, RestrictsConfigsToHints) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());
  TritonGemmConfig matching_hint = {
      /*block_m=*/32, /*block_n=*/32,   /*block_k=*/32,
      /*split_k=*/1,  /*num_stages=*/1, /*num_warps=*/4,
      /* num_ctas=*/1};
  TritonGemmConfig non_matching_hint = {
      /*block_m=*/64, /*block_n=*/32,   /*block_k=*/32,
      /*split_k=*/1,  /*num_stages=*/1, /*num_warps=*/4,
      /*num_ctas=*/1};
  TritonGemmConfig other_config = {
      /*block_m=*/32, /*block_n=*/64,   /*block_k=*/32,
      /*split_k=*/1,  /*num_stages=*/1, /*num_warps=*/4,
      /*num_ctas=*/1};

  EXPECT_THAT(
      search_space.OptimizeConfigSet({other_config, matching_hint},
                                     {matching_hint, non_matching_hint}),
      ElementsAre(matching_hint));
}

TEST_F(DotSearchSpaceTest, RestrictsConfigsWithPartialMatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      GetDefaultDotModule(/*lhs_parallel_dim=*/4096, /*rhs_parallel_dim=*/16,
                          /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());
  TritonGemmConfig hint = {/*block_m=*/32,   /*block_n=*/32,
                           /*block_k=*/32,   /*split_k=*/1,
                           /*num_stages=*/1, /*num_warps=*/4,
                           /*num_ctas=*/1};
  TritonGemmConfig expected = {/*block_m=*/32,   /*block_n=*/16,
                               /*block_k=*/32,   /*split_k=*/2,
                               /*num_stages=*/1, /*num_warps=*/4,
                               /*num_ctas=*/1};

  EXPECT_THAT(
      search_space.OptimizeConfigSet(
          search_space.GenerateConfigs(/*force_contracting_split=*/2), {hint}),
      ElementsAre(expected));
}

TEST_F(DotSearchSpaceTest, ReturnsNonEmptySetForUnusualHints) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/4096,
                                              /*rhs_parallel_dim=*/4096));
  TritonDotFusionSearchSpace search_space = MakeSearchSpace(module.get());

  TritonGemmConfig hint = {/*block_m=*/1024, /*block_n=*/1024,
                           /*block_k=*/32,   /*split_k=*/1,
                           /*num_stages=*/1, /*num_warps=*/4,
                           /*num_ctas=*/1};

  EXPECT_THAT(
      search_space.OptimizeConfigSet(search_space.GenerateConfigs(), {hint}),
      Not(IsEmpty()));
}

}  // namespace
}  // namespace xla::gpu
