/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/triton/cost_model_config_optimization.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "google/protobuf/map.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;
namespace detail = cost_model_config_optimization_detail;

class CostModelConfigOptimizationHloTest
    : public HloHardwareIndependentTestBase {
 protected:
  struct ParsedModuleAndDot {
    std::unique_ptr<HloModule> module;
    const HloDotInstruction* dot = nullptr;
  };

  CostModelConfigOptimizationHloTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  ParsedModuleAndDot GetHloModuleAndDot_F32_1024_1024_1024() {
    const char kHlo[] = R"(
      HloModule module

      computation {
        p0 = f32[1024,1024]{1,0} parameter(0)
        p1 = f32[1024,1024]{1,0} parameter(1)
        ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}
      }

      ENTRY main {
        p0 = f32[1024,1024]{1,0} parameter(0)
        p1 = f32[1024,1024]{1,0} parameter(1)
        ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
          kind=kCustom, calls=computation,
          backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
      })";
    auto module_or = ParseAndReturnVerifiedModule(kHlo);
    CHECK_OK(module_or.status());
    auto module = std::move(module_or).value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    const HloDotInstruction* dot =
        Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
            *root->fused_instructions_computation(), HloOpcode::kDot));
    return {std::move(module), dot};
  }

  mlir::MLIRContext mlir_context_;
};

TEST(CostModelConfigOptimizationTest,
     ParseCostModelGemmTilingOptionsParsesOptions) {
  google::protobuf::Map<std::string, std::string> options;
  options["top"] = "3";
  options["top_from_default"] = "1";
  options["mixin"] = "2";
  options["filter"] = "0.5";
  options["mixin_max_same_mnk"] = "1";
  options["mixin_only_faster"] = "1";

  TF_ASSERT_OK_AND_ASSIGN(detail::CostModelGemmTilingOptions parsed,
                          detail::ParseCostModelGemmTilingOptions(options));

  EXPECT_EQ(parsed.top, 3);
  EXPECT_EQ(parsed.top_from_default, true);
  EXPECT_EQ(parsed.mixin, 2);
  EXPECT_EQ(parsed.filter, 0.5f);
  EXPECT_EQ(parsed.mixin_max_same_mnk, 1);
  EXPECT_EQ(parsed.mixin_only_faster, true);
}

TEST(CostModelConfigOptimizationTest,
     FilterConfigsByRatioVsFastestFiltersSlowerConfigs) {
  detail::OrderedEstimatesAndConfigs input;
  input.insert(
      {absl::Milliseconds(10), TritonGemmConfig(32, 32, 32, 1, 1, 1, false)});
  input.insert(
      {absl::Milliseconds(15), TritonGemmConfig(64, 64, 64, 1, 1, 1, false)});
  input.insert({absl::Milliseconds(25),
                TritonGemmConfig(128, 128, 128, 1, 1, 1, false)});

  detail::OrderedEstimatesAndConfigs filtered =
      detail::FilterConfigsByRatioVsFastest(input,
                                            0.6f);  // Threshold > 10*1.6 = 16ms

  EXPECT_THAT(
      filtered,
      ElementsAre(std::pair{absl::Milliseconds(10),
                            TritonGemmConfig(32, 32, 32, 1, 1, 1, false)},
                  std::pair{absl::Milliseconds(15),
                            TritonGemmConfig(64, 64, 64, 1, 1, 1, false)}));
}

TEST(CostModelConfigOptimizationTest,
     FilterConfigsByRatioVsFastestWithZeroThresholdReturnsOnlyFastest) {
  const TritonGemmConfig config_32 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);
  const TritonGemmConfig config_128 =
      TritonGemmConfig(128, 128, 128, 1, 1, 1, false);

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10), config_32});
  input.insert({absl::Milliseconds(15), config_64});
  input.insert({absl::Milliseconds(25), config_128});

  detail::OrderedEstimatesAndConfigs filtered =
      detail::FilterConfigsByRatioVsFastest(input, 0.0f);

  EXPECT_THAT(filtered,
              ElementsAre(std::pair{absl::Milliseconds(10), config_32}));
}

TEST(CostModelConfigOptimizationTest,
     GetTopEstimatedConfigsReturnsTopNFastestWithoutExclusion) {
  const TritonGemmConfig config_32 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);
  const TritonGemmConfig config_128 =
      TritonGemmConfig(128, 128, 128, 1, 1, 1, false);

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10), config_32});
  input.insert({absl::Milliseconds(15), config_64});
  input.insert({absl::Milliseconds(25), config_128});

  detail::OrderedEstimatesAndConfigs top =
      detail::GetTopEstimatedConfigs(input, 2, nullptr);

  EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32},
                               std::pair{absl::Milliseconds(15), config_64}));
}

TEST(CostModelConfigOptimizationTest,
     GetTopEstimatedConfigsReturnsTopNFastestWithExclusion) {
  const TritonGemmConfig config_32 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);
  const TritonGemmConfig config_128 =
      TritonGemmConfig(128, 128, 128, 1, 1, 1, false);

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10), config_32});
  input.insert({absl::Milliseconds(15), config_64});
  input.insert({absl::Milliseconds(25), config_128});

  detail::OrderedEstimatesAndConfigs to_skip;
  to_skip.insert({absl::Milliseconds(15), config_64});

  detail::OrderedEstimatesAndConfigs top =
      detail::GetTopEstimatedConfigs(input, 2, &to_skip);

  EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32},
                               std::pair{absl::Milliseconds(25), config_128}));
}

TEST(CostModelConfigOptimizationTest,
     GetTopEstimatedConfigsRespectsMaxSameMnk) {
  const TritonGemmConfig config_32_1 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_32_2 =
      TritonGemmConfig(32, 32, 32, 2, 2, 1, false);
  const TritonGemmConfig config_32_3 =
      TritonGemmConfig(32, 32, 32, 3, 3, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10), config_32_1});
  input.insert({absl::Milliseconds(11), config_32_2});
  input.insert({absl::Milliseconds(12), config_32_3});
  input.insert({absl::Milliseconds(15), config_64});

  detail::OrderedEstimatesAndConfigs top =
      detail::GetTopEstimatedConfigs(input, 4, nullptr, /*max_same_mnk=*/2);

  EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32_1},
                               std::pair{absl::Milliseconds(11), config_32_2},
                               std::pair{absl::Milliseconds(15), config_64}));
}

TEST(CostModelConfigOptimizationTest,
     GetTopEstimatedConfigsRespectsOnlyFasterThanSkip) {
  const TritonGemmConfig config_32 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);
  const TritonGemmConfig config_128 =
      TritonGemmConfig(128, 128, 128, 1, 1, 1, false);

  // Base set (configs to skip) has config_64 (15ms).
  // Fastest in base set is 15ms.
  detail::OrderedEstimatesAndConfigs base_set;
  base_set.insert({absl::Milliseconds(15), config_64});

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10),
                config_32});  // Faster than 15ms -> should be kept
  input.insert(
      {absl::Milliseconds(15), config_64});  // In base set -> skipped anyway
  input.insert({absl::Milliseconds(25),
                config_128});  // Slower than 15ms -> should be skipped

  // We want top 2, but only faster than skip list.
  detail::OrderedEstimatesAndConfigs top =
      detail::GetTopEstimatedConfigs(input, 2, &base_set, std::nullopt, true);

  EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32}));
}

TEST(CostModelConfigOptimizationTest,
     GetTopEstimatedConfigsRespectsOnlyFasterThanSkipHandlesEmptySkipList) {
  const TritonGemmConfig config_32 =
      TritonGemmConfig(32, 32, 32, 1, 1, 1, false);
  const TritonGemmConfig config_64 =
      TritonGemmConfig(64, 64, 64, 1, 1, 1, false);

  detail::OrderedEstimatesAndConfigs input;
  input.insert({absl::Milliseconds(10), config_32});
  input.insert({absl::Milliseconds(15), config_64});

  // Case 1: configs_to_skip is nullptr
  {
    detail::OrderedEstimatesAndConfigs top =
        detail::GetTopEstimatedConfigs(input, 2, nullptr, std::nullopt,
                                       /*only_faster_than_skip=*/true);
    EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32},
                                 std::pair{absl::Milliseconds(15), config_64}));
  }

  // Case 2: configs_to_skip is empty
  {
    detail::OrderedEstimatesAndConfigs configs_to_skip;
    detail::OrderedEstimatesAndConfigs top =
        detail::GetTopEstimatedConfigs(input, 2, &configs_to_skip, std::nullopt,
                                       /*only_faster_than_skip=*/true);
    EXPECT_THAT(top, ElementsAre(std::pair{absl::Milliseconds(10), config_32},
                                 std::pair{absl::Milliseconds(15), config_64}));
  }
}

TEST_F(CostModelConfigOptimizationHloTest,
       OptimizeConfigsWithCostModelSelectsTopConfigs) {
  const se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();

  auto [module, dot] = GetHloModuleAndDot_F32_1024_1024_1024();

  std::vector<TritonGemmConfig> all_configs = {
      TritonGemmConfig(32, 32, 32, 1, 4, 1, false),
      TritonGemmConfig(64, 64, 64, 1, 4, 1, false),
      TritonGemmConfig(128, 128, 128, 1, 4, 1, false),
  };

  DebugOptions debug_options;
  (*debug_options
        .mutable_xla_gpu_experimental_cost_model_gemm_tiling_options())["top"] =
      "2";

  TF_ASSERT_OK_AND_ASSIGN(std::vector<TritonGemmConfig> optimized,
                          OptimizeConfigsWithCostModel(
                              dot, all_configs, all_configs, h100_device_info,
                              debug_options, &mlir_context_));

  EXPECT_THAT(optimized,
              ElementsAre(TritonGemmConfig(128, 128, 128, 1, 4, 1, false),
                          TritonGemmConfig(64, 64, 64, 1, 4, 1, false)));
}

TEST_F(CostModelConfigOptimizationHloTest, MixinKeepsInvalidConfigs) {
  const se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();

  auto [module, dot] = GetHloModuleAndDot_F32_1024_1024_1024();

  TritonGemmConfig valid_config(64, 64, 64, 1, 4, 1, false);
  TritonGemmConfig invalid_config(33, 33, 33, 1, 4, 1, false);

  std::vector<TritonGemmConfig> all_configs = {valid_config, invalid_config};
  std::vector<TritonGemmConfig> optimized_configs = {valid_config,
                                                     invalid_config};

  DebugOptions debug_options;
  (*debug_options.mutable_xla_gpu_experimental_cost_model_gemm_tiling_options())
      ["mixin"] = "1";

  TF_ASSERT_OK_AND_ASSIGN(std::vector<TritonGemmConfig> optimized,
                          OptimizeConfigsWithCostModel(
                              dot, all_configs, optimized_configs,
                              h100_device_info, debug_options, &mlir_context_));

  EXPECT_THAT(optimized, UnorderedElementsAre(valid_config, invalid_config));
}

TEST_F(CostModelConfigOptimizationHloTest, FilterRemovesInvalidConfigs) {
  const se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();

  auto [module, dot] = GetHloModuleAndDot_F32_1024_1024_1024();

  TritonGemmConfig valid_config(64, 64, 64, 1, 4, 1, false);
  TritonGemmConfig invalid_config(33, 33, 33, 1, 4, 1, false);

  std::vector<TritonGemmConfig> all_configs = {valid_config, invalid_config};
  std::vector<TritonGemmConfig> optimized_configs = {valid_config,
                                                     invalid_config};

  DebugOptions debug_options;
  (*debug_options.mutable_xla_gpu_experimental_cost_model_gemm_tiling_options())
      ["filter"] = "1.0";

  TF_ASSERT_OK_AND_ASSIGN(std::vector<TritonGemmConfig> optimized,
                          OptimizeConfigsWithCostModel(
                              dot, all_configs, optimized_configs,
                              h100_device_info, debug_options, &mlir_context_));

  EXPECT_THAT(optimized, UnorderedElementsAre(valid_config));
}

}  // namespace
}  // namespace xla::gpu
