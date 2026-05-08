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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
namespace detail = cost_model_config_optimization_detail;

TEST(CostModelConfigOptimizationTest,
     ParseCostModelGemmTilingOptionsParsesOptions) {
  google::protobuf::Map<std::string, std::string> options;
  options["top"] = "3";
  options["top_from_default"] = "1";
  options["mixin"] = "2";
  options["filter"] = "0.5";

  TF_ASSERT_OK_AND_ASSIGN(detail::CostModelGemmTilingOptions parsed,
                          detail::ParseCostModelGemmTilingOptions(options));

  EXPECT_EQ(parsed.top, 3);
  EXPECT_EQ(parsed.top_from_default, true);
  EXPECT_EQ(parsed.mixin, 2);
  EXPECT_EQ(parsed.filter, 0.5f);
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

TEST_F(HloHardwareIndependentTestBase,
       OptimizeConfigsWithCostModelSelectsTopConfigs) {
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

  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  const se::DeviceDescription h100_device_info =
      TestGpuDeviceInfo::H100SXMDeviceInfo();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *root->fused_instructions_computation(), HloOpcode::kDot));

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
                              debug_options, &mlir_context));

  EXPECT_THAT(optimized,
              ElementsAre(TritonGemmConfig(128, 128, 128, 1, 4, 1, false),
                          TritonGemmConfig(64, 64, 64, 1, 4, 1, false)));
}

}  // namespace
}  // namespace xla::gpu
