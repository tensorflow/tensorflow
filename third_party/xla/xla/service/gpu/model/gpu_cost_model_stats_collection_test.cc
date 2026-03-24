/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"

#include <stdint.h>

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::mlir::MLIRContext;
using ::testing::Contains;
using ::testing::Truly;

class GpuCostModelStatsCollectionTest : public HloHardwareIndependentTestBase {
 public:
  GpuCostModelStatsCollection cost_model_stats_{
      TestGpuDeviceInfo::H100SXMDeviceInfo(),
      GpuHloCostAnalysis::Options{.count_multiple_input_accesses = true},
      &mlir_context_};

 protected:
  mlir::MLIRContext mlir_context_;
};

TEST_F(GpuCostModelStatsCollectionTest, FusionInEntryComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
    HloModule test_module

    log {
      p = f32[16384]{0} parameter(0)
      ROOT l = f32[16384]{0} log(p)
    }

    ENTRY main {
      %p0 = f32[16384] parameter(0)
      ROOT %res = f32[16384]{0} fusion(p0), kind=kInput, calls=log
    }
    )hlo"));

  EXPECT_THAT(cost_model_stats_.Run(module.get()), IsOkAndHolds(false));

  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          root->backend_config<GpuBackendConfig>());

  EXPECT_EQ(gpu_config.reification_cost_size(), 1);
  EXPECT_GT(gpu_config.reification_cost()[0].end_to_end_cycles(), 0);
}

TEST_F(GpuCostModelStatsCollectionTest, FusionInWhileComputation) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
    HloModule test_module

    cond {
      p = f32[16384]{0} parameter(0)
      ROOT %constant.2 = pred[] constant(true)
    }

    log {
      p = f32[16384]{0} parameter(0)
      ROOT l = f32[16384]{0} log(p)
    }

    loop {
      %p0 = f32[16384] parameter(0)
      ROOT %res = f32[16384]{0} fusion(p0), kind=kInput, calls=log
    }

    ENTRY main {
      %p0 = f32[16384] parameter(0)
      ROOT %while = f32[16384] while(%p0), body=%loop, condition=%cond
    })hlo"));

  EXPECT_THAT(cost_model_stats_.Run(module.get()), IsOkAndHolds(false));

  HloInstruction* root = module->entry_computation()
                             ->root_instruction()
                             ->while_body()
                             ->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          root->backend_config<GpuBackendConfig>());

  EXPECT_EQ(gpu_config.reification_cost_size(), 1);
  EXPECT_GT(gpu_config.reification_cost()[0].end_to_end_cycles(), 0);
}

TEST_F(GpuCostModelStatsCollectionTest, GemmCostModelAddedToGemmFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
  HloModule test_module

  gemm_fusion_dot_computation {
    p0 = f16[1024,512]{1,0} parameter(0)
    p1 = f16[512,2048]{1,0} parameter(1)
    ROOT %dot.1 = f16[1024,2048]{1,0} dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f16[1024,512]{1,0} parameter(0)
    p1 = f16[512,2048]{1,0} parameter(1)
    ROOT gemm_fusion_dot = f16[1024,2048]{1,0} fusion(p0, p1), kind=kCustom,
      calls=gemm_fusion_dot_computation,
      backend_config={
        "fusion_backend_config": {
          "kind":"__triton_nested_gemm_fusion",
          "block_level_fusion_config": {
            "num_warps":"4",
            "output_tiles":[{"sizes":["64","128"]}],
            "num_ctas":1,
            "num_stages":3
          }
        }
      }
    }
    )hlo"));

  EXPECT_THAT(cost_model_stats_.Run(module.get()), IsOkAndHolds(false));

  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          root->backend_config<GpuBackendConfig>());

  EXPECT_THAT(gpu_config.reification_cost(),
              Contains(Truly([](const ReificationCost& cost) {
                return cost.name() == "experimental-gemm-cost-model" &&
                       cost.end_to_end_cycles() > 0;
              })));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
