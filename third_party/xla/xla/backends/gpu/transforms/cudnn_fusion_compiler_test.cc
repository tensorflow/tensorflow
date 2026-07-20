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

#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace {

const char kCudnnConvolutionFusionWithS8ConstHlo[] = R"(
  fusion1 {
    p0 = s8[4,48,48,64] parameter(0)
    p1 = s8[64,3,3,64] parameter(1)
    conv = s8[4,48,48,64] convolution(p0, p1),
      window={size=3x3 pad=1_1x1_1},
      dim_labels=b01f_o01i->b01f,
      convolution_kind=fprop
    one = s8[] constant(1)
    ones = s8[4,48,48,64] broadcast(one), dimensions={}
    result = s8[4,48,48,64] add(conv, ones)
    zero = s8[] constant(0)
    zeros = s8[4,48,48,64] broadcast(zero), dimensions={}
    ROOT relu = s8[4,48,48,64] maximum(result, zeros)
  }

  ENTRY e {
    p0 = s8[4,48,48,64] parameter(0)
    p1 = s8[64,3,3,64] parameter(1)
    ROOT _ = s8[4,48,48,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$fusion",
        }
      }
  })";

class CudnnFusionCompilerTest : public HloPjRtGpuTestBase {
 protected:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }
};

TEST_F(CudnnFusionCompilerTest,
       GetAvailablePlanCountFromCudnnConvolutionFusionWithS8Const) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kCudnnConvolutionFusionWithS8ConstHlo));

  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  auto* fusion = Cast<HloFusionInstruction>(root);

  ASSERT_OK_AND_ASSIGN(int plan_count,
                       CuDnnFusionCompiler::GetAvailablePlanCount(
                           stream_executor(),
                           stream_executor()->GetDeviceDescription(), *fusion));
  EXPECT_GT(plan_count, 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
