/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/custom_kernel_fusion_autotuner.h"

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class CustomKernelFusionAutotunerTest : public HloTestBase {
 public:
  CustomKernelFusionAutotunerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  void SetUp() override { HloTestBase::SetUp(); }

  void TearDown() override { HloTestBase::TearDown(); }
};

TEST_F(CustomKernelFusionAutotunerTest, DontRunOnNonCustomFusions) {
  const std::string hlo_string = R"(
  HloModule test_module, entry_computation_layout={(f32[20000,20000]{1,0}, f32[20000,20000]{1,0})->(f32[20000,20000]{1,0}, f32[20000,20000]{1,0})}

    // Not a CustomFusion!
    %fused_computation (p0.param_0: f32[20000,20000], p1.param_1: f32[20000,20000]) -> (f32[20000,20000], f32[20000,20000]) {
      %p0.param_0 = f32[20000,20000]{1,0} parameter(0)
      %p1.param_1 = f32[20000,20000]{1,0} parameter(1)
      %add = f32[20000,20000]{1,0} add(f32[20000,20000]{1,0} %p0.param_0, f32[20000,20000]{1,0} %p1.param_1)
      %mul = f32[20000,20000]{1,0} multiply(f32[20000,20000]{1,0} %p0.param_0, f32[20000,20000]{1,0} %p1.param_1)
      ROOT %tuple = (f32[20000,20000]{1,0}, f32[20000,20000]{1,0}) tuple(f32[20000,20000]{1,0} %add, f32[20000,20000]{1,0} %mul)
    }

    ENTRY %BroadcastIntoAdd (p0: f32[20000,20000], p1: f32[20000,20000]) -> (f32[20000,20000], f32[20000,20000]) {
      %p0 = f32[20000,20000]{1,0} parameter(0)
      %p1 = f32[20000,20000]{1,0} parameter(1)
      ROOT %fusion = (f32[20000,20000]{1,0}, f32[20000,20000]{1,0}) fusion(f32[20000,20000]{1,0} %p0, f32[20000,20000]{1,0} %p1), kind=kLoop, calls=%fused_computation
    }
  )";
  std::unique_ptr<HloModule> hlo_module =
      ParseAndReturnVerifiedModule(hlo_string).value();

  HloPassPipeline pipeline("custom_kernel_fusion_autotuner");
  DebugOptions debug_options;
  AutotuneConfig autotune_config =
      AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                  backend().memory_allocator()},
                     debug_options};
  pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);

  // Check that that an HLO computation, which is a non custom fusion gets
  // filtered out and passes. If the autotuner would try to run on a non custom
  // fusion it would fail.
  ASSERT_TRUE(pipeline.Run(hlo_module.get()).ok());
}

TEST_F(CustomKernelFusionAutotunerTest,
       CustomKernelFusionAutotunerPassSucceeds) {
  const std::string hlo_string = R"(
    HloModule extracted

    cutlass_gemm {
      p0 = f32[15,19]{1,0} parameter(0)
      p1 = f32[19,17]{1,0} parameter(1)
      ROOT r = f32[15, 17]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY region_198.14436 {
      p.0 = f32[15,19]{1,0} parameter(0)
      p.1 = f32[19,17]{1,0} parameter(1)
      ROOT cutlass_gemm = f32[15,17]{1,0} fusion(p.0, p.1), kind=kCustom, calls=cutlass_gemm, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"cutlass_gemm","kernel_index":0}},"force_earliest_schedule":false}
    }
  )";
  std::unique_ptr<HloModule> hlo_module =
      ParseAndReturnVerifiedModule(hlo_string).value();

  HloPassPipeline pipeline("custom_kernel_fusion_autotuner");
  DebugOptions debug_options;
  AutotuneConfig autotune_config =
      AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                  backend().memory_allocator()},
                     debug_options};
  pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);
  ASSERT_TRUE(pipeline.Run(hlo_module.get()).ok());
}

TEST_F(CustomKernelFusionAutotunerTest,
       CustomKernelFusionAutotunerPassUpdatesUpdatesKernelIndex) {
  const std::string hlo_string = R"(
    HloModule extracted

    cutlass_gemm {
      p0 = f32[15,19]{1,0} parameter(0)
      p1 = f32[19,17]{1,0} parameter(1)
      ROOT r = f32[15, 17]{1,0} dot(p0, p1), lhs_contracting_dims={1},
      rhs_contracting_dims={0}
    }

    ENTRY region_198.14436 {
      p.0 = f32[15,19]{1,0} parameter(0)
      p.1 = f32[19,17]{1,0} parameter(1)
      ROOT cutlass_gemm = f32[15,17]{1,0} fusion(p.0, p.1), kind=kCustom,
      calls=cutlass_gemm,
      backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"cutlass_gemm","kernel_index":-1}},"force_earliest_schedule":false}
    }
  )";

  HloPassPipeline pipeline("custom_kernel_fusion_autotuner");
  DebugOptions debug_options;
  AutotuneConfig autotune_config =
      AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                  backend().memory_allocator()},
                     debug_options};
  pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);

  std::string expected = R"(
    CHECK: "kernel_index":0
  )";
  RunAndFilecheckHloRewrite(hlo_string, std::move(pipeline), expected);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
