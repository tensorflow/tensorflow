/* Copyright 2018 The OpenXLA Authors.

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

#include <utility>

#include "xla/debug_options_flags.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuUnrollingTest : public GpuCodegenTest {};

const char *const kAddModule = R"(
    HloModule test_module

    fused_computation {
      p0.param_0 = f32[20000,20000]{1,0} parameter(0)
      p1.param_1 = f32[20000,20000]{1,0} parameter(1)
      ROOT add = f32[20000,20000] add(p0.param_0, p1.param_1)
    }

    ENTRY BroadcastIntoAdd {
      p0 = f32[20000,20000]{1,0} parameter(0)
      p1 = f32[20000,20000]{1,0} parameter(1)
      ROOT fusion = f32[20000,20000]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_computation
    })";

TEST_F(GpuUnrollingTest, UnrollDefaultTimes) {
  // The default unrolling factor is 4.
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsFromFlags();
  config.set_debug_options(debug_options);
  auto hlo_module = ParseAndReturnVerifiedModule(kAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @{{[a-z_]*}}fusion
; CHECK-NOT: load float
; CHECK-NOT: store float
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: store <4 x float>
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollUnfusedAdd) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module
    ENTRY AddFunc {
      p0 = f32[20000,20000]{1,0} parameter(0)
      p1 = f32[20000,20000]{1,0} parameter(1)
      ROOT add = f32[20000,20000]{1,0} add(p0, p1)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @wrapped_add
; CHECK-NOT: load float
; CHECK-NOT: store float
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: store <4 x float>
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollUnfusedSine) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} sine(p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load <4 x float>
; CHECK-NOT: load <4 x float>
; CHECK: store <4 x float>
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollMultiOutputFusion) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  // Disable layout assignment for this test.  Layout assignment does not expect
  // fusions to be present, and so it does the wrong thing.
  debug_options.add_xla_disable_hlo_passes("layout-assignment");
  config.set_debug_options(debug_options);

  const char *const kMultiOutputFusionModule = R"(
    HloModule test_module

    fused_computation {
      p0.param_0 = f32[20000,20000]{1,0} parameter(0)
      p1.param_1 = f32[20000,20000]{1,0} parameter(1)
      add = f32[20000,20000]{1,0} add(p0.param_0, p1.param_1)
      mul = f32[20000,20000]{1,0} multiply(p0.param_0, p1.param_1)
      ROOT tuple = (f32[20000,20000]{1,0}, f32[20000,20000]{1,0}) tuple(add, mul)
    }

    ENTRY BroadcastIntoAdd {
      p0 = f32[20000,20000]{1,0} parameter(0)
      p1 = f32[20000,20000]{1,0} parameter(1)
      ROOT fusion = (f32[20000,20000]{1,0}, f32[20000,20000]{1,0}) fusion(p0, p1), kind=kLoop,
                                                   calls=fused_computation
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kMultiOutputFusionModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @{{[a-z_]*}}fusion
; CHECK-NOT: load float
; CHECK-NOT: store float
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: store <4 x float>
; CHECK: store <4 x float>
      )",
                     /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
