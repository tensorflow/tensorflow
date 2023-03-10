/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

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
  debug_options.set_xla_gpu_enable_mlir_lowering(false);
  config.set_debug_options(debug_options);
  auto hlo_module = ParseAndReturnVerifiedModule(kAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @fusion
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK-NOT: fadd
; CHECK: }
      )",
                     /*match_optimized_ir=*/false);
}

TEST_F(GpuUnrollingTest, UnrollUnfusedAdd) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_mlir_lowering(false);
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
; CHECK-LABEL: @add
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fadd
; CHECK: store float
; CHECK-NOT: fadd
; CHECK: }
      )",
                     /*match_optimized_ir=*/false);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedSine) {
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
; CHECK: load float
; CHECK-NOT: load float
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedCosine) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} cosine(p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load float
; CHECK-NOT: load float
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedPower) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} power(p0, p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  // There are 2 loads, because the 2 parameters are read separately - the
  // kernel is not aware that they are the same.
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedAtan2) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[16000000]{0} parameter(0)
      ROOT s = f32[16000000]{0} atan2(p0, p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  // There are 2 loads, because the 2 parameters are read separately - the
  // kernel is not aware that they are the same.
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK: }
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
; CHECK-LABEL: @fusion
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK-NOT: load float
; CHECK: fadd
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK-NOT: load float
; CHECK: fmul
; CHECK: store float
; CHECK: store float
; CHECK-NOT: store float
; CHECK-NOT: store float
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK-NOT: load float
; CHECK: fadd
; CHECK: load float
; CHECK: load float
; CHECK-NOT: load float
; CHECK-NOT: load float
; CHECK: fmul
; CHECK: store float
; CHECK: store float
; CHECK-NOT: store float
; CHECK-NOT: store float
; CHECK: }
      )",
                     /*match_optimized_ir=*/false);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
