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
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuUnrollingTest : public GpuCodegenTest {};

const char *const kAddModule = R"(
    HloModule test_module

    fused_computation {
      p0.param_0 = f32[2,2]{1,0} parameter(0)
      p1.param_1 = f32[2,2]{1,0} parameter(1)
      ROOT add = f32[2,2] add(p0.param_0, p1.param_1)
    }

    ENTRY BroadcastIntoAdd {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = f32[2,2]{1,0} parameter(1)
      ROOT fusion = f32[2,2]{1,0} fusion(p0, p1), kind=kLoop,
                                                  calls=fused_computation
    })";

TEST_F(GpuUnrollingTest, DoNotUnroll) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  config.set_debug_options(debug_options);
  auto hlo_module = ParseAndReturnVerifiedModule(kAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @fusion
; CHECK: fadd
; CHECK-NOT: fadd
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollFourTimes) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  // We request a factor of 8, but the computation works on 4 elements, limiting
  // the maximum unroll factor.
  debug_options.set_xla_gpu_max_kernel_unroll_factor(8);
  debug_options.set_xla_gpu_enable_mlir_lowering(false);
  config.set_debug_options(debug_options);
  auto hlo_module = ParseAndReturnVerifiedModule(kAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @fusion
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK-NOT: fadd
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

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
; CHECK: load <4 x float>
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK-NOT: fadd
; CHECK: store <4 x float>
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollUnfusedAdd) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(4);
  debug_options.set_xla_gpu_enable_mlir_lowering(false);
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY AddFunc {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = f32[2,2]{1,0} parameter(1)
      ROOT add = f32[2,2]{1,0} add(p0, p1)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @add
; CHECK: load <4 x float>
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK: fadd
; CHECK-NOT: fadd
; CHECK: store <4 x float>
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedSine) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(4);
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} sine(p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  // Note: On ROCm side, we do bare minimal to make the test pass.
  // "sine" function is in different code generation path from nvptx: on
  // ROCm platform, it get pulled in from ROCm-Device-Libs, whereas in
  // Cuda, generated llvm IR is compiled PTX.
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK: __ocml_sin_f32
; CHECK-NOT: load float
)"
                                         : R"(
; CHECK: load float
; CHECK-NOT: load float
}
)";

  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedCosine) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(4);
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} cosine(p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  // Note: On ROCm side, we do bare minimal to make the test pass.
  // "cosine" function is in different code generation path from nvptx: on
  // ROCm platform, it get pulled in from ROCm-Device-Libs, whereas in
  // Cuda, generated llvm IR is compiled PTX.
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK: __ocml_cos_f32
; CHECK-NOT: load float
)"
                                         : R"(
; CHECK: load float
; CHECK-NOT: load float
}
)";

  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedPower) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(4);
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[1600000]{0} parameter(0)
      ROOT s = f32[1600000]{0} power(p0, p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load float
; CHECK-NOT: load float
}
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, DisabledUnrollUnfusedAtan2) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(4);
  config.set_debug_options(debug_options);

  const char *const kUnfusedAddModule = R"(
    HloModule test_module

    ENTRY SineFunc {
      p0 = f32[16000000]{0} parameter(0)
      ROOT s = f32[16000000]{0} atan2(p0, p0)
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kUnfusedAddModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: load float
; CHECK-NOT: load float
}
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuUnrollingTest, UnrollMultiOutputFusion) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(2);
  // Disable layout assignment for this test.  Layout assignment does not expect
  // fusions to be present, and so it does the wrong thing.
  debug_options.add_xla_disable_hlo_passes("layout-assignment");
  config.set_debug_options(debug_options);

  const char *const kMultiOutputFusionModule = R"(
    HloModule test_module

    fused_computation {
      p0.param_0 = f32[2,2]{1,0} parameter(0)
      p1.param_1 = f32[2,2]{1,0} parameter(1)
      add = f32[2,2]{1,0} add(p0.param_0, p1.param_1)
      mul = f32[2,2]{1,0} multiply(p0.param_0, p1.param_1)
      ROOT tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(add, mul)
    }

    ENTRY BroadcastIntoAdd {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = f32[2,2]{1,0} parameter(1)
      ROOT fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(p0, p1), kind=kLoop,
                                                   calls=fused_computation
    })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(kMultiOutputFusionModule, config).value();

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: @fusion
; CHECK: load <2 x float>
; CHECK: load <2 x float>
; CHECK-NOT: load <2 x float>
; CHECK: fadd
; CHECK: fmul
; CHECK: fadd
; CHECK: fmul
; CHECK: store <2 x float>
; CHECK: store <2 x float>
; CHECK-NOT: store <2 x float>
; CHECK-NOT: fadd
; CHECK-NOT: fmul
; CHECK: }
      )",
                     /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
