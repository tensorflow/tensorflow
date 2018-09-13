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

class GpuKernelTilingTest : public GpuCodegenTest {
 protected:
  GpuKernelTilingTest() {}

  // Most tests in this file want to skip layout assignment, but a few need it
  // enabled.
  HloModuleConfig ConfigWithLayoutAssignment() {
    return GetModuleConfigForTest();
  }

  HloModuleConfig ConfigWithoutLayoutAssignment() {
    HloModuleConfig config;
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable layout_assignment to use the preassigned layouts.
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(GpuKernelTilingTest, UnnestedTransposeWithProperDimensionsTiled) {
  const char *const kHloString = R"(
    HloModule unnested_transpose_1

    ENTRY unnested_transpose_1 {
      para0 = f16[32,3,64]{2,1,0} parameter(0)
      ROOT copy1 = f16[32,3,64]{1,0,2} copy(para0)
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  //
  // We must enable layout assignment in order for this test to work correctly.
  // AlgebraicSimplifier removes copy1; it's added back by layout assignment,
  // which respects the module's entry computation layout.  But if we don't run
  // layout assignment...well, nobody else adds the copy back.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @copy
; CHECK: tail call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, UnnestedTransposeWithSmallDimensionsNotTiled) {
  const char *const kHloString = R"(
    HloModule unnested_transpose_2

    ENTRY unnested_transpose_2 {
      para0 = f16[2,3,64]{2,1,0} parameter(0)
      ROOT copy1 = f16[2,3,64]{1,0,2} copy(para0)
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.  As in
  // UnnestedTransposeWithProperDimensionsTiled, we must run layout assignment
  // here.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @copy
; CHECK-NOT: tail call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, SimpleFusionWithTransposeTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_1
    fused_computation.1 {
      param0 = f32[4,5,6,7,8]{4,3,2,1,0} parameter(0)
      copy = f32[4,5,6,7,8]{2,1,4,3,0} copy(param0)
      ROOT convert = f16[4,5,6,7,8]{2,1,4,3,0} convert(copy)
    }

    ENTRY copy_in_fusion_run_without_hlo_passes {
      para0 = f32[4,5,6,7,8]{4,3,2,1,0} parameter(0)
      ROOT fusion.1 = f16[4,5,6,7,8]{2,1,4,3,0} fusion(para0), kind=kLoop,
        calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: tail call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, MultipleOutputFusionWithOnePossibleTransposeTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_1
    fused_computation.1 {
      param0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      param1 = f16[8,31,31,65]{3,2,1,0} parameter(1)
      copy0 = f16[8,31,31,65]{2,1,3,0} copy(param0)
      copy1 = f16[8,31,31,65]{2,1,3,0} copy(param1)
      ROOT tuple1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        tuple(copy0, copy1)
    }

    ENTRY multiple_output_fusion_1 {
      para0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      para1 = f16[8,31,31,65]{3,2,1,0} parameter(1)
      ROOT fusion.1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        fusion(para0,para1), kind=kLoop, calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: tail call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest,
       MultipleOutputFusionWithTwoPossibleTransposesNotTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_2
    fused_computation.1 {
      param0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      param1 = f16[8,31,31,65]{1,3,2,0} parameter(1)
      copy2 = f16[8,31,31,65]{2,1,3,0} copy(param0)
      copy3 = f16[8,31,31,65]{2,1,3,0} copy(param1)
      ROOT tuple1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        tuple(copy2, copy3)
    }

    ENTRY multiple_output_fusion_2 {
      para0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      para1 = f16[8,31,31,65]{1,3,2,0} parameter(1)
      ROOT fusion1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        fusion(para0,para1), kind=kLoop, calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: tail call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
