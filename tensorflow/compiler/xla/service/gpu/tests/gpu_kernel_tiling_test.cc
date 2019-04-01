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
; CHECK: call void @llvm.nvvm.barrier0()
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
; CHECK-NOT: call void @llvm.nvvm.barrier0()
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
; CHECK: call void @llvm.nvvm.barrier0()
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
; CHECK: call void @llvm.nvvm.barrier0()
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
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, TransposedInputWithUserReverseNotTiled) {
  const char *const kHloString = R"(
    HloModule FusionTransposeWithReverseNotTiled
    fused_computation.1 {
      arg0 = f32[128,64]{1,0} parameter(0)
      copy0 = f32[128,64]{0,1} copy(arg0)
      ROOT reverse0 = f32[128,64]{0,1} reverse(copy0), dimensions={0}
    }

    ENTRY reverse_break_assumption {
      param0 = f32[128,64]{1,0} parameter(0)
      ROOT fusion0 = f32[128,64]{0,1} fusion(param0), kind=kLoop,
        calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, TransposedInputWithUserBitcastNotTiled) {
  const char *const kHloString = R"(
    HloModule TransposedInputWithUserBitcast

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      ROOT bitcast = f32[20,20]{0,1} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = f32[20,20]{0,1} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, TransposedInputWithoutUnsafeUseTiled) {
  const char *const kHloString = R"(
    HloModule TwoTransposedInputs

    fused_computation {
      param_0 = f32[64,64]{1,0} parameter(0)
      param_1 = f32[64,64]{1,0} parameter(1)
      bitcast = f32[64,64]{0,1} bitcast(param_0)
      copy = f32[64,64]{0,1} copy(param_1)
      ROOT tuple = (f32[64,64]{0,1}, f32[64,64]{0,1}) tuple(bitcast, copy)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[64,64]{1,0} parameter(0)
      parameter.1 = f32[64,64]{1,0} parameter(1)
      ROOT fusion = (f32[64,64]{0,1}, f32[64,64]{0,1})
        fusion(parameter.0, parameter.1),
        kind=kLoop, calls=fused_computation
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionWithPowerOf2OutputElementsUnrolled) {
  const char *const kHloString = R"(
  HloModule column_reduce_powerof2

  reduction {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY kernel_entry {
    constant0 = f32[] constant(0)
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg1_conv = f32[1024,512]{1,0} convert(arg1)
    ROOT reduce = f32[512]{0} reduce(arg1_conv, constant0), dimensions={0}, to_apply=reduction
  })";

  // Check that two calls to llvm.nvvm.atomic are generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK-NOT: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest,
       ColumnReductionWithInputLargerThenReduceInputNotUnrolled) {
  const char *const kHloString = R"(
  HloModule larger_than_reduce_input_parameter

  reduction22 {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  fused_computation {
    constant0 = f32[] constant(0)
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1027,513]{1,0} parameter(1)
    arg1.conv = f32[1024,512]{1,0} convert(arg.1)
    arg2.conv = f32[1027,513]{1,0} convert(arg.2)
    slice2 = f32[1024,512]{1,0} slice(arg2.conv), slice={[2:1026], [1:513]}
    add2 = f32[1024,512]{1,0} add(arg1.conv, slice2)
    ROOT reduce = f32[512]{0} reduce(add2, constant0), dimensions={0},
      to_apply=reduction22
  }

  ENTRY kernel_entry {
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg2 = f16[1027,513]{1,0} parameter(1)
    ROOT fusion = f32[512]{0} fusion(arg1, arg2), kind=kInput,
      calls=fused_computation
  })";

  // Check that one call to llvm.nvvm.atomic is generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK-NOT: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionMOFUnrolled) {
  const char *const kHloString = R"(
  HloModule column_reduce_powerof2_mof

  reduction22 {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  fused_computation {
    constant0 = f32[] constant(0)
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    arg1.conv = f32[1024,512]{1,0} convert(arg.1)
    arg2.conv = f32[1024,512]{1,0} convert(arg.2)
    reduce1 = f32[512]{0} reduce(arg1.conv, constant0), dimensions={0},
      to_apply=reduction22
    reduce2 = f32[512]{0} reduce(arg2.conv, constant0), dimensions={0},
      to_apply=reduction22
    add = f32[1024,512]{1,0} add(arg1.conv, arg2.conv)
    ROOT tuple = (f32[512]{0}, f32[512]{0}, f32[1024,512]{1,0})
      tuple(reduce1, reduce2, add)
  }

  ENTRY kernel_entry {
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg2 = f16[1024,512]{1,0} parameter(1)
    ROOT fusion = (f32[512]{0}, f32[512]{0}, f32[1024,512]{1,0})
      fusion(arg1, arg2), kind=kInput, calls=fused_computation
  })";

  // Check that four calls to llvm.nvvm.atomic are generated.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK-NOT: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[4,32,32,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[4,32,16,12,12]{4,3,2,1,0} reduce(arg0, constant0),
        dimensions={1,6,7}, to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.atomic.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @reduce
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest, RowReductionWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[8,6,64]{2,1,0}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[8,6]{0,1} reduce(arg0, constant0), dimensions={2},
        to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.shfl.sync.down.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @reduce
; CHECK: call float @llvm.nvvm.shfl.sync.down.f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest,
       ColumnReductionResultTwoPartsWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_no_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[8,64,4]{2,1,0}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[8,4]{0,1} reduce(arg0, constant0), dimensions={1},
        to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.atomic.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutLayoutAssignment()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @reduce
; CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
; CHECK: }
)",
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
