/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

class GpuSliceInputFusionTest : public GpuCodegenTest {
 protected:
  GpuSliceInputFusionTest() {}

  // Disable the layout assignment pass as it would throw away the layouts
  // in the fusion computation.
  HloModuleConfig ConfigWithoutLayoutAssignment() {
    HloModuleConfig config;
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable layout_assignment to use the preassigned layouts.
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(GpuSliceInputFusionTest, InputFusionWithOnlyOneSlice) {
  const char *const kHloString = R"(
  HloModule input_fusion_with_only_one_slice

  fused_computation {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    arg1.conv = f32[1024,512]{1,0} convert(arg.1)
    arg2.conv = f32[1024,512]{1,0} convert(arg.2)
    add.1 = f32[1024,512]{1,0} add(arg1.conv, arg2.conv)
    ROOT slice.1 = f32[512,511]{1,0} slice(add.1), slice={[512:1024], [1:512]}
  }

  ENTRY kernel_entry {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    ROOT fusion = f32[512, 511]{1,0} fusion(arg.1, arg.2), kind=kInput,
        calls=fused_computation
  })";

  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: slice_id0
; CHECK: }
)",
                     /*match_optimized_ir=*/false);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0, 0}));
}

TEST_F(GpuSliceInputFusionTest, InputFusionWithATupleOfSlices) {
  const char *const kHloString = R"(
  HloModule input_fusion_with_a_tuple_of_slices

  fused_computation {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    mul.1 = f16[1024,512]{1,0} multiply(arg.1, arg.2)
    add.1 = f16[1024,512]{1,0} add(mul.1, arg.2)
    slice.1 = f16[512,511]{1,0} slice(arg.1), slice={[512:1024], [1:512]}
    slice.2 = f16[0,512]{1,0} slice(add.1), slice={[512:512], [0:512]}
    slice.3 = f16[1,1]{1,0} slice(add.1), slice={[512:513], [511:512]}
    ROOT tuple.1 = (f16[512,511]{1,0}, f16[0,512]{1,0}, f16[1,1]{1,0})
        tuple(slice.1, slice.2, slice.3)
  }

  ENTRY kernel_entry {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    ROOT fusion = (f16[512,511]{1,0}, f16[0,512]{1,0}, f16[1,1]{1,0})
        fusion(arg.1, arg.2), kind=kInput, calls=fused_computation
  })";

  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: slice_id2
; CHECK: }
)",
                     /*match_optimized_ir=*/false);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0, 0}));
}

TEST_F(GpuSliceInputFusionTest, ConcatThenSplit) {
  const char *const kHloString = R"(
  HloModule input_fusion_with_a_tuple_of_slices

  fused_computation {
    arg.1 = f16[1024]{0} parameter(0)
    arg.2 = f16[1024]{0} parameter(1)
    arg.3 = f16[1023]{0} parameter(2)
    arg.4 = f16[1023]{0} parameter(3)
    mul.1 = f16[1024]{0} multiply(arg.1, arg.2)
    add.1 = f16[1023]{0} add(arg.3, arg.4)
    concat.1 = f16[2047]{0} concatenate(mul.1, add.1), dimensions={0}
    slice.1 = f16[1024]{0} slice(concat.1), slice={[0:1024]}
    slice.2 = f16[1023]{0} slice(concat.1), slice={[1024:2047]}
    slice.3 = f16[0]{0} slice(concat.1), slice={[2047:2047]}
    ROOT tuple.1 = (f16[1024]{0}, f16[1023]{0}, f16[0]{0})
        tuple(slice.1, slice.2, slice.3)
  }

  ENTRY kernel_entry {
    arg.1 = f16[1024]{0} parameter(0)
    arg.2 = f16[1024]{0} parameter(1)
    arg.3 = f16[1023]{0} parameter(2)
    arg.4 = f16[1023]{0} parameter(3)
    ROOT fusion = (f16[1024]{0}, f16[1023]{0}, f16[0]{0})
        fusion(arg.1, arg.2, arg.3, arg.4), kind=kInput, calls=fused_computation
  })";

  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-LABEL: define void @fusion
; CHECK: slice_id2
; CHECK: }
)",
                     /*match_optimized_ir=*/false);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0, 0}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
