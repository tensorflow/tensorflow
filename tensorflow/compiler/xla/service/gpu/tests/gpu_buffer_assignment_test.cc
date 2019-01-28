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

class GpuBufferAssignmentTest : public GpuCodegenTest {
 public:
  HloModuleConfig ConfigWithoutHloPasses() {
    HloModuleConfig config;
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable layout_assignment to use the preassigned layouts.
    debug_options.xla_disable_all_hlo_passes();
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(GpuBufferAssignmentTest, InstructionNameWithHyphenSanitized) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal-to = s32[2]{0} constant({42, 73})
    })";

  // Check that '-' in the instruction name is changed to '_'.
  auto hlo_module = ParseHloString(kHloString).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: buffer_for_equal_to =
)",
                     /*match_optimized_ir=*/true);

  // TODO(bixia): The run fails randomly.
  // Check that the kernel runs correctly.
  // EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuBufferAssignmentTest, BufferSanitizedNameCollisionResolved) {
  const char *const kHloString = R"(
    HloModule BufferSanitizedName
      ENTRY kernelEntry {
      equal.to = s32[2]{0} constant({42, 73})
      equal-to = s32[2]{0} constant({67, 3})
      ROOT add = s32[2]{0} add(equal.to, equal-to)
    })";

  // Turn of Hlo passes to prevent constant folding.
  //
  // Check that '-' and '.' in the instruction name are changed to '_', and
  // name collision is resolved by LLVM.
  auto hlo_module =
      ParseHloString(kHloString, ConfigWithoutHloPasses()).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK: buffer_for_equal_to =
; CHECK: buffer_for_equal_to1 =
)",
                     /*match_optimized_ir=*/false);

  // TODO(bixia): There is another bug that prevents this from running
  //              correctly.
  // Check that the kernel runs correctly.
  // EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
