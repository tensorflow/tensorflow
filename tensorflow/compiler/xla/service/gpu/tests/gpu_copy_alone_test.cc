/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

namespace xla {
namespace gpu {

namespace {

// WARNING: This tests must be alone in its file!  Otherwise, the
// error isn't caught. We expect and CUDA_ERROR_ILLEGAL_ADDRESS to be
// thrown with the old buggy code.
class CopyAloneNoOptTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // The test MultiOutputStore contain a MOF fusion and XLA optimizer pass
    // doesn't like this.
    debug_options.set_xla_disable_all_hlo_passes(true);
    return debug_options;
  }
};

TEST_F(CopyAloneNoOptTest, CopyTranspose) {
  const char* hlo_text = R"(
HloModule mod
ENTRY main {
  %param = f32[8,32,32,32,16]{4,3,2,1,0} parameter(0)
  ROOT %copy = f32[8,32,32,32,16]{3,2,1,4,0} copy(f32[8,32,32,32,16]{4,3,2,1,0} %param)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK-NOT: ld.global.nc.v2
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
