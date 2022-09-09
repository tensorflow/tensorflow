/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class TransposeEmitterTest : public gpu::GpuCodegenTest {
 protected:
  TransposeEmitterTest() {}
};

TEST_F(TransposeEmitterTest, Simple) {
  const char* const kHloString = R"(
  HloModule m

  ENTRY e {
    para0 = f16[32,16,64]{2,1,0} parameter(0)
    ROOT copy1 = f16[32,16,64]{1,0,2} copy(para0)
  })";

  auto expected_ir = R"(
; CHECK: call void BARRIER()
)";
  CompileAndVerifyIr(kHloString, MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(TransposeEmitterTest, MultipleCopies) {
  const char* hlo = R"(
HloModule m

%fused_computation {
  %param_0.1 = f32[16,32]{1,0} parameter(0)
  %s.1 = f32[16,32]{1,0} sqrt(%param_0.1)
  %c.1 = f32[16,32]{0,1} copy(%s.1)
  %c1.1 = f32[16,32]{0,1} copy(%param_0.1)
  ROOT %tuple = (f32[16,32]{0,1}, f32[16,32]{0,1}) tuple(%c.1, %c1.1)
}

ENTRY main {
  %p = f32[16,32]{1,0} parameter(0)
  ROOT %fusion = (f32[16,32]{0,1}, f32[16,32]{0,1}) fusion(%p), kind=kInput, calls=%fused_computation
}
  )";

  CompileAndVerifyIr(hlo, MakePlatformSpecificLlvm(R"(
// CHECK: call void BARRIER()
  )"),
                     /*match_optimized_ir=*/true,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo, ErrorSpec{1e-3}));
}

TEST_F(TransposeEmitterTest, CopyAndInput) {
  const char* hlo = R"(
HloModule m

%fused_computation {
  %param_0.1 = f32[16,32]{1,0} parameter(0)
  %s.1 = f32[16,32]{1,0} sqrt(%param_0.1)
  %c.1 = f32[16,32]{0,1} copy(%s.1)
  %c1.1 = f32[16,32]{1,0} exponential(%param_0.1)
  ROOT %tuple = (f32[16,32]{0,1}, f32[16,32]{1,0}) tuple(%c.1, %c1.1)
}

ENTRY entry {
  %p = f32[16,32]{1,0} parameter(0)
  ROOT %fusion = (f32[16,32]{0,1}, f32[16,32]{1,0}) fusion(%p), kind=kInput, calls=%fused_computation
}
  )";

  CompileAndVerifyIr(hlo, MakePlatformSpecificLlvm(R"(
// CHECK: call void BARRIER()
  )"),
                     /*match_optimized_ir=*/true,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace xla
