/* Copyright 2022 The OpenXLA Authors.

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
#include <utility>

#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

class CompilationParallelismTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Use multiple threads for compilation
    debug_options.set_xla_gpu_force_compilation_parallelism(4);
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
    return debug_options;
  }
};

TEST_F(CompilationParallelismTest, SharedConstantInMultipleReduce) {
  const char* hlo_text = R"(
HloModule Module

%add_computation (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add0 = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %fused_computation.371 {
  %param_0 = f32[4,8,32]{2,1,0} parameter(0)
  %param_1 = f32[4,10,32]{2,1,0} parameter(1)
  %constant_0 = f32[] constant(0.0)
  %reduce_0 = f32[4,8]{1,0} reduce(f32[4,8,32]{2,1,0} %param_0, f32[] %constant_0), dimensions={2}, to_apply=%add_computation
  %reduce_1 = f32[4,10]{1,0} reduce(f32[4,10,32]{2,1,0} %param_1, f32[] %constant_0), dimensions={2}, to_apply=%add_computation
  ROOT %tule = (f32[4,8]{1,0}, f32[4,10]{1,0}) tuple(%reduce_0, %reduce_1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  // Make sure constant folding works correctly
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK-NOT: ld.global.nc.f32 	%f2, [buffer_for_constant_0];
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
