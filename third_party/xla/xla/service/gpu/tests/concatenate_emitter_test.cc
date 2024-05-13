/* Copyright 2024 The OpenXLA Authors.

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

#include <string>

#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class ConcatenateEmitterTest : public gpu::GpuCodegenTest {
 protected:
  ConcatenateEmitterTest() = default;
};

TEST_F(ConcatenateEmitterTest, Simple) {
  const char* const kHloString = R"(
  HloModule module

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT concat = f32[256] concatenate(param0, param1), dimensions={0}
  })";

  auto expected_ir = R"(
; CHECK-DAG: %[[ARG0:.*]] = addrspacecast ptr %arg0
; CHECK-DAG: %[[ARG1:.*]] = addrspacecast ptr %arg1
; CHECK-DAG: %[[ARG2:.*]] = addrspacecast ptr %arg2
; CHECK: %[[PTR:.*]] = getelementptr float, ptr addrspace(1) %[[ARG0]]
; CHECK-DAG: %[[VAL:.*]] = load float, ptr addrspace(1) %[[PTR]]
; CHECK-DAG: %[[DST:.*]] = getelementptr inbounds [256 x float], ptr addrspace(1) %[[ARG2]]
; CHECK: store float %[[VAL]], ptr addrspace(1) %[[DST]]
; CHECK: %[[PTR:.*]] = getelementptr float, ptr addrspace(1) %[[ARG1]]
; CHECK-DAG: %[[VAL:.*]] = load float, ptr addrspace(1) %[[PTR]]
; CHECK-DAG: %[[PTR:.*]] = getelementptr inbounds i8, ptr addrspace(1) %[[DST]], i64 512
; CHECK: store float %[[VAL]], ptr addrspace(1) %[[PTR]]
; CHECK: !"reqntidx", i32 128
)";
  CompileAndVerifyIr(kHloString, MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ConcatenateEmitterTest, PrologueAndEpilogue) {
  const char* const kHloString = R"(
  HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    negate = f32[128] negate(param0)
    param1 = f32[128] parameter(1)
    concat = f32[256] concatenate(negate, param1), dimensions={0}
    param2 = f32[256] parameter(2)
    ROOT add = f32[256] add(concat, param2)
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    param2 = f32[256] parameter(2)
    ROOT %fusion = f32[256] fusion(param0, param1, param2), kind=kInput, calls=fused_computation
  })";

  auto expected_ir = R"(
; CHECK-DAG: %[[ARG0:.*]] = addrspacecast ptr %arg0
; CHECK-DAG: %[[ARG1:.*]] = addrspacecast ptr %arg1
; CHECK-DAG: %[[ARG2:.*]] = addrspacecast ptr %arg2
; CHECK-DAG: %[[ARG3:.*]] = addrspacecast ptr %arg3
; CHECK: %[[PTR:.*]] = getelementptr float, ptr addrspace(1) %[[ARG0]]
; CHECK: %[[RHS:.*]] = load float, ptr addrspace(1) %[[PTR]]
; CHECK: %[[SRC:.*]] = getelementptr inbounds [256 x float], ptr addrspace(1) %[[ARG2]]
; CHECK: %[[LHS:.*]] = load float, ptr addrspace(1) %[[SRC]]
; CHECK: %[[VAL:.*]] = fsub float %[[LHS]], %[[RHS]]
; CHECK: %[[DST:.*]] = getelementptr inbounds [256 x float], ptr addrspace(1) %[[ARG3]]
; CHECK: store float %[[VAL]], ptr addrspace(1) %[[DST]]
; CHECK: %[[PTR:.*]] = getelementptr float, ptr addrspace(1) %[[ARG1]]
; CHECK: %[[LHS:.*]] = load float, ptr addrspace(1) %[[PTR]]
; CHECK: %[[PTR:.*]] = getelementptr inbounds i8, ptr addrspace(1) %[[SRC]], i64 512
; CHECK: %[[RHS:.*]] = load float, ptr addrspace(1) %[[PTR]]
; CHECK: %[[VAL:.*]] = fadd float %[[LHS]], %[[RHS]]
; CHECK: %[[PTR:.*]] = getelementptr inbounds i8, ptr addrspace(1) %[[DST]], i64 512
; CHECK: store float %[[VAL]], ptr addrspace(1) %[[PTR]]
)";
  CompileAndVerifyIr(kHloString, MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true,
                     /*run_optimization_passes=*/false);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ConcatenateEmitterTest, MajorDimension) {
  const char* const kHloString = R"(
  HloModule module

  fused_computation {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT concat = f32[32,16] concatenate(param0, param1), dimensions={0}
  }

  ENTRY main {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT %fusion = f32[32,16] fusion(param0, param1), kind=kInput, calls=fused_computation
  })";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ConcatenateEmitterTest, DifferentSizes) {
  const char* const kHloString = R"(
  HloModule module

  ENTRY main {
    param0 = f32[112] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT concat = f32[240] concatenate(param0, param1), dimensions={0}
  })";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ConcatenateEmitterTest, RepeatedInput) {
  const char* const kHloString = R"(
  HloModule module

  ENTRY main {
    param0 = f32[128] parameter(0)
    ROOT concat = f32[256] concatenate(param0, param0), dimensions={0}
  })";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ConcatenateEmitterTest, BitcastEpilogue) {
  const char* const kHloString = R"(
  HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    concat = f32[256] concatenate(param0, param1), dimensions={0}
    ROOT bitcast = f32[1,16,16] bitcast(concat)
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT %fusion = f32[1,16,16] fusion(param0, param1), kind=kInput, calls=fused_computation
  })";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace xla
