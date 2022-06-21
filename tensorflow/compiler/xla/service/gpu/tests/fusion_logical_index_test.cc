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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class FusionLogicalIndexTest : public GpuCodegenTest {};

TEST_F(FusionLogicalIndexTest, FusionLogicalIndexStore) {
  const char* hlo_text = R"(
HloModule TestModule

fused_computation.18 {
  select.12 = f32[768000,16]{1,0} parameter(0)
  select.13 = f32[768000,16]{1,0} parameter(1)
  maximum.1 = f32[768000,16]{1,0} maximum(select.13, select.12)
  ROOT reshape.437 = f32[1,480,400,64]{2,1,3,0} reshape(maximum.1)
}


ENTRY entry {
    select.12 = f32[768000,16]{1,0} parameter(0)
    select.13 = f32[768000,16]{1,0} parameter(1)
    ROOT fusion.18 = f32[1,480,400,64]{2,1,3,0} fusion(select.12, select.13), kind=kLoop, calls=fused_computation.18
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK:  %[[block_id:.*]] = call i32 @llvm.amdgcn.workgroup.id.x(), !range !1
; CHECK:  %[[thread_id:.*]] = call i32 @llvm.amdgcn.workitem.id.x(), !range !2
; CHECK:  %[[block_start_index:.*]] = mul nuw nsw i32 %[[block_id]], [[block_size:.*]]
; CHECK:  %[[linear_index:.*]] = add nuw nsw i32 %[[block_start_index]], %[[thread_id]]
; CHECK:  %[[index_1:.*]] = urem i32 %[[linear_index]], 64
; CHECK:  %[[base_1:.*]] = udiv i32 %[[linear_index]], 64
; CHECK:  %[[index_3:.*]] = urem i32 %[[base_1]], 400
; CHECK:  %[[base_3:.*]] = udiv i32 %[[base_1]], 400
; CHECK:  %[[index_2:.*]] = urem i32 %[[base_3]], 480
; CHECK:  %[[pointer:.*]] = getelementptr inbounds [1 x [64 x [480 x [400 x float]]]], [1 x [64 x [480 x [400 x float]]]]* %5, i32 0, i32 0, i32 %[[index_1]], i32 %[[index_2]], i32 %[[index_3]]
; CHECK:  store float %[[result_value:.*]], float* %[[pointer]], align 4
  )"                                     : R"(
; CHECK:  %[[block_id:.*]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK:  %[[thread_id:.*]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK:  %[[block_start_index:.*]] = mul nuw nsw i32 %[[block_id]], [[block_size:.*]]
; CHECK:  %[[linear_index:.*]] = add nuw nsw i32 %[[block_start_index]], %[[thread_id]]
; CHECK:  %[[index_1:.*]] = urem i32 %[[linear_index]], 64
; CHECK:  %[[base_1:.*]] = udiv i32 %[[linear_index]], 64
; CHECK:  %[[index_3:.*]] = urem i32 %[[base_1]], 400
; CHECK:  %[[base_3:.*]] = udiv i32 %[[base_1]], 400
; CHECK:  %[[index_2:.*]] = urem i32 %[[base_3]], 480
; CHECK:  %[[pointer:.*]] = getelementptr inbounds [1 x [64 x [480 x [400 x float]]]], ptr %2, i32 0, i32 0, i32 %[[index_1]], i32 %[[index_2]], i32 %[[index_3]]
; CHECK:  store float %[[result_value:.*]], ptr %[[pointer]], align 4
  )";

  CompileAndVerifyIr(hlo_text, expected_ir);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
