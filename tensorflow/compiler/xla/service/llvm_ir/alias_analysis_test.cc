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

#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace cpu {
namespace {
class AliasAnalysisTest : public CpuCodegenTest {};

void FakeCustomCallTarget(float* out, float** in) {}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(FakeCustomCallTarget);

TEST_F(AliasAnalysisTest, EmbeddedComputationParamsMayAliasTemps) {
  const char* hlo_string = R"(
HloModule while

body {
  const.0.125 = f32[] constant(0.125)
  body.state = f32[] parameter(0)
  ROOT add.2.2 = f32[] add(const.0.125, body.state)
}

condition {
  const.100 = f32[] constant(100)
  condition.state = f32[] parameter(0)
  addend = f32[] custom-call(condition.state), custom_call_target="FakeCustomCallTarget"
  add = f32[] add(addend, condition.state)
  ROOT greater-than = pred[] compare(const.100, add), direction=GT
}

ENTRY while3 {
  const.0 = f32[] constant(0)
  ROOT while = f32[] while(const.0), condition=condition, body=body
}
)";

  CompileAndVerifyIr(hlo_string, R"(
; CHECK-LABEL: @body(i8* %retval
; CHECK: %[[add_result:.*]] = fadd fast float %[[fadd_lhs:.*]], %[[fadd_rhs:.*]]
; CHECK: store float %[[add_result]], float* %[[store_dest:.*]], align 4, !alias.scope ![[alias_scope_md_for_store:[0-9]+]]
;
; CHECK-LABEL: @condition(i8* %retval, i8* noalias %run_options, i8** noalias %params
; CHECK: %[[cond_state_buf_ptr:.*]] = getelementptr inbounds i8*, i8** %buffer_table, i64 0
; CHECK: %[[cond_state_buf_untyped:.*]] = load i8*, i8** %[[cond_state_buf_ptr]]
; CHECK: %[[cond_state_buf_typed:.*]] = bitcast i8* %[[cond_state_buf_untyped]] to float*
; CHECK: load float, float* %[[cond_state_buf_typed]], align 4, !alias.scope ![[alias_scope_md_for_store]], !noalias ![[noalias_md_for_load:.*]]
;
; CHECK-LABEL: @while3(

![[alias_scope_md_for_store]] = !{![[buffer_idx_0:.*]]}
![[buffer_idx_0]] = !{!"buffer: {index:0, offset:0, size:4}", ![[aa_md_root:.*]]}
![[aa_md_root]] = !{!"XLA global AA domain"}
![[buffer_idx_1:.*]] = !{!"buffer: {index:1, offset:0, size:4}", !3}
![[buffer_idx_1_offset_16:.*]] = !{!"buffer: {index:1, offset:16, size:1}", !3}
![[noalias_md_for_load]] = !{![[buffer_idx_1_offset_16]], ![[buffer_idx_1]]}
}
)");
}

}  // namespace
}  // namespace cpu
}  // namespace xla
