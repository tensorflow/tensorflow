/* Copyright 2018 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class AliasAnalysisTest : public CpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    // We do not generate IR for while loops with thunks runtime, so we
    // explicitly disable it for this test.
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    return debug_options;
  }
};

static absl::Status FakeCustomCallTarget(ffi::AnyBuffer,
                                         ffi::Result<ffi::AnyBuffer>) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFakeCustomCallTarget, FakeCustomCallTarget,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()  // in
                           .Ret<ffi::AnyBuffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$FakeCustomCallTarget", "Host",
                         kFakeCustomCallTarget);

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
  addend = f32[] custom-call(condition.state), custom_call_target="__xla_test$$FakeCustomCallTarget", api_version=API_VERSION_TYPED_FFI
  add = f32[] add(addend, condition.state)
  ROOT greater-than = pred[] compare(const.100, add), direction=GT
}

ENTRY while3 {
  const.0 = f32[] constant(0)
  ROOT while = f32[] while(const.0), condition=condition, body=body
}
)";

  CompileAndVerifyIr(hlo_string, R"(
; CHECK-LABEL: @body(ptr %retval
; CHECK: %[[add_result:.*]] = fadd float %[[fadd_lhs:.*]], %[[fadd_rhs:.*]]
; CHECK: store float %[[add_result]], ptr %[[store_dest:.*]], align 4, !alias.scope ![[alias_scope_md_for_store:[0-9]+]]
;
; CHECK-LABEL: @condition(ptr %retval, ptr noalias %run_options, ptr noalias %params
; CHECK: %[[cond_state_buf_ptr:.*]] = getelementptr inbounds ptr, ptr %buffer_table, i64 0
; CHECK: %[[cond_state_buf_untyped:.*]] = load ptr, ptr %[[cond_state_buf_ptr]]
; CHECK: load float, ptr %[[cond_state_buf_untyped]], align 4, !alias.scope ![[alias_scope_md_for_store]], !noalias ![[noalias_md_for_load:.*]]
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
}  // namespace xla::cpu
