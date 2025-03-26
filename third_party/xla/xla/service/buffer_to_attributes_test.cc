/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/buffer_to_attributes.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class BuffersToAttributesTest : public HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseAndVerifyBuffers(
      absl::string_view hlo_string) {
    auto module = ParseAndReturnUnverifiedModule(hlo_string);
    if (!module.ok()) {
      return module.status();
    }
    auto status = HloVerifier{HloVerifierOpts{}.VerifyBuffers()}
                      .Run(module.value().get())
                      .status();
    if (!status.ok()) {
      return status;
    }
    return std::move(module.value());
  }
  void DoFileCheck(HloModule* module, absl::string_view hlo_string) {
    HloPrintOptions options;
    options.set_include_layout_in_shapes(false);
    VLOG(2) << module->ToString(options);
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, RunFileCheck(module->ToString(options), hlo_string));
    EXPECT_TRUE(result);
  }
};

TEST_F(BuffersToAttributesTest, TrivialPinUnpin) {
  const char* const hlo = R"(
  HloModule module

  ENTRY main {
    p0 = f32[32] parameter(0)
    b0 = f32[32]{buffer_id=1} custom-call(p0), custom_call_target="pin"
    b1 = f32[32] custom-call(b0), custom_call_target="unpin"
    ROOT a = f32[32] add(b1, p0)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK: ENTRY %main (p0: f32[32]) -> f32[32] {
    // CHECK:   %p0 = f32[32] parameter(0)
    // CHECK:   ROOT %a = f32[32] add(%p0, %p0)
  )");
}

TEST_F(BuffersToAttributesTest, SimpleChain) {
  const char* const hlo = R"(
  HloModule module

  ENTRY main {
    p0 = f32[32] parameter(0)
    b0 = f32[32]{buffer_id=1} custom-call(), custom_call_target="allocateBuffer"
    async = (f32[32]{buffer_id=1}, u32[], token[])
      custom-call(b0, p0), custom_call_target="bar"
    a = f32[32] add(p0, p0)
    b1 = f32[32]{buffer_id=1} get-tuple-element(async), index=0
    b2 = f32[32]{buffer_id=1} custom-call(b1), custom_call_target="foo"
    v = f32[32] custom-call(b2), custom_call_target="unpin"
    ROOT c = f32[32] add(v, a)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK: ENTRY %main (p0: f32[32]) -> f32[32] {
    // CHECK-DAG: %p0 = f32[32] parameter(0)
    // CHECK-DAG: %b0 = f32[32] custom-call(), custom_call_target="allocateBuffer"
    // CHECK{LITERAL}: %async = (f32[32], u32[], token[]) custom-call(%b0, %p0), custom_call_target="bar", output_to_operand_aliasing={{0}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %b1 = f32[32] get-tuple-element(%async), index=0
    // CHECK{LITERAL}: %b2 = f32[32] custom-call(%b1), custom_call_target="foo", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %a = f32[32] add(%p0, %p0)
    // CHECK: ROOT %c = f32[32] add(%b2, %a)
  )");
}

TEST_F(BuffersToAttributesTest, ChainCompletelyInsideWhileLoop) {
  const char* const hlo = R"(
  HloModule module

  while_body {
    param = (f32[16], f32[16]) parameter(0)

    p0-w0 = f32[16] get-tuple-element(param), index=0
    p1-w0 = f32[16] get-tuple-element(param), index=1
    p0-w1 = f32[16]{buffer_id=1} custom-call(p0-w0), custom_call_target="pin"
    p0-w2 = f32[16]{buffer_id=1} custom-call(p0-w1), custom_call_target="update0"
    p0-w3 = f32[16]{buffer_id=1} custom-call(p0-w2), custom_call_target="update1"
    p0-w4 = f32[16] custom-call(p0-w3), custom_call_target="unpin"
    p1-w1 = f32[16]add(p1-w0, p1-w0)
    ROOT tuple = (f32[16], f32[16]) tuple(p0-w4, p1-w1)
  }

  // Infinite loop to keep IR small.
  while_condition {
    param = (f32[16], f32[16]) parameter(0)
    ROOT infinite_loop = pred[] constant(true)
  }

  ENTRY main {
    p0 = f32[16] parameter(0)
    init = (f32[16], f32[16]) tuple(p0, p0)
    while = (f32[16], f32[16]) while(init), condition=while_condition, body=while_body
    v0 = f32[16] get-tuple-element(while), index=0
    v1 = f32[16] get-tuple-element(while), index=1
    ROOT c = f32[16] add(v0, v1)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK: %while_body (param: (f32[16], f32[16])) -> (f32[16], f32[16]) {
    // CHECK: %param = (f32[16], f32[16]) parameter(0)
    // CHECK: %p0-w0 = f32[16] get-tuple-element(%param), index=0
    // CHECK{LITERAL}: %p0-w2 = f32[16] custom-call(%p0-w0), custom_call_target="update0", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %p0-w3 = f32[16] custom-call(%p0-w2), custom_call_target="update1", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %p1-w0 = f32[16] get-tuple-element(%param), index=1
    // CHECK: %p1-w1 = f32[16] add(%p1-w0, %p1-w0)
    // CHECK: ROOT %tuple = (f32[16], f32[16]) tuple(%p0-w3, %p1-w1)
    // CHECK: ENTRY %main (p0: f32[16]) -> f32[16] {
    // CHECK: %p0 = f32[16] parameter(0)
    // CHECK: %init = (f32[16], f32[16]) tuple(%p0, %p0)
    // CHECK: %while = (f32[16], f32[16]) while(%init), condition=%while_condition, body=%while_body
    // CHECK: %v0 = f32[16] get-tuple-element(%while), index=0
    // CHECK: %v1 = f32[16] get-tuple-element(%while), index=1
    // CHECK: ROOT %c = f32[16] add(%v0, %v1)
  )");
}

TEST_F(BuffersToAttributesTest, ChainPartiallyInsideWhileLoop) {
  const char* const hlo = R"(
  HloModule module

  while_body {
    param = (f32[16]{buffer_id=1}, f32[16]) parameter(0)
    p0-w0 = f32[16]{buffer_id=1} get-tuple-element(param), index=0
    p1-w0 = f32[16] get-tuple-element(param), index=1
    p0-w1 = f32[16]{buffer_id=1} custom-call(p0-w0), custom_call_target="update0"
    p0-w2 = f32[16]{buffer_id=1} custom-call(p0-w1), custom_call_target="update1"
    p1-w1 = f32[16]add(p1-w0, p1-w0)
    ROOT tuple = (f32[16]{buffer_id=1}, f32[16]) tuple(p0-w2, p1-w1)
  }

  // Infinite loop to keep IR small.
  while_condition {
    param = (f32[16]{buffer_id=1}, f32[16]) parameter(0)
    ROOT infinite_loop = pred[] constant(true)
  }

  ENTRY main {
    p0 = f32[16] parameter(0)
    b0 = f32[16]{buffer_id=1} custom-call(p0), custom_call_target="pin"
    init = (f32[16]{buffer_id=1}, f32[16]) tuple(b0, p0)
    while = (f32[16]{buffer_id=1}, f32[16]) while(init), condition=while_condition, body=while_body
    b1 = f32[16]{buffer_id=1} get-tuple-element(while), index=0
    v0 = f32[16] custom-call(b1), custom_call_target="unpin"
    v1 = f32[16] get-tuple-element(while), index=1
    ROOT c = f32[16] add(v0, v1)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK: %while_body (param: (f32[16], f32[16])) -> (f32[16], f32[16]) {
    // CHECK: %param = (f32[16], f32[16]) parameter(0)
    // CHECK: %p0-w0 = f32[16] get-tuple-element(%param), index=0
    // CHECK{LITERAL}: %p0-w1 = f32[16] custom-call(%p0-w0), custom_call_target="update0", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %p0-w2 = f32[16] custom-call(%p0-w1), custom_call_target="update1", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %p1-w0 = f32[16] get-tuple-element(%param), index=1
    // CHECK: %p1-w1 = f32[16] add(%p1-w0, %p1-w0)
    // CHECK{LITERAL}: ROOT %tuple = (f32[16], f32[16]) tuple(%p0-w2, %p1-w1), frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: ENTRY %main (p0: f32[16]) -> f32[16] {
    // CHECK: %p0 = f32[16] parameter(0)
    // CHECK{LITERAL}: %init = (f32[16], f32[16]) tuple(%p0, %p0), frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %while = (f32[16], f32[16]) while(%init), condition=%while_condition, body=%while_body, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %b1 = f32[16] get-tuple-element(%while), index=0
    // CHECK: %v1 = f32[16] get-tuple-element(%while), index=1
    // CHECK: ROOT %c = f32[16] add(%b1, %v1)
  )");
}

// To write a rotated chain of non-copyable, here we pin the same buffer_id
// twice, one inside the while loop and one outside before calling the while
// loop. Conceptually, the one outside while loop is from peeling off part of
// the first iteration of the while loop.
TEST_F(BuffersToAttributesTest, ChainRotatedWhileLoop) {
  const char* const hlo = R"(
  HloModule module

  while_body {
    param = (f32[16]{buffer_id=1}, f32[16]) parameter(0)
    p0-w0 = f32[16]{buffer_id=1} get-tuple-element(param), index=0
    p1-w0 = f32[16] get-tuple-element(param), index=1
    p0-w1 = f32[16]{buffer_id=1} custom-call(p0-w0), custom_call_target="update0"
    p0-w2 = f32[16]{buffer_id=1} custom-call(p0-w1), custom_call_target="update1"
    p0-w3 = f32[16] custom-call(p0-w2), custom_call_target="unpin"
    p1-w1 = f32[16]add(p1-w0, p0-w3)
    p0-w4 = f32[16]{buffer_id=1} custom-call(p0-w3), custom_call_target="pin"
    ROOT tuple = (f32[16]{buffer_id=1}, f32[16]) tuple(p0-w4, p1-w1)
  }

  // Infinite loop to keep IR small.
  while_condition {
    param = (f32[16]{buffer_id=1}, f32[16]) parameter(0)
    ROOT infinite_loop = pred[] constant(true)
  }

  ENTRY main {
    p0 = f32[16] parameter(0)
    b0 = f32[16]{buffer_id=1} custom-call(p0), custom_call_target="pin"
    init = (f32[16]{buffer_id=1}, f32[16]) tuple(b0, p0)
    while = (f32[16]{buffer_id=1}, f32[16]) while(init), condition=while_condition, body=while_body
    b1 = f32[16]{buffer_id=1} get-tuple-element(while), index=0
    v0 = f32[16] custom-call(b1), custom_call_target="unpin"
    v1 = f32[16] get-tuple-element(while), index=1
    ROOT c = f32[16] add(v0, v1)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK: %while_body (param: (f32[16], f32[16])) -> (f32[16], f32[16]) {
    // CHECK: %param = (f32[16], f32[16]) parameter(0)
    // CHECK: %p0-w0 = f32[16] get-tuple-element(%param), index=0
    // CHECK{LITERAL}: %p0-w1 = f32[16] custom-call(%p0-w0), custom_call_target="update0", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %p0-w2 = f32[16] custom-call(%p0-w1), custom_call_target="update1", output_to_operand_aliasing={{}: (0, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %p1-w0 = f32[16] get-tuple-element(%param), index=1
    // CHECK: %p1-w1 = f32[16] add(%p1-w0, %p0-w2)
    // CHECK{LITERAL}: ROOT %tuple = (f32[16], f32[16]) tuple(%p0-w2, %p1-w1), frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: ENTRY %main (p0: f32[16]) -> f32[16] {
    // CHECK: %p0 = f32[16] parameter(0)
    // CHECK{LITERAL}: %init = (f32[16], f32[16]) tuple(%p0, %p0), frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %while = (f32[16], f32[16]) while(%init), condition=%while_condition, body=%while_body, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %b1 = f32[16] get-tuple-element(%while), index=0
    // CHECK: %v1 = f32[16] get-tuple-element(%while), index=1
    // CHECK: ROOT %c = f32[16] add(%b1, %v1)
  )");
}

TEST_F(BuffersToAttributesTest, ChainRotatedWhileLoop2) {
  const char* const hlo = R"(
  HloModule module

  while_body {
    param = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16], f32[16]) parameter(0)
    send-buffer-2 = f32[16]{buffer_id=3} get-tuple-element(param), index=0
    recv-buffer-2 = f32[16]{buffer_id=1} get-tuple-element(param), index=1
    sema-buffer-2 = f32[2]{buffer_id=2} get-tuple-element(param), index=2
    async-done = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16])
      custom-call(send-buffer-2, recv-buffer-2, sema-buffer-2), custom_call_target="end_send_recv"
    fwd = f32[16] get-tuple-element(async-done), index=3

    prev-fwd = f32[16] get-tuple-element(param), index=4
    send-buffer-3 = f32[16]{buffer_id=3} get-tuple-element(async-done), index=0
    recv-buffer-3 = f32[16]{buffer_id=1} get-tuple-element(async-done), index=1
    sema-buffer-3 = f32[2]{buffer_id=2} get-tuple-element(async-done), index=2
    // send-buffer and recv-buffer aren't swapped in this example, we need to unroll=2 to swap buffers
    async-start = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2})
      custom-call(prev-fwd, sema-buffer-3, send-buffer-3, recv-buffer-3), custom_call_target="start_send_recv"

    send-buffer-4 = f32[16]{buffer_id=3} get-tuple-element(async-start), index=0
    recv-buffer-4 = f32[16]{buffer_id=1} get-tuple-element(async-start), index=1
    sema-buffer-4 = f32[2]{buffer_id=2} get-tuple-element(async-start), index=2
    ROOT tuple = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16], f32[16])
      tuple(send-buffer-4, recv-buffer-4, sema-buffer-4, prev-fwd, fwd)
  }

  // Infinite loop to keep IR small.
  while_condition {
    param = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16], f32[16]) parameter(0)
    ROOT infinite_loop = pred[] constant(true)
  }

  ENTRY main_spmd {
    data = f32[16] parameter(0)
    send-buffer-0 = f32[16]{buffer_id=3} custom-call(), custom_call_target="allocateBuffer"
    recv-buffer-0 = f32[16]{buffer_id=1} custom-call(), custom_call_target="allocateBuffer"
    sema-buffer-0 = f32[2]{buffer_id=2} custom-call(), custom_call_target="allocateBuffer"
    // The orderings of the three buffers are different in operands and results.
    async-start-m = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2})
      custom-call(data, sema-buffer-0, send-buffer-0, recv-buffer-0), custom_call_target="start_send_recv"

    send-buffer-1 = f32[16]{buffer_id=3} get-tuple-element(async-start-m), index=0
    recv-buffer-1 = f32[16]{buffer_id=1} get-tuple-element(async-start-m), index=1
    sema-buffer-1 = f32[2]{buffer_id=2} get-tuple-element(async-start-m), index=2
    init = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16], f32[16])
      tuple(send-buffer-1, recv-buffer-1, sema-buffer-1, data, data)
    while = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16], f32[16])
      while(init), condition=while_condition, body=while_body
    send-buffer-5 = f32[16]{buffer_id=3} get-tuple-element(while), index=0
    recv-buffer-5 = f32[16]{buffer_id=1} get-tuple-element(while), index=1
    sema-buffer-5 = f32[2]{buffer_id=2} get-tuple-element(while), index=2
    async-done-m = (f32[16]{buffer_id=3}, f32[16]{buffer_id=1}, f32[2]{buffer_id=2}, f32[16])
      custom-call(send-buffer-5, recv-buffer-5, sema-buffer-5), custom_call_target="end_send_recv"
    ROOT data-out = f32[16] get-tuple-element(async-done-m), index=3
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndVerifyBuffers(hlo));
  ConvertBufferRepresentationToAttributes pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  auto status = HloVerifier{HloVerifierOpts{}}.Run(module.get()).status();
  ASSERT_TRUE(status.ok());
  DoFileCheck(module.get(), R"(
    // CHECK-LABEL: %while_body
    // CHECK: %param = (f32[16], f32[16], f32[2], f32[16], f32[16]) parameter(0)
    // CHECK-DAG: %prev-fwd = f32[16] get-tuple-element(%param), index=4
    // CHECK-DAG: %send-buffer-2 = f32[16] get-tuple-element(%param), index=0
    // CHECK-DAG: %recv-buffer-2 = f32[16] get-tuple-element(%param), index=1
    // CHECK-DAG: %sema-buffer-2 = f32[2] get-tuple-element(%param), index=2
    // CHECK{LITERAL}: %async-done = (f32[16], f32[16], f32[2], f32[16]) custom-call(%send-buffer-2, %recv-buffer-2, %sema-buffer-2), custom_call_target="end_send_recv", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {}), {2}: (2, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %send-buffer-3 = f32[16] get-tuple-element(%async-done), index=0
    // CHECK-DAG: %recv-buffer-3 = f32[16] get-tuple-element(%async-done), index=1
    // CHECK-DAG: %sema-buffer-3 = f32[2] get-tuple-element(%async-done), index=2
    // CHECK{LITERAL}: %async-start = (f32[16], f32[16], f32[2]) custom-call(%prev-fwd, %sema-buffer-3, %send-buffer-3, %recv-buffer-3), custom_call_target="start_send_recv", output_to_operand_aliasing={{0}: (2, {}), {1}: (3, {}), {2}: (1, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %send-buffer-4 = f32[16] get-tuple-element(%async-start), index=0
    // CHECK-DAG: %recv-buffer-4 = f32[16] get-tuple-element(%async-start), index=1
    // CHECK-DAG: %sema-buffer-4 = f32[2] get-tuple-element(%async-start), index=2
    // CHECK-DAG: %fwd = f32[16] get-tuple-element(%async-done), index=3
    // CHECK{LITERAL}: ROOT %tuple = (f32[16], f32[16], f32[2], f32[16], f32[16]) tuple(%send-buffer-4, %recv-buffer-4, %sema-buffer-4, %prev-fwd, %fwd), frontend_attributes={_xla_non_copyable_attribute={}}


    // CHECK-LABEL: ENTRY %main_spmd (data: f32[16]) -> f32[16] {
    // CHECK: %data = f32[16] parameter(0)
    // CHECK-DAG: %send-buffer-0 = f32[16] custom-call(), custom_call_target="allocateBuffer", frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %recv-buffer-0 = f32[16] custom-call(), custom_call_target="allocateBuffer", frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %sema-buffer-0 = f32[2] custom-call(), custom_call_target="allocateBuffer", frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK{LITERAL}: %async-start-m = (f32[16], f32[16], f32[2]) custom-call(%data, %sema-buffer-0, %send-buffer-0, %recv-buffer-0), custom_call_target="start_send_recv", output_to_operand_aliasing={{0}: (2, {}), {1}: (3, {}), {2}: (1, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %send-buffer-1 = f32[16] get-tuple-element(%async-start-m), index=0
    // CHECK-DAG: %recv-buffer-1 = f32[16] get-tuple-element(%async-start-m), index=1
    // CHECK-DAG: %sema-buffer-1 = f32[2] get-tuple-element(%async-start-m), index=2
    // CHECK{LITERAL}: %init = (f32[16], f32[16], f32[2], f32[16], f32[16]) tuple(%send-buffer-1, %recv-buffer-1, %sema-buffer-1, %data, %data), frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: %while = (f32[16], f32[16], f32[2], f32[16], f32[16]) while(%init), condition=%while_condition, body=%while_body, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK-DAG: %send-buffer-5 = f32[16] get-tuple-element(%while), index=0
    // CHECK-DAG: %recv-buffer-5 = f32[16] get-tuple-element(%while), index=1
    // CHECK-DAG: %sema-buffer-5 = f32[2] get-tuple-element(%while), index=2
    // CHECK{LITERAL}: %async-done-m = (f32[16], f32[16], f32[2], f32[16]) custom-call(%send-buffer-5, %recv-buffer-5, %sema-buffer-5), custom_call_target="end_send_recv", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {}), {2}: (2, {})}, frontend_attributes={_xla_non_copyable_attribute={}}
    // CHECK: ROOT %data-out = f32[16] get-tuple-element(%async-done-m), index=3
  )");
}

}  // namespace
}  // namespace xla
