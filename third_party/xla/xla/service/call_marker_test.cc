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

#include "xla/service/call_marker.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/call_inliner.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using CallMarkerTest = HloHardwareIndependentTestBase;

TEST_F(CallMarkerTest, MarkSingleCall) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    a = f32[] call(c), to_apply=a
    ROOT tuple = (f32[], f32[]) tuple(a, c)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  /* Changed module will lool like this:

  ENTRY %inline () -> (f32[], f32[]) {
    %c = f32[] constant(1)
    %custom-call = (f32[]) custom-call(%c), custom_call_target="a_before"
    %get-tuple-element = f32[] get-tuple-element(%custom-call), index=0
    %a = f32[] call(%get-tuple-element), to_apply=%a
    %custom-call.1 = f32[] custom-call(%a), custom_call_target="a_after"
    ROOT %tuple = (f32[], f32[]) tuple(%custom-call.1, %c)
  }
  */
  HloInstruction* custom_call_before =
      FindInstruction(module.get(), "custom-call");
  HloInstruction* gte = FindInstruction(module.get(), "get-tuple-element");
  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* custom_call_after =
      FindInstruction(module.get(), "custom-call.1");

  EXPECT_TRUE(nullptr != custom_call_before);
  EXPECT_TRUE(nullptr != gte);
  EXPECT_TRUE(nullptr != a);
  EXPECT_TRUE(nullptr != custom_call_after);

  EXPECT_EQ(custom_call_after->operand(0), a);
  EXPECT_EQ(a->operand(0), gte);
  EXPECT_EQ(gte->operand(0), custom_call_before);
}

TEST_F(CallMarkerTest, MarkRootCall) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    ROOT a = f32[] call(c), to_apply=a
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  /* Changed module will lool like this:
  ENTRY %inline () -> f32[] {
    %c = f32[] constant(1)
    %custom-call = (f32[]) custom-call(%c), custom_call_target=\"a_before\"
    %get-tuple-element = f32[] get-tuple-element(%custom-call), index=0
    %a = f32[] call(%get-tuple-element), to_apply=%a
    ROOT %custom-call.1 = f32[] custom-call(%a), custom_call_target=\"a_after\"
  }
  */
  HloInstruction* custom_call_before =
      FindInstruction(module.get(), "custom-call");
  HloInstruction* gte = FindInstruction(module.get(), "get-tuple-element");
  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* custom_call_after =
      FindInstruction(module.get(), "custom-call.1");

  EXPECT_TRUE(nullptr != custom_call_before);
  EXPECT_TRUE(nullptr != gte);
  EXPECT_TRUE(nullptr != a);
  EXPECT_TRUE(nullptr != custom_call_after);

  EXPECT_EQ(module->entry_computation()->root_instruction(), custom_call_after);
  EXPECT_EQ(custom_call_after->operand(0), a);
  EXPECT_EQ(a->operand(0), gte);
  EXPECT_EQ(gte->operand(0), custom_call_before);
}

TEST_F(CallMarkerTest, MarkNestedCalls) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  c {
    p = f32[] parameter(0)
    ROOT result = f32[] add(p, p)
  }

  b {
    p = f32[] parameter(0)
    ROOT result = f32[] call(p), to_apply=c
  }

  a {
    p = f32[] parameter(0)
    ROOT result = f32[] call(p), to_apply=b
  }

  ENTRY inline {
    c = f32[] constant(1)
    ROOT result = f32[] call(c), to_apply=a
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);
  HloInstruction* call_before_c = FindInstruction(module.get(), "custom-call");
  EXPECT_TRUE(call_before_c != nullptr);
  EXPECT_EQ(call_before_c->custom_call_target(), kCallMarkerBeforeTarget);
  EXPECT_TRUE(call_before_c->has_frontend_attributes());
  EXPECT_EQ(call_before_c->frontend_attributes().map().at(
                kCallMarkedComputationAttribute),
            "c");

  HloInstruction* call_after_a = FindInstruction(module.get(), "custom-call.5");
  EXPECT_TRUE(call_after_a != nullptr);
  EXPECT_EQ(call_after_a->custom_call_target(), kCallMarkerAfterTarget);
  EXPECT_TRUE(call_after_a->has_frontend_attributes());
  EXPECT_EQ(call_after_a->frontend_attributes().map().at(
                kCallMarkedComputationAttribute),
            "a");
}
}  // namespace
}  // namespace xla
