/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/root_instruction_sinker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using RootInstructionSinkerTest = HloHardwareIndependentTestBase;

TEST_F(RootInstructionSinkerTest, TupleNoChange) {
  // ROOTS are already sunk, no change performed to the module.
  absl::string_view hlo_string = R"(
  HloModule While, is_scheduled=true
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto while_body =
      module->entry_computation()->root_instruction()->while_body();
  int num_body_instructions = while_body->instruction_count();
  RootInstructionSinker sinker;
  EXPECT_FALSE(sinker.Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->while_body()
                ->instruction_count(),
            num_body_instructions);
}

TEST_F(RootInstructionSinkerTest, Tuple) {
  // Sink tuple return type.
  absl::string_view hlo_string = R"(
  HloModule While, is_scheduled=true
  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
    after-all = token[] after-all()
    send = (s32[3]{0}, u32[], token[]) send(multiply, after-all), channel_id=1
    send-done = token[] send-done(send), channel_id=1
  }
  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY While {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      While.condition, body=While.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  RootInstructionSinker sinker;
  EXPECT_TRUE(sinker.Run(module.get()).value());
  auto while_body =
      module->entry_computation()->root_instruction()->while_body();
  const auto& sequence = module->schedule().sequence(while_body);
  EXPECT_EQ(sequence.instructions().at(sequence.size() - 1),
            while_body->root_instruction());
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Tuple()),
                        op::GetTupleElement(op::Tuple())));
}

TEST_F(RootInstructionSinkerTest, NontupleNoChange) {
  // ROOTS are already sunk, no change performed to the module.
  absl::string_view hlo_string = R"(
  HloModule Call, is_scheduled=true
  Call {
    param = s32[3]{0} parameter(0)
    ROOT multiply = s32[3]{0} multiply(param, param)
  }
  ENTRY While {
    constant.4 = s32[3]{0} constant({0, 1, 2})
    ROOT call = s32[3]{0} call(constant.4), to_apply=Call
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto called_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  int num_instructions = called_computation->instruction_count();
  RootInstructionSinker sinker;
  EXPECT_FALSE(sinker.Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->called_computations()[0]
                ->instruction_count(),
            num_instructions);
}

TEST_F(RootInstructionSinkerTest, Nontuple) {
  // Sink a non-tuple return type.
  absl::string_view hlo_string = R"(
  HloModule Call, is_scheduled=true
  Call {
    param = s32[3]{0} parameter(0)
    ROOT multiply = s32[3]{0} multiply(param, param)
    after-all = token[] after-all()
    send = (s32[3]{0}, u32[], token[]) send(multiply, after-all), channel_id=1
    send-done = token[] send-done(send), channel_id=1
  }
  ENTRY While {
    constant.4 = s32[3]{0} constant({0, 1, 2})
    ROOT call = s32[3]{0} call(constant.4), to_apply=Call
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  RootInstructionSinker sinker;
  EXPECT_TRUE(sinker.Run(module.get()).value());
  auto called_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const auto& sequence = module->schedule().sequence(called_computation);
  EXPECT_EQ(sequence.instructions().at(sequence.size() - 1),
            called_computation->root_instruction());
  EXPECT_THAT(called_computation->root_instruction(),
              op::Bitcast(op::Multiply()));
}

}  // namespace
}  // namespace xla
