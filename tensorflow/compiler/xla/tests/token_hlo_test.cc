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

#include <array>

#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class TokenHloTest : public HloTestBase {};

XLA_TEST_F(TokenHloTest, SingleTokenInstruction) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(HloInstruction::CreateAfterAll({}));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          Execute(std::move(module), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(*result, *Literal::CreateToken()));
}

XLA_TEST_F(TokenHloTest, TokenTree) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto token0 = builder.AddInstruction(HloInstruction::CreateAfterAll({}));
  auto token1 = builder.AddInstruction(HloInstruction::CreateAfterAll({}));
  auto token2 = builder.AddInstruction(HloInstruction::CreateAfterAll({}));
  builder.AddInstruction(
      HloInstruction::CreateAfterAll({token0, token0, token1, token2}));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          Execute(std::move(module), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(*result, *Literal::CreateToken()));
}

XLA_TEST_F(TokenHloTest, InvalidTokenShapedEntryParameter) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeTokenShape(), "p1"));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));
  module->AddEntryComputation(builder.Build());

  Status status = HloVerifier().Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Entry parameter 1 is or contains a token shape"));
}

XLA_TEST_F(TokenHloTest, InvalidTupleTokenShapedEntryParameter) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(F32, {1, 2, 3}), ShapeUtil::MakeTokenShape()}),
      "param"));
  module->AddEntryComputation(builder.Build());

  Status status = HloVerifier().Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Entry parameter 0 is or contains a token shape"));
}

XLA_TEST_F(TokenHloTest, InvalidOperandToTokenInstruction) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  builder.AddInstruction(HloInstruction::CreateAfterAll({param}));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(123)));
  module->AddEntryComputation(builder.Build());

  Status status = HloVerifier().Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "Operands of token instructions must be TOKEN types"));
}

XLA_TEST_F(TokenHloTest, TokenInWhileLoop) {
  // Thread a token around a while loop. Token is created and consumed by a
  // AfterAll instruction in the while body.
  string module_string = R"(
HloModule TokenInWhileLoop

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] less-than(s32[] %get-tuple-element, s32[] %constant)
}

ENTRY %TokenInWhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  DebugOptions debug_options = GetDebugOptionsForTest();
  // Module DCE pass removes the generate token instructions.
  debug_options.add_xla_disable_hlo_passes("hlo-module-dce");
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      HloRunner::CreateModuleFromString(module_string, debug_options));

  EXPECT_TRUE(RunAndCompare(std::move(module), error_spec_));
}

XLA_TEST_F(TokenHloTest, TokenInConditional) {
  string module_string = R"(
HloModule TokenInConditional

%True (param.1: token[]) -> (s32[], token[]) {
  %param.1 = token[] parameter(0)
  %forty_two = s32[] constant(42)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %forty_two, token[] %param.1)
}

%False (param.2: s32[]) -> (s32[], token[]) {
  %param.2 = s32[] parameter(0)
  %new_token = token[] after-all()
  ROOT %tuple = (s32[], token[]) tuple(s32[] %param.2, token[] %new_token)
}

ENTRY %TokenInConditional (param.3: pred[]) -> s32[] {
  %param.3 = pred[] parameter(0)
  %init_token = token[] after-all()
  %seven = s32[] constant(7)
  %cond = (s32[], token[]) conditional(pred[] %param.3, token[] %init_token, s32[] %seven), true_computation=True, false_computation=False
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %cond), index=0
}
)";

  DebugOptions debug_options = GetDebugOptionsForTest();
  // Module DCE pass removes the generate token instructions.
  debug_options.add_xla_disable_hlo_passes("hlo-module-dce");

  {
    // True case.
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        HloRunner::CreateModuleFromString(module_string, debug_options));
    auto arg = Literal::CreateR0<bool>(true);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                            Execute(std::move(module), {arg.get()}));
    EXPECT_EQ(42, result->Get<int32>({}));
  }

  {
    // False case.
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        HloRunner::CreateModuleFromString(module_string, debug_options));
    auto arg = Literal::CreateR0<bool>(false);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                            Execute(std::move(module), {arg.get()}));
    EXPECT_EQ(7, result->Get<int32>({}));
  }
}

}  // namespace
}  // namespace xla
