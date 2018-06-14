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
  builder.AddInstruction(HloInstruction::CreateGenerateToken({}));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));

  module->AddEntryComputation(builder.Build());
  EXPECT_IS_OK(HloVerifier().Run(module.get()).status());
}

XLA_TEST_F(TokenHloTest, TokenTree) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto token0 = builder.AddInstruction(HloInstruction::CreateGenerateToken({}));
  auto token1 = builder.AddInstruction(HloInstruction::CreateGenerateToken({}));
  auto token2 = builder.AddInstruction(HloInstruction::CreateGenerateToken({}));
  builder.AddInstruction(
      HloInstruction::CreateGenerateToken({token0, token0, token1, token2}));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));

  module->AddEntryComputation(builder.Build());
  EXPECT_IS_OK(HloVerifier().Run(module.get()).status());
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

XLA_TEST_F(TokenHloTest, InvalidTokenRoot) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(HloInstruction::CreateGenerateToken({}));
  module->AddEntryComputation(builder.Build());

  Status status = HloVerifier().Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("Entry root is or contains a token shape"));
}

XLA_TEST_F(TokenHloTest, InvalidOperandToTokenInstruction) {
  std::unique_ptr<HloModule> module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  builder.AddInstruction(HloInstruction::CreateGenerateToken({param}));
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
  // GenerateToken instruction in the while body.
  string module_string = R"(
HloModule TokenInWhileLoop

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %generate-token = token[] generate-token(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %generate-token)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] less-than(s32[] %get-tuple-element, s32[] %constant)
}

ENTRY %TokenInWhileLoop () -> s32[] {
  %zero = s32[] constant(0)
  %init_token = token[] generate-token()
  %init_tuple = (s32[], token[]) tuple(s32[] %zero, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";

  EXPECT_TRUE(RunAndCompare(module_string, error_spec_));
}

}  // namespace
}  // namespace xla
