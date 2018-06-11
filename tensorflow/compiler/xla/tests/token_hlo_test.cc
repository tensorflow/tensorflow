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

// TODO(b/79770375): Compile, not just verify the HLO module when the backends
// support kGenerateToken.
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

}  // namespace
}  // namespace xla
