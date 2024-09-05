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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class TokenHloTest : public HloTestBase {};

XLA_TEST_F(TokenHloTest, SingleTokenInstruction) {
  if (IsMlirLoweringEnabled()) {
    // The MLIR pipeline doesn't support returning tokens.
    GTEST_SKIP() << "Returning tokens unsupported by MLIR";
  }
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(HloInstruction::CreateToken());

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(result, LiteralUtil::CreateToken()));
}

XLA_TEST_F(TokenHloTest, TokenInTuple) {
  if (IsMlirLoweringEnabled()) {
    // The MLIR pipeline doesn't support returning tokens.
    GTEST_SKIP() << "Returning tokens unsupported by MLIR";
  }
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  builder.AddInstruction(HloInstruction::CreateTuple({token}));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  Literal token_literal = LiteralUtil::CreateToken();
  EXPECT_TRUE(
      LiteralTestUtil::Equal(result, LiteralUtil::MakeTuple({&token_literal})));
}

XLA_TEST_F(TokenHloTest, TokenTree) {
  if (IsMlirLoweringEnabled()) {
    // The MLIR pipeline doesn't support returning tokens.
    GTEST_SKIP() << "Returning tokens unsupported by MLIR";
  }
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto token0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto token1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto token2 = builder.AddInstruction(HloInstruction::CreateToken());
  builder.AddInstruction(
      HloInstruction::CreateAfterAll({token0, token0, token1, token2}));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(result, LiteralUtil::CreateToken()));
}

XLA_TEST_F(TokenHloTest, TokenInWhileLoop) {
  // Thread a token around a while loop. Token is created and consumed by a
  // AfterAll instruction in the while body.
  std::string module_string = R"(
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
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
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
  std::string module_string = R"(
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
    auto arg = LiteralUtil::CreateR0<bool>(true);
    TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&arg}));
    EXPECT_EQ(42, result.Get<int32_t>({}));
  }

  {
    // False case.
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        HloRunner::CreateModuleFromString(module_string, debug_options));
    auto arg = LiteralUtil::CreateR0<bool>(false);
    TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&arg}));
    EXPECT_EQ(7, result.Get<int32_t>({}));
  }
}

XLA_TEST_F(TokenHloTest, AddDependencyOfParameter) {
  if (IsMlirLoweringEnabled()) {
    // This test generates invalid HLO. The after-all op only takes tokens.
    GTEST_SKIP() << "Invalid HLO unsupported by MLIR";
  }
  std::string module_string = R"(
HloModule AddDependency, is_scheduled=true

// Computes (p0 + 42) * (-p1)
// where there is a dependency from the add to the negation using a token
// with after-all and add-dependency instructions.
ENTRY %AddDependency (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)

  %forty_two = f32[] constant(42.0)
  %add = f32[] add(f32[] %p0, f32[] %forty_two)
  %token0 = token[] after-all(f32[] %add)
  %p1_after_token = f32[] add-dependency(f32[] %p1, token[] %token0)
  %neg = f32[] negate(f32[] %p1_after_token)
  ROOT %product = f32[] multiply(f32[] %add, f32[] %neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));
  auto p0 = LiteralUtil::CreateR0<float>(10.0);
  auto p1 = LiteralUtil::CreateR0<float>(3.0);
  auto expected = LiteralUtil::CreateR0<float>(-156.0);
  EXPECT_EQ(expected, ExecuteNoHloPasses(std::move(module), {&p0, &p1}));
}

XLA_TEST_F(TokenHloTest, AddDependencyOfOperation) {
  if (IsMlirLoweringEnabled()) {
    // This test generates invalid HLO. The after-all op only takes tokens.
    GTEST_SKIP() << "Invalid HLO unsupported by MLIR";
  }
  std::string module_string = R"(
HloModule AddDependency, is_scheduled=true

// Computes (p0 + 42) * ( -(p1 - 45))
// where there is a dependency from the add to the negation using a token
// with after-all and add-dependency instructions.
ENTRY %AddDependency (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)

  %forty_two = f32[] constant(42.0)
  %add = f32[] add(f32[] %p0, f32[] %forty_two)
  %forty_five = f32[] constant(45.0)
  %sub = f32[] subtract(f32[] %p1, f32[] %forty_five)
  %token0 = token[] after-all(f32[] %add)
  %sub_after_token = f32[] add-dependency(f32[] %sub, token[] %token0)
  %neg = f32[] negate(f32[] %sub_after_token)
  ROOT %product = f32[] multiply(f32[] %add, f32[] %neg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));
  auto p0 = LiteralUtil::CreateR0<float>(10.0);
  auto p1 = LiteralUtil::CreateR0<float>(3.0);
  auto expected = LiteralUtil::CreateR0<float>(2184.0);
  EXPECT_EQ(expected, ExecuteNoHloPasses(std::move(module), {&p0, &p1}));
}

XLA_TEST_F(TokenHloTest, AddDependencyOfConstant) {
  if (IsMlirLoweringEnabled()) {
    // This test generates invalid HLO. The after-all op only takes tokens.
    GTEST_SKIP() << "Invalid HLO unsupported by MLIR";
  }
  std::string module_string = R"(
HloModule AddDependencyOfConstant, is_scheduled=true

ENTRY %AddDependency (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %forty_two = f32[] constant(42.0)
  %token0 = token[] after-all(f32[] %p0)
  %forty_two_after_token = f32[] add-dependency(f32[] %forty_two, token[] %token0)
  ROOT %product = f32[] multiply(f32[] %p0, f32[] %forty_two_after_token)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));
  auto p0 = LiteralUtil::CreateR0<float>(10.0);
  auto expected = LiteralUtil::CreateR0<float>(420.0);
  EXPECT_EQ(expected, ExecuteNoHloPasses(std::move(module), {&p0}));
}

XLA_TEST_F(TokenHloTest, AddDependencyAsRoot) {
  std::string module_string = R"(
HloModule AddDependencyAsRoot, is_scheduled=true
ENTRY %AddDependency (p: f32[3]) -> f32[3] {
  %p = f32[3] parameter(0)
  %neg = f32[3] negate(f32[3] %p)
  %token0 = token[] after-all()
  ROOT %add_dep = f32[3] add-dependency(f32[3] %neg, token[] %token0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));
  auto input = LiteralUtil::CreateR1<float>({1.0, 3.0, 7.0});
  auto expected = LiteralUtil::CreateR1<float>({-1.0, -3.0, -7.0});
  EXPECT_EQ(expected, ExecuteNoHloPasses(std::move(module), {&input}));
}

XLA_TEST_F(TokenHloTest, TupleShapedAddDependency) {
  if (IsMlirLoweringEnabled()) {
    // This test generates invalid MHLO. The add-dependency op doesn't take
    // tuples.
    GTEST_SKIP() << "Invalid MHLO";
  }
  std::string module_string = R"(
HloModule TupleShapedAddDependency, is_scheduled=true
ENTRY %TupleShapedAddDependency (p0: f32[3], p1: f32[3]) -> f32[3] {
  %p0 = f32[3] parameter(0)
  %p1 = f32[3] parameter(1)
  %forty_two = f32[] constant(42.0)
  %token0 = token[] after-all()
  %tuple = (f32[3], token[], f32[3], f32[]) tuple(f32[3] %p0, token[] %token0, f32[3] %p1, f32[] %forty_two)
  %add_dep = (f32[3], token[], f32[3], f32[]) add-dependency((f32[3], token[], f32[3], f32[]) %tuple, token[] %token0)
  %elem0 = f32[3] get-tuple-element((f32[3], token[], f32[3], f32[]) %add_dep), index=0
  %elem2 = f32[3] get-tuple-element((f32[3], token[], f32[3], f32[]) %add_dep), index=2
  ROOT %diff = f32[3] subtract(f32[3] %elem0, f32[3] %elem2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));
  auto p0 = LiteralUtil::CreateR1<float>({3.0, 3.0, 47.0});
  auto p1 = LiteralUtil::CreateR1<float>({1.0, -2.0, 2.0});
  auto expected = LiteralUtil::CreateR1<float>({2.0, 5.0, 45.0});
  EXPECT_EQ(expected, ExecuteNoHloPasses(std::move(module), {&p0, &p1}));
}

}  // namespace
}  // namespace xla
