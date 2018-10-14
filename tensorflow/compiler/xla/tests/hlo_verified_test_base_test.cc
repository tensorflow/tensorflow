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

#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// This class includes unit tests which are expected to fail because invalid HLO
// modules are intentionally built. Unfortunately, Tensorflow doesn't appear to
// include the necessary gunit parts to test this test machinery (needs the
// macro EXPECT_NONFATAL_FAILURE). The disabled tests can be run with the
// disabled tests enabled and failures can be manually compared against
// expectations.
class HloVerifiedTestBaseTest : public HloVerifiedTestBase {};

XLA_TEST_F(HloVerifiedTestBaseTest, NoModule) {
  // Test shouldn't fail if no module is created at all.
}

XLA_TEST_F(HloVerifiedTestBaseTest, GoodLazilyCreatedModule) {
  // Use module() to lazily create an empty module, build it up, and verify no
  // failures.
  HloModule& hlo_module = module();
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateUnary(input->shape(), HloOpcode::kNegate, input));
  hlo_module.AddEntryComputation(builder.Build());
}

// This test is expected to fail. See test class comment.
XLA_TEST_F(HloVerifiedTestBaseTest, DISABLED_BadLazilyCreatedModule) {
  // Use module() to lazily create an empty module and build up an invalid
  // module.
  HloModule& hlo_module = module();
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateUnary(input->shape(), HloOpcode::kNegate, input));
  hlo_module.AddEntryComputation(builder.Build());

  *hlo_module.entry_computation()->root_instruction()->mutable_shape() =
      ShapeUtil::MakeShape(PRED, {1, 2, 3});
}

XLA_TEST_F(HloVerifiedTestBaseTest, GoodCreateNewModule) {
  // Call CreateNewModule and build up a valid module.
  HloModule* module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateUnary(input->shape(), HloOpcode::kNegate, input));
  module->AddEntryComputation(builder.Build());
}

// This test is expected to fail. See test class comment.
XLA_TEST_F(HloVerifiedTestBaseTest, DISABLED_BadCreateNewModule) {
  // Call CreateNewModule and build up a invalid module.
  HloModule* module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateUnary(input->shape(), HloOpcode::kNegate, input));
  module->AddEntryComputation(builder.Build());

  *module->entry_computation()->root_instruction()->mutable_shape() =
      ShapeUtil::MakeShape(PRED, {1, 2, 3});
}

XLA_TEST_F(HloVerifiedTestBaseTest, ParseAndVerifyModuleGood) {
  const char* const hlo_string = R"(
HloModule ParseAndVerifyModuleGood

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x,y)
}
)";

  ParseAndVerifyModule(hlo_string);
  EXPECT_EQ(module().entry_computation()->instruction_count(), 3);
}

XLA_TEST_F(HloVerifiedTestBaseTest, ParseAndReturnVerifiedModuleGood) {
  const char* const hlo_string = R"(
HloModule ParseAndReturnVerifiedModuleGood

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x,y)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
}

XLA_TEST_F(HloVerifiedTestBaseTest, ParseAndReturnVerifiedModuleInvalidText) {
  const char* const hlo_string = R"(
HloModule ParseAndReturnVerifiedModuleGood

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x,y)
}

RANDOM GARBAGE
)";

  ASSERT_IS_NOT_OK(ParseAndReturnVerifiedModule(hlo_string).status());
}

// This test is expected to fail. See test class comment.
XLA_TEST_F(HloVerifiedTestBaseTest, DISABLED_ParseAndReturnVerifiedModuleBad) {
  const char* const hlo_string = R"(
HloModule ParseAndReturnVerifiedModuleBad

ENTRY entry {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[1234] add(x,y)
}
)";

  ASSERT_IS_NOT_OK(ParseAndReturnVerifiedModule(hlo_string).status());
}

}  // namespace
}  // namespace xla
