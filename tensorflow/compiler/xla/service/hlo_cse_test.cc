/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_cse.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
namespace m = xla::match;

class HloCseTest : public HloTestBase {
 protected:
  HloCseTest() {}
};

TEST_F(HloCseTest, CombineTwoConstants) {
  // Test that two identical constants are commoned.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
  HloInstruction* constant = *computation->instructions().begin();
  EXPECT_EQ(42.0f, constant->literal().Get<float>({}));

  auto result = ExecuteAndTransfer(module->Clone(), {});
  auto expected = LiteralUtil::CreateR0<float>(84.0);
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec(1e-4)));
}

TEST_F(HloCseTest, CombineTwoConstantsDifferentLayouts) {
  // Test that two identical constants with different layouts are *not*
  // combined.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<float>(
          {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout({0, 1}))));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<float>(
          {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout({1, 0}))));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  auto result = ExecuteAndTransfer(module->Clone(), {});
  auto expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec(1e-4)));
}

TEST_F(HloCseTest, ConstantsSameValueDifferentType) {
  // Test that constants with the same value but different type are *not*
  // commoned.
  auto builder = HloComputation::Builder(TestName());
  std::vector<HloInstruction*> constants;
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(42))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint64_t>(42.0))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(42.0))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<double>(42.0))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f))));
  // Duplicate the float constant to verify something happens.
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f))));

  const Shape shape_r0 = ShapeUtil::MakeShape(F32, {});
  for (int64_t i = 0; i < constants.size(); ++i) {
    constants[i] = builder.AddInstruction(
        HloInstruction::CreateConvert(shape_r0, constants[i]));
  }
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_r0, HloOpcode::kAdd, constants[0], constants[1]));
  for (int64_t i = 2; i < constants.size(); ++i) {
    root = builder.AddInstruction(HloInstruction::CreateBinary(
        shape_r0, HloOpcode::kAdd, root, constants[i]));
  }

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(20, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  // CSE will remove both the second float(42.0f) and the corresponding
  // convert/cast.
  EXPECT_EQ(18, computation->instruction_count());
}

TEST_F(HloCseTest, NonscalarConstants) {
  // Test that identical nonscalar constants are merged.
  auto builder = HloComputation::Builder(TestName());
  auto common_constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto common_constant2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  // Create a constant which has the same shape but a different value.
  auto uncommon_constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}})));

  // Tie the constants together with a tuple. This makes it easier to refer to
  // the constant instructions via their use.
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple(
      {common_constant1, common_constant2, uncommon_constant}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple,
              op::Tuple(common_constant1, common_constant2, uncommon_constant));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand,
              ::testing::AnyOf(common_constant1, common_constant2));
  EXPECT_THAT(tuple,
              op::Tuple(first_operand, first_operand, uncommon_constant));
}

TEST_F(HloCseTest, IdenticalInstructions) {
  // Test that three identical instructions are commoned.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  auto exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  auto exp2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  auto exp3 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({exp1, exp2, exp3}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2, exp3));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(exp1, exp2, exp3));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand, first_operand));
}

// Test two identical while loops with same inputs
TEST_F(HloCseTest, WhileLoopsIdenticalConditionsAndBodiesSameInput) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalConditionsAndBodiesSameInput

    %body (param: (f32[], f32[])) -> (f32[], f32[]) {
      %param = (f32[], f32[]) parameter(0)
      %gte0 = get-tuple-element(%param), index=0
      %gte1 = get-tuple-element(%param), index=1
      %add = add(%gte0, %gte1)
      ROOT %tuple = tuple(%gte0, %add)
    }

    %condition {
      %param.1 = (f32[], f32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 {
      %param.2 = (f32[], f32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(false)
    }

    ENTRY %WhileLoopsIdenticalConditionsAndBodiesSameInput {
      %c0 = f32[] constant(1)
      %c1 = f32[] constant(2)
      %t = tuple(c0, c1)
      %while = while(%t), condition=%condition, body=%body
      %while.1 = while(%t), condition=%condition.1, body=%body
      ROOT r = tuple(while, while.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(6, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_TRUE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

// Test two while loops with same conditions, same inputs, but different
// bodies
TEST_F(HloCseTest, WhileLoopsIdenticalConditionsSameInputAndDifferentBodies) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalConditionsSameInputAndDifferentBodies

    %body {
      %param = (f32[], f32[]) parameter(0)
      %get-tuple-element = get-tuple-element(%param), index=0
      %get-tuple-element.1 = get-tuple-element(%param), index=1
      %add = add(%get-tuple-element, %get-tuple-element.1)
      ROOT %tuple = tuple(%get-tuple-element, %add)
    }

    %body2 {
      %param.1 = (f32[], f32[]) parameter(0)
      %get-tuple-element.2 = get-tuple-element(%param.1), index=0
      %get-tuple-element.3 = get-tuple-element(%param.1), index=1
      %sub = subtract(%get-tuple-element.2, %get-tuple-element.3)
      ROOT %tuple.2 = tuple(%get-tuple-element.2, %sub)
    }

    %condition (param.2: (f32[], f32[])) -> pred[] {
      %param.2 = (f32[], f32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 (param.3: (f32[], f32[])) -> pred[] {
      %param.3 = (f32[], f32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(false)
    }

    ENTRY %WhileLoopsIdenticalConditionsSameInputAndDifferentBodies {
      %constant.2 = f32[] constant(1)
      %constant.3 = f32[] constant(2)
      %tuple.1 = tuple(f32[] %constant.2, f32[] %constant.3)
      %while = while(%tuple.1), condition=%condition, body=%body
      ROOT %while.1 = while(%tuple.1), condition=%condition.1, body=%body2
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(5, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_FALSE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

// Test two identical while loops with different inputs
TEST_F(HloCseTest, WhileLoopsIdenticalConditionsAndBodiesDifferentInput) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalConditionsAndBodiesDifferentInput

    %body {
      %param = (f32[], f32[]) parameter(0)
      %get-tuple-element = get-tuple-element(%param), index=0
      %get-tuple-element.1 = get-tuple-element(%param), index=1
      %add = add(%get-tuple-element, %get-tuple-element.1)
      ROOT %tuple = tuple(%get-tuple-element, %add)
    }

    %body.1 {
      %param.1 = (f32[], f32[]) parameter(0)
      %gte = get-tuple-element(%param.1), index=0
      %gte1 = get-tuple-element(%param.1), index=1
      %add.1 = add(%gte, %gte1)
      ROOT %tuple = tuple(%gte, %add.1)
    }

    %condition {
      %param.1 = (f32[], f32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 {
      %param.2 = (f32[], f32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(false)
    }

    ENTRY %WhileLoopsIdenticalConditionsAndBodiesDifferentInput {
      %constant.2 = f32[] constant(1)
      %constant.3 = f32[] constant(2)
      %tuple.1 =  tuple(%constant.2, %constant.3)
      %while = while(%tuple.1), condition=%condition, body=%body
      %constant.4 = f32[] constant(1)
      %constant.5 = f32[] constant(3)
      %tuple.2 = tuple(%constant.4, %constant.5)
      ROOT %while.1 = while(%tuple.2), condition=%condition.1, body=%body.1
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(8, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_FALSE(cse.Run(m.get()).value());
  EXPECT_EQ(8, computation->instruction_count());
}

// Test two while loops with identical bodies and same inputs, but different
// conditions
TEST_F(HloCseTest, WhileLoopsIdenticalBodiesAndInputDifferentConditions) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalBodiesAndInputDifferentConditions

    %body {
      %param = (f32[], f32[]) parameter(0)
      %get-tuple-element = get-tuple-element(%param), index=0
      %get-tuple-element.1 = get-tuple-element((f32[], f32[]) %param), index=1
      %add = add(%get-tuple-element, %get-tuple-element.1)
      ROOT %tuple = tuple(%get-tuple-element, %add)
    }

    %condition {
      %param.1 = (f32[], f32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 {
      %param.2 = (f32[], f32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(true)
    }

    ENTRY %WhileLoopsIdenticalBodiesAndInputDifferentConditions {
      %constant.2 = f32[] constant(1)
      %constant.3 = f32[] constant(2)
      %tuple.1 = tuple(%constant.2, %constant.3)
      %while = while(%tuple.1), condition=%condition, body=%body
      ROOT %while.1 = while(%tuple.1), condition=%condition.1, body=%body
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(5, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_FALSE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

TEST_F(HloCseTest, IdenticalInstructionsDifferentLayoutsSensitive) {
  // Test that two identical instructions with different layouts are *not*
  // commoned if the pass is layout sensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));

  auto exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  *exp1->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  auto exp2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  *exp2->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({exp1, exp2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).value());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2));
}

TEST_F(HloCseTest, IdenticalInstructionsDifferentLayoutsInsensitive) {
  // Test that two identical instructions with different layouts are commoned if
  // the pass is layout insensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));

  auto exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  *exp1->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  auto exp2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  *exp2->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({exp1, exp2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(exp1, exp2));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand));
}

TEST_F(HloCseTest, FusionInternalCSE) {
  // Test that we can CSE expressions that live within a fusion node
  // computation.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  const Shape shape_r0 = ShapeUtil::MakeShape(F32, {});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape_r0, "p0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape_r0, "p1"));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kAdd, param0, param1));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kAdd, param0, param1));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kMultiply, add1, add2));

  auto computation = module->AddEntryComputation(builder.Build());
  auto fused_computation =
      computation
          ->CreateFusionInstruction({mul, add1, add2},
                                    HloInstruction::FusionKind::kLoop)
          ->fused_instructions_computation();

  EXPECT_EQ(5, fused_computation->instruction_count());
  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());
  EXPECT_EQ(4, fused_computation->instruction_count());

  auto root = fused_computation->root_instruction();
  EXPECT_THAT(root, op::Multiply(root->operand(0), root->operand(0)));
}

TEST_F(HloCseTest, IdenticalExpressions) {
  // Test that two identical expressions are commoned. Build the following
  // computation:
  //
  //   constant = 42.0
  //   negate1 = neg(constant)
  //   exp1 = exp(constant)
  //   add1 = add(negate1, exp1)
  //   negate2 = neg(constant)
  //   exp2 = exp(constant)
  //   add2 = add(negate2, exp2)
  //   tuple = tuple(add1, add2)
  //
  // The *1 instructions should be merged with the *2 instructions.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

  auto negate1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  auto exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, negate1, exp1));

  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  auto exp2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kExp, constant));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, negate2, exp2));

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(8, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(op::Add(negate1, exp1), op::Add(negate2, exp2)));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
  auto operand = tuple->operand(0);
  EXPECT_THAT(tuple, op::Tuple(operand, operand));
  EXPECT_THAT(operand, op::Add(op::Negate(), op::Exp()));
}

TEST_F(HloCseTest, DoNotCombineRng) {
  // Test that two RNG ops are not commoned.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto rng1 = builder.AddInstruction(HloInstruction::CreateRng(
      ShapeUtil::MakeShape(F32, {}), RandomDistribution::RNG_UNIFORM,
      {constant1, constant2}));
  auto rng2 = builder.AddInstruction(HloInstruction::CreateRng(
      ShapeUtil::MakeShape(F32, {}), RandomDistribution::RNG_UNIFORM,
      {constant1, constant2}));

  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, rng1, rng2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(rng1, rng2));

  uint32_t count_before = computation->instruction_count();

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_FALSE(cse.Run(module.get()).value());

  uint32_t count_after = computation->instruction_count();
  EXPECT_EQ(count_before, count_after);
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(rng1, rng2));
}

TEST_F(HloCseTest, DoNotCombineCallsToImpureFunctions) {
  // Test that two calls to an impure function are not commoned. RNG
  // is the source of the impurity.

  auto module = CreateNewVerifiedModule();

  // rng_function is an impure function because it does RNG.
  HloComputation* rng_function = nullptr;
  {
    Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto builder = HloComputation::Builder(TestName() + "_rng_fun");
    auto constant1 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
    auto constant2 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
    auto rng = builder.AddInstruction(HloInstruction::CreateRng(
        scalar_shape, RandomDistribution::RNG_UNIFORM, {constant1, constant2}));
    auto param = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "param"));
    builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, rng, param));
    rng_function = module->AddEmbeddedComputation(builder.Build());
  }

  // Computation calls rng_function twice with the same parameter.
  HloComputation* computation = nullptr;
  {
    auto builder = HloComputation::Builder(TestName());
    auto constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({5.0f})));
    auto rng1 = builder.AddInstruction(
        HloInstruction::CreateMap(constant->shape(), {constant}, rng_function));
    auto rng2 = builder.AddInstruction(
        HloInstruction::CreateMap(constant->shape(), {constant}, rng_function));
    builder.AddInstruction(HloInstruction::CreateBinary(
        constant->shape(), HloOpcode::kAdd, rng1, rng2));
    computation = module->AddEntryComputation(builder.Build());
  }

  EXPECT_EQ(4, computation->instruction_count());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(op::Map(), op::Map()));

  VLOG(3) << "before: " << module->ToString();

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_FALSE(cse.Run(module.get()).value());

  VLOG(3) << "after: " << module->ToString();

  EXPECT_EQ(4, computation->instruction_count());
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(op::Map(op::Constant()), op::Map(op::Constant())));
}

TEST_F(HloCseTest, CompareComputations) {
  const char* const hlo_string = R"(
    HloModule m

    add_computation {
      add_lhs = f32[] parameter(0)
      add_rhs = f32[] parameter(1)
      ROOT add_root = add(add_lhs, add_rhs)
    }

    add_computation2 {
      add_lhs2 = f32[] parameter(0)
      add_rhs2 = f32[] parameter(1)
      ROOT add_root2 = add(add_lhs2, add_rhs2)
    }

    ENTRY entry {
      p = f32[10]{0} parameter(0)
      c = f32[] constant(0)
      r1 = reduce(p, c), dimensions={0}, to_apply=add_computation
      r2 = reduce(p, c), dimensions={0}, to_apply=add_computation2
      ROOT f2 = tuple(r1, r2)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1));
}

TEST_F(HloCseTest, ConstantsSameValueInDifferentDomains) {
  // Test that constants and iotas with the same value but in different domains
  // (disjoint in this case) are not collapsed.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(42)));
  builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(S32, {42}), 0));
  builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(S32, {42}), 0));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_FALSE(cse.Run(module.get()).value());

  EXPECT_EQ(4, computation->instruction_count());
}

TEST_F(HloCseTest, Domain) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param = f32[] parameter(0), sharding={maximal device=0}
  %domain.0 = f32[] domain(%param),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  %domain.1 = f32[] domain(%param),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  %domain.2 = f32[] domain(%param),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=2}}
  %negate.0 = f32[] negate(%domain.0)
  %negate.1 = f32[] negate(%domain.1)
  %negate.2 = f32[] negate(%domain.2)
  %domain.3 = f32[] domain(%negate.0),
    domain={kind="sharding", entry={maximal device=1}, exit={maximal device=0}}
  %domain.4 = f32[] domain(%negate.1),
    domain={kind="sharding", entry={maximal device=1}, exit={maximal device=0}}
  %domain.5 = f32[] domain(%negate.2),
    domain={kind="sharding", entry={maximal device=2}, exit={maximal device=0}}
  %add = f32[] add(%domain.3, %domain.4)
  ROOT %sub = f32[] subtract(%add, %domain.5)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(m.get()).value());
  const HloInstruction* sub = m->entry_computation()->root_instruction();
  const HloInstruction* add = sub->operand(0);
  EXPECT_EQ(add->operand(0), add->operand(1));
  EXPECT_NE(add->operand(0), sub->operand(1));
  EXPECT_NE(add->operand(1), sub->operand(1));
}

TEST_F(HloCseTest, Iota) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      i1 = s64[16,16] iota(), iota_dimension=0
      i2 = s64[16,16] iota(), iota_dimension=0
      i3 = s64[17,16] iota(), iota_dimension=0
      i4 = s64[16,16] iota(), iota_dimension=1
      ROOT root = (s64[16,16], s64[16,16], s64[17,16], s64[16,16]) tuple(i1, i2, i3, i4)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));
  EXPECT_TRUE(changed);
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_NE(root->operand(0), root->operand(2));
  EXPECT_NE(root->operand(0), root->operand(3));
}

TEST_F(HloCseTest, OptimizationBarrier) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      %param.0 = f32[] parameter(0)
      %param.1 = f32[] parameter(1)
      %add.0 = f32[] add(%param.0, %param.1)
      %cse_tmp.0 = (f32[], f32[], f32[]) tuple(%param.0, %param.1, %add.0)
      %cse_tmp.1 = (f32[], f32[], f32[]) opt-barrier(%cse_tmp.0)

      %param.0.1 = f32[] get-tuple-element(%cse_tmp.1), index=0
      %param.1.1 = f32[] get-tuple-element(%cse_tmp.1), index=1
      %add.0.1 = f32[] get-tuple-element(%cse_tmp.1), index=2

      %add.1 = f32[] add(%param.0.1, %param.1.1)
      ROOT %add.2 = f32[] add(%add.1, %add.0.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));
  EXPECT_FALSE(changed);
}

class HloCseCustomCallTest
    : public HloCseTest,
      public ::testing::WithParamInterface<std::tuple<
          std::string /*op1*/, std::string /*op2*/, bool /*should_cse*/>> {};

TEST_P(HloCseCustomCallTest, DoIt) {
  std::string op1 = std::get<0>(GetParam());
  std::string op2 = std::get<1>(GetParam());
  bool should_cse = std::get<2>(GetParam());

  const char* const hlo_string_tmpl = R"(
    HloModule m
    ENTRY entry {
      p0 = f32[1,1,1] parameter(0)

      op0 = $0
      op1 = $0
      op2 = $1
      ROOT root = tuple(op0, op1, op2)
    }
  )";
  std::string hlo_string = absl::Substitute(hlo_string_tmpl, op1, op2);
  SCOPED_TRACE(absl::StrCat("Module before CSE:\n", hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));

  SCOPED_TRACE(absl::StrCat("Module after CSE:\n", m->ToString()));
  EXPECT_EQ(changed, true);  // we always CSE op0 and op1, which are identical.
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1))
      << "Identical ops should be CSE'ed";
  if (should_cse) {
    EXPECT_EQ(root->operand(0), root->operand(2)) << "Ops should be CSE'ed";
  } else {
    EXPECT_NE(root->operand(0), root->operand(2)) << "Ops should not be CSE'ed";
  }
}

static std::vector<
    std::tuple<std::string /*op1*/, std::string /*op2*/, bool /*should_cse*/>>
CustomCallTests() {
  auto build = [](absl::string_view args1, absl::string_view args2) {
    absl::string_view prefix =
        "f32[] custom-call(p0), custom_call_target=\"foo\", ";
    return std::make_tuple(absl::StrCat(prefix, args1),
                           absl::StrCat(prefix, args2), false);
  };
  return {
      {
          // metadata shouldn't prevent CSE
          "f32[] custom-call(p0), custom_call_target=\"foo\"",
          "f32[] custom-call(p0), custom_call_target=\"foo\", "
          "metadata={op_name=\"bar\"}",
          true,
      },
      {
          "f32[] custom-call(p0), custom_call_target=\"foo\"",
          "f32[] custom-call(p0, p0), custom_call_target=\"foo\"",
          false,
      },
      {
          "f32[1] custom-call(p0), custom_call_target=\"foo\"",
          "f32[2] custom-call(p0), custom_call_target=\"foo\"",
          false,
      },
      {
          "f32[] custom-call(p0), custom_call_target=\"foo\"",
          "f32[] custom-call(p0), custom_call_target=\"bar\"",
          false,
      },

      build("window={size=1}", "window={size=2}"),
      build("dim_labels=b0f_0oi->b0f", "dim_labels=b0f_0oi->bf0"),
      build("backend_config=\"foo\"", "backend_config=\"bar\""),
      build("literal=s32[] 0", "literal=s32[] 1"),
      build("literal=s32[] 0", "literal=f32[] 0"),
      build("operand_precision={high,default}",
            "operand_precision={high, high}"),
      build("api_version=API_VERSION_STATUS_RETURNING",
            "api_version=API_VERSION_ORIGINAL"),
      build("feature_group_count=0", "feature_group_count=1"),
  };
}

INSTANTIATE_TEST_SUITE_P(HloCseCustomCallTestSuite, HloCseCustomCallTest,
                         ::testing::ValuesIn(CustomCallTests()));

TEST_F(HloCseTest, CustomCallCalledComputations) {
  const char* const hlo_string = R"(
    HloModule m

    comp {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT maximum = f32[] maximum(lhs, rhs)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)

      op0 = f32[] custom-call(p0), custom_call_target="foo", called_computations={comp}
      op1 = f32[] custom-call(p0), custom_call_target="foo", called_computations={comp, comp}
      ROOT root = tuple(op0, op1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));

  SCOPED_TRACE(absl::StrCat("Module after CSE:\n", m->ToString()));
  EXPECT_EQ(changed, false);
}

TEST_F(HloCseTest, CustomCallSideEffects) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      p0 = f32[] parameter(0)

      op0 = f32[] custom-call(p0), custom_call_target="foo", custom_call_has_side_effect=true
      op1 = f32[] custom-call(p0), custom_call_target="foo", custom_call_has_side_effect=true
      ROOT root = tuple(op0, op1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));

  SCOPED_TRACE(absl::StrCat("Module after CSE:\n", m->ToString()));
  EXPECT_EQ(changed, false);
}

class HloCseCommutativeOpTest
    : public HloCseTest,
      public ::testing::WithParamInterface<std::string /*op*/> {};

TEST_P(HloCseCommutativeOpTest, DoIt) {
  std::string op = GetParam();
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      op1 = s32[10] $0(p0, p1)
      op2 = s32[10] $0(p1, p0)
      ROOT t = tuple(op1, op2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kModuleStr, op)));
  ASSERT_TRUE(HloCSE(/*is_layout_sensitive=*/false).Run(module.get()).value());
  SCOPED_TRACE(module->ToString());

  const HloInstruction* op0;
  const HloInstruction* op1;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Op(&op0), m::Op(&op1))));
  EXPECT_EQ(op0, op1);
}

INSTANTIATE_TEST_SUITE_P(AlgebraicSimplifierCanonicalizeCommutativeTestSuite,
                         HloCseCommutativeOpTest,
                         ::testing::Values("add", "multiply", "and", "or",
                                           "xor", "minimum", "maximum"));

}  // namespace
}  // namespace xla
