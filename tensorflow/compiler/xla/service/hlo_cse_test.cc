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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(2, computation->instruction_count());
  HloInstruction* constant = computation->instructions().begin()->get();
  EXPECT_EQ(42.0f, LiteralUtil::Get<float>(constant->literal(), {}));

  auto result = ExecuteAndTransfer(std::move(module), {});
  auto expected = LiteralUtil::CreateR0<float>(84.0);
  LiteralTestUtil::ExpectNear(*expected, *result, ErrorSpec(1e-4));
}

TEST_F(HloCseTest, CombineTwoConstantsDifferentLayoutsAndInsensitive) {
  // Test that two identical constants with different layouts are commoned if
  // the pass is not layout sensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   /*minor_to_major=*/{0, 1})));
  auto constant2 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   /*minor_to_major=*/{1, 0})));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(2, computation->instruction_count());
  auto first_operand = add->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(constant1, constant2));
  EXPECT_THAT(add, op::Add(first_operand, first_operand));

  auto result = ExecuteAndTransfer(std::move(module), {});
  auto expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
  LiteralTestUtil::ExpectNear(*expected, *result, ErrorSpec(1e-4));
}

TEST_F(HloCseTest, CombineTwoConstantsDifferentLayoutsAndSensitive) {
  // Test that two identical constants with different layouts are *not* commoned
  // if the pass is layout sensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   /*minor_to_major=*/{0, 1})));
  auto constant2 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   /*minor_to_major=*/{1, 0})));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  auto result = ExecuteAndTransfer(std::move(module), {});
  auto expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
  LiteralTestUtil::ExpectNear(*expected, *result, ErrorSpec(1e-4));
}

TEST_F(HloCseTest, ConstantsSameValueDifferentType) {
  // Test that constants with the same value but different type are *not*
  // commoned.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint64>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<double>(42.0)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  // Duplicate the float constant to verify something happens.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(7, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(6, computation->instruction_count());
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple,
              op::Tuple(common_constant1, common_constant2, uncommon_constant));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2, exp3));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(exp1, exp2, exp3));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand, first_operand));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).ValueOrDie());

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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(exp1, exp2));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(exp1, exp2));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(8, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(op::Add(negate1, exp1), op::Add(negate2, exp2)));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(rng1, rng2));

  uint32 count_before = computation->instruction_count();

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_FALSE(cse.Run(module.get()).ValueOrDie());

  uint32 count_after = computation->instruction_count();
  EXPECT_EQ(count_before, count_after);
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(rng1, rng2));
}

// TODO(b/28245743): Handle impure functions correctly in CSE.
TEST_F(HloCseTest, DISABLED_DoNotCombineCallsToImpureFunctions) {
  // Test that two calls to an impure function are not commoned. RNG
  // is the source of the impurity.

  auto module = CreateNewModule();

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

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).ValueOrDie());

  EXPECT_EQ(4, computation->instruction_count());
  root = computation->root_instruction();
  auto operand = root->operand(0)->operand(0);
  EXPECT_THAT(operand, op::Map());
  EXPECT_THAT(root, op::Add(operand, operand));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}
