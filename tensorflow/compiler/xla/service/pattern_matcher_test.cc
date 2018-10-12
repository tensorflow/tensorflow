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

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(PatternMatcherTest, AddOp) {
  constexpr char kModuleStr[] = R"(HloModule two_plus_two_module
    ENTRY %two_plus_two_computation () -> f32[] {
      %two = f32[] constant(2)
      ROOT %two_plus_two = f32[] add(f32[] %two, f32[] %two)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));

  const HloInstruction* matched_inst;
  HloInstruction* matched_operand;
  Shape* matched_shape;
  Layout* matched_layout;

  ASSERT_TRUE(Match(
      hlo_module->entry_computation()->root_instruction(),
      match::Op(&matched_inst)
          .WithName("two_plus_two")
          .WithOpcode(HloOpcode::kAdd)
          .WithShape(
              match::Shape(&matched_shape)
                  .WithLayout(match::Layout(&matched_layout).WithDenseFormat()))
          .WithOperand(
              0,
              match::Op(&matched_operand).WithOpcode(HloOpcode::kConstant))));
  ASSERT_NE(matched_inst, nullptr);
  EXPECT_EQ(matched_inst->name(), "two_plus_two");
  EXPECT_EQ(matched_inst->opcode(), HloOpcode::kAdd);

  EXPECT_TRUE(Match(hlo_module->entry_computation()->root_instruction(),
                    match::Add(match::Constant(), match::Constant())));

  EXPECT_FALSE(Match(hlo_module->entry_computation()->root_instruction(),
                     match::Op().WithName("bad_name")));
  matched_inst = nullptr;
  EXPECT_FALSE(Match(hlo_module->entry_computation()->root_instruction(),
                     match::Multiply(&matched_inst, match::Op(), match::Op())));
}

TEST(PatternMatcherTest, ScalarShape) {
  auto scalar_shape = ShapeUtil::MakeShape(F32, {});
  Shape* matched_shape;
  EXPECT_TRUE(Match(&scalar_shape, match::Shape(&matched_shape).IsScalar()));
  EXPECT_EQ(matched_shape, &scalar_shape);
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().IsArray()));
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().IsDenseArray()));
  EXPECT_FALSE(Match(&scalar_shape, match::Shape().IsTuple()));
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().WithElementType(F32)));
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().WithRank(0)));
  EXPECT_FALSE(Match(
      &scalar_shape,
      match::Shape().WithSubshape({0}, match::Shape()).WithElementType(F32)));
}

TEST(PatternMatcherTest, DenseArrayShape) {
  auto array_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape* matched_shape;
  EXPECT_TRUE(Match(&array_shape, match::Shape(&matched_shape).IsArray()));
  EXPECT_EQ(matched_shape, &array_shape);
  EXPECT_TRUE(Match(&array_shape, match::Shape().IsDenseArray()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsSparseArray()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsScalar()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsTuple()));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithElementType(F32)));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithRank(3)));
  EXPECT_FALSE(
      Match(&array_shape, match::Shape().WithSubshape({0}, match::Shape())));
  Layout* matched_layout;
  EXPECT_FALSE(Match(&array_shape,
                     match::Shape().WithLayout(
                         match::Layout(&matched_layout).WithSparseFormat())));
  EXPECT_TRUE(Match(&array_shape,
                    match::Shape().WithLayout(
                        match::Layout(&matched_layout).WithDenseFormat())));
  EXPECT_EQ(matched_layout, &array_shape.layout());
}

TEST(PatternMatcherTest, SparseArrayShape) {
  auto array_shape = ShapeUtil::MakeShapeWithSparseLayout(F32, {2, 3, 4}, 10);
  Shape* matched_shape;
  EXPECT_TRUE(Match(&array_shape, match::Shape(&matched_shape).IsArray()));
  EXPECT_EQ(matched_shape, &array_shape);
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsDenseArray()));
  EXPECT_TRUE(Match(&array_shape, match::Shape().IsSparseArray()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsScalar()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsTuple()));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithElementType(F32)));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithRank(3)));
  EXPECT_FALSE(
      Match(&array_shape, match::Shape().WithSubshape({0}, match::Shape())));
  Layout* matched_layout;
  EXPECT_FALSE(Match(&array_shape,
                     match::Shape().WithLayout(
                         match::Layout(&matched_layout).WithDenseFormat())));
  EXPECT_TRUE(Match(&array_shape,
                    match::Shape().WithLayout(
                        match::Layout(&matched_layout).WithSparseFormat())));
  EXPECT_EQ(matched_layout, &array_shape.layout());
}

TEST(PatternMatcherTest, TupleShape) {
  auto tuple_shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {1, 2, 3}),
      ShapeUtil::MakeShape(S32, {4, 5}),
  });
  EXPECT_TRUE(Match(&tuple_shape, match::Shape().IsTuple()));
  EXPECT_FALSE(Match(&tuple_shape, match::Shape().IsArray()));
  EXPECT_FALSE(Match(&tuple_shape, match::Shape().IsScalar()));

  Shape* subshape;
  ASSERT_TRUE(Match(
      &tuple_shape,
      match::Shape().WithSubshape(
          {0}, match::Shape(&subshape).WithElementType(F32).WithRank(3))));
  ASSERT_NE(subshape, nullptr);
  EXPECT_TRUE(
      ShapeUtil::Equal(*subshape, ShapeUtil::GetSubshape(tuple_shape, {0})));
  EXPECT_TRUE(Match(&tuple_shape,
                    match::Shape().WithSubshape(
                        {0}, match::Shape().EqualTo(
                                 &ShapeUtil::GetSubshape(tuple_shape, {0})))));
  EXPECT_FALSE(Match(&tuple_shape,
                     match::Shape().WithSubshape(
                         {0}, match::Shape().EqualTo(
                                  &ShapeUtil::GetSubshape(tuple_shape, {1})))));

  ASSERT_TRUE(Match(
      &tuple_shape,
      match::Shape().WithSubshape(
          {1}, match::Shape(&subshape).WithElementType(S32).WithRank(2))));
  ASSERT_NE(subshape, nullptr);
  EXPECT_TRUE(
      ShapeUtil::Equal(*subshape, ShapeUtil::GetSubshape(tuple_shape, {1})));
  EXPECT_TRUE(Match(&tuple_shape,
                    match::Shape().WithSubshape(
                        {1}, match::Shape().EqualTo(
                                 &ShapeUtil::GetSubshape(tuple_shape, {1})))));
  EXPECT_FALSE(Match(&tuple_shape,
                     match::Shape().WithSubshape(
                         {1}, match::Shape().EqualTo(
                                  &ShapeUtil::GetSubshape(tuple_shape, {0})))));

  EXPECT_FALSE(
      Match(&tuple_shape, match::Shape().WithSubshape({2}, match::Shape())));
  EXPECT_FALSE(
      Match(&tuple_shape, match::Shape().WithSubshape({0, 0}, match::Shape())));
}

TEST(PatternMatcherTest, FusionKind) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    fused_computation {
      ROOT fp0 = f32[] parameter(0)
    }

    ENTRY while.v11 {
      p0 = f32[] parameter(0)
      ROOT fusion = f32[] fusion(p0), kind=kLoop, calls=fused_computation
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));

  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root, match::Op().WithFusionKind(HloInstruction::FusionKind::kLoop)));
  EXPECT_FALSE(Match(
      root, match::Op().WithFusionKind(HloInstruction::FusionKind::kInput)));
  EXPECT_FALSE(Match(root->operand(0), match::Op().WithFusionKind(
                                           HloInstruction::FusionKind::kLoop)));
}

TEST(PatternMatcherTest, GetTupleElement) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    ENTRY while.v11 {
      p0 = (f32[], f32[], f32[]) parameter(0)
      ROOT gte = f32[] get-tuple-element(p0), index=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));

  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_FALSE(Match(root, match::Op().WithTupleIndex(0)));
  EXPECT_TRUE(Match(root, match::Op().WithTupleIndex(1)));
  EXPECT_FALSE(Match(root, match::Op().WithTupleIndex(2)));
  EXPECT_FALSE(Match(root, match::GetTupleElement(match::Op(), 0)));
  EXPECT_TRUE(Match(root, match::GetTupleElement(match::Op(), 1)));
}

TEST(PatternMatcherTest, AnyOf) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(1) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root, match::AnyOf<HloInstruction>(match::ConstantScalar(0),
                                               match::ConstantScalar(1))));
  EXPECT_TRUE(
      Match(root, match::AnyOf<HloInstruction>(match::ConstantScalar(1),
                                               match::ConstantScalar(0))));
  EXPECT_FALSE(
      Match(root, match::AnyOf<HloInstruction>(match::ConstantScalar(0),
                                               match::ConstantScalar(2))));
}

TEST(PatternMatcherTest, ConstantScalar) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(42) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_TRUE(Match(root, match::ConstantScalar(42)));
  EXPECT_FALSE(Match(root, match::ConstantScalar(41)));
  EXPECT_FALSE(Match(root, match::ConstantScalar(0)));
}

TEST(PatternMatcherTest, NoMatchConstantScalar) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT v = f16[] parameter(0) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_FALSE(Match(root, match::ConstantScalar(42)));
}

TEST(PatternMatcherTest, MultiplyAnyOrder) {
  using match::ConstantScalar;
  using match::MultiplyAnyOrder;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      lhs = f16[] constant(42)
      rhs = f16[] constant(52)
      ROOT multiply = f16[] multiply(lhs, rhs)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  const HloInstruction* instr;

  EXPECT_TRUE(Match(
      root, MultiplyAnyOrder(&instr, ConstantScalar(42), ConstantScalar(52))));
  EXPECT_TRUE(Match(
      root, MultiplyAnyOrder(&instr, ConstantScalar(52), ConstantScalar(42))));
}

TEST(PatternMatcherTest, AnyOfShortCircuit) {
  using match::AnyOf;
  using match::Multiply;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      lhs = f16[] constant(42)
      rhs = f16[] constant(52)
      ROOT multiply = f16[] multiply(lhs, rhs)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  {
    const HloInstruction* mul = nullptr;
    const HloInstruction* any = nullptr;

    ASSERT_TRUE(Match(
        root, AnyOf<HloInstruction>(Multiply(&mul, Op(), Op()), Op(&any))));
    EXPECT_NE(nullptr, mul);
    EXPECT_EQ(nullptr, any);
  }
  {
    const HloInstruction* mul = nullptr;
    const HloInstruction* any = nullptr;

    ASSERT_TRUE(Match(
        root, AnyOf<HloInstruction>(Op(&any), Multiply(&mul, Op(), Op()))));
    EXPECT_NE(nullptr, any);
    EXPECT_EQ(nullptr, mul);
  }
}

TEST(PatternMatcherTest, AllOf) {
  using match::AllOf;
  using match::Broadcast;
  using match::Constant;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(1) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  auto scalar_pattern = Constant().WithShape(match::Shape().IsScalar());
  auto f16_pattern = Constant().WithShape(match::Shape().WithElementType(F16));
  ASSERT_TRUE(Match(root, scalar_pattern));
  ASSERT_TRUE(Match(root, f16_pattern));
  EXPECT_TRUE(Match(root, AllOf<HloInstruction>(scalar_pattern, f16_pattern)));
  EXPECT_TRUE(Match(root, AllOf<HloInstruction>(f16_pattern, scalar_pattern)));
  EXPECT_FALSE(
      Match(root, AllOf<HloInstruction>(Broadcast(Op()), f16_pattern)));
  EXPECT_FALSE(
      Match(root, AllOf<HloInstruction>(Broadcast(Op()), scalar_pattern)));
}

TEST(PatternMatcherTest, AllOfNoCaptureIfNotMatch) {
  using match::AllOf;
  using match::Broadcast;
  using match::Constant;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT v = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* constant = nullptr;
  ASSERT_FALSE(
      Match(root, AllOf<HloInstruction>(Constant(&constant), Broadcast(Op()))));
  EXPECT_EQ(nullptr, constant);
  ASSERT_TRUE(Match(root, Constant(&constant)));
  EXPECT_NE(nullptr, constant);
}

TEST(PatternMatcherTest, TestNoCapture) {
  using match::Constant;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT v = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* constant = nullptr;
  ASSERT_TRUE(Match(root, Constant(&constant), {/*capture=*/false}));
  EXPECT_EQ(nullptr, constant);
}

TEST(PatternMatcherTest, TestCaptureMatchedSubPatternForAnyOf) {
  using match::Add;
  using match::AddAnyOrder;
  using match::AnyOf;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      u = f16[] parameter(0)
      v = f16[] parameter(1)
      ROOT add = f16[] add(u, v)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* addend0 = nullptr;
  const HloInstruction* addend1 = nullptr;
  const HloInstruction* addend2 = nullptr;
  auto add2_pattern = Add(Op(&addend0), Op(&addend1));
  auto add3_pattern = AnyOf<HloInstruction>(
      AddAnyOrder(add2_pattern, Op(&addend2)), add2_pattern, Op(&addend0));

  ASSERT_TRUE(Match(root, add3_pattern));
  EXPECT_NE(nullptr, addend0);
  EXPECT_NE(nullptr, addend1);
  EXPECT_EQ(nullptr, addend2);
}

}  // namespace
}  // namespace xla
