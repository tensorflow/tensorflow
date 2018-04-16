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
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, tools::Parse(kModuleStr));

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
  EXPECT_FALSE(Match(&scalar_shape, match::Shape().IsTuple()));
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().WithElementType(F32)));
  EXPECT_TRUE(Match(&scalar_shape, match::Shape().WithRank(0)));
  EXPECT_FALSE(Match(
      &scalar_shape,
      match::Shape().WithSubshape({0}, match::Shape()).WithElementType(F32)));
}

TEST(PatternMatcherTest, ArrayShape) {
  auto array_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape* matched_shape;
  EXPECT_TRUE(Match(&array_shape, match::Shape(&matched_shape).IsArray()));
  EXPECT_EQ(matched_shape, &array_shape);
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

}  // namespace
}  // namespace xla
