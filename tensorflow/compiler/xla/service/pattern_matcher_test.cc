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
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

namespace m = match;

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
  using match::ConstantEffectiveScalar;
  using match::ConstantScalar;
  using match::Op;
  using match::Tuple;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      a = s32[] constant(1)
      b = s32[1,1] constant(s32[1,1]{{2}})
      c = s32[1,2] constant(s32[1,2]{{2,2}})
      d = f32[] constant(1)
      e = f32[] constant(1.25)
      ROOT tuple = (s32[], s32[1,1], s32[1,2], f32[], f32[]) tuple(a,b,c,d,e)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* a = root->operand(0);
  const HloInstruction* b = root->operand(1);
  const HloInstruction* c = root->operand(2);
  const HloInstruction* d = root->operand(3);
  const HloInstruction* e = root->operand(4);
  EXPECT_TRUE(Match(a, ConstantScalar()));
  EXPECT_TRUE(Match(a, ConstantScalar(1)));
  EXPECT_TRUE(Match(a, ConstantEffectiveScalar()));
  EXPECT_TRUE(Match(a, ConstantEffectiveScalar(1)));
  EXPECT_FALSE(Match(a, ConstantScalar(2)));
  EXPECT_FALSE(Match(a, ConstantScalar(2.01)));
  EXPECT_FALSE(Match(a, ConstantEffectiveScalar(2)));
  EXPECT_FALSE(Match(a, ConstantEffectiveScalar(1.01)));

  EXPECT_FALSE(Match(b, ConstantScalar()));
  EXPECT_FALSE(Match(b, ConstantScalar(2)));
  EXPECT_TRUE(Match(b, ConstantEffectiveScalar()));
  EXPECT_TRUE(Match(b, ConstantEffectiveScalar(2)));

  EXPECT_FALSE(Match(c, ConstantScalar()));
  EXPECT_FALSE(Match(c, ConstantScalar(2)));
  EXPECT_FALSE(Match(c, ConstantEffectiveScalar()));
  EXPECT_FALSE(Match(c, ConstantEffectiveScalar(2)));

  EXPECT_TRUE(Match(d, ConstantScalar(1)));
  EXPECT_TRUE(Match(d, ConstantEffectiveScalar(1)));
  EXPECT_TRUE(Match(d, ConstantScalar(1.0)));
  EXPECT_TRUE(Match(d, ConstantEffectiveScalar(1.0)));

  EXPECT_TRUE(Match(e, ConstantScalar(1.25f)));
  EXPECT_TRUE(Match(e, ConstantScalar(1.25)));
  EXPECT_TRUE(Match(e, ConstantEffectiveScalar(1.25)));
  EXPECT_FALSE(Match(e, ConstantScalar(1)));
  EXPECT_FALSE(Match(e, ConstantEffectiveScalar(1)));

  const HloInstruction* instr = nullptr;
  EXPECT_TRUE(Match(a, ConstantScalar(&instr)));
  EXPECT_EQ(instr, a);

  instr = nullptr;
  EXPECT_TRUE(Match(a, ConstantScalar(&instr, 1)));
  EXPECT_EQ(instr, a);

  instr = nullptr;
  EXPECT_TRUE(Match(a, ConstantEffectiveScalar(&instr)));
  EXPECT_EQ(instr, a);

  instr = nullptr;
  EXPECT_TRUE(Match(a, ConstantEffectiveScalar(&instr, 1)));
  EXPECT_EQ(instr, a);
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

  // Check that MultiplyAnyOrder exposes the same API as Op(), so we can call
  // e.g. IsNonConstant() on it.
  EXPECT_TRUE(Match(
      root, MultiplyAnyOrder(&instr, ConstantScalar(42), ConstantScalar(52))
                .IsNonConstant()));
  EXPECT_TRUE(
      Match(root, MultiplyAnyOrder(ConstantScalar(42), ConstantScalar(52))
                      .IsNonConstant()));
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

  auto f16_scalar = ShapeUtil::MakeShape(F16, {});
  auto f16_pattern = Constant().WithShapeEqualTo(&f16_scalar);
  auto f16_compatible_pattern = Constant().WithShapeCompatibleTo(&f16_scalar);
  auto scalar_pattern = Constant().WithShape(match::Shape().IsScalar());
  ASSERT_TRUE(Match(root, scalar_pattern));
  ASSERT_TRUE(Match(root, f16_pattern));
  ASSERT_TRUE(Match(root, f16_compatible_pattern));
  EXPECT_TRUE(Match(root, AllOf<HloInstruction>(scalar_pattern, f16_pattern,
                                                f16_compatible_pattern)));
  EXPECT_TRUE(
      Match(root, AllOf<HloInstruction>(f16_pattern, f16_compatible_pattern,
                                        scalar_pattern)));
  EXPECT_FALSE(
      Match(root, AllOf<HloInstruction>(Broadcast(Op()), f16_pattern)));
  EXPECT_FALSE(Match(
      root, AllOf<HloInstruction>(Broadcast(Op()), f16_compatible_pattern)));
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

TEST(PatternMatcherTest, TestConcat) {
  using match::Concatenate;
  using match::ConstantScalar;
  using match::Op;
  using match::Reshape;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      c1 = u32[] constant(1)
      c2 = u32[] constant(2)
      c3 = u32[] constant(3)
      c4 = u32[] constant(4)
      r1 = u32[1] reshape(c1)
      r2 = u32[1] reshape(c2)
      r3 = u32[1] reshape(c3)
      r4 = u32[1] reshape(c4)
      ROOT concat = u32[4] concatenate(r1, r2, r3, r4), dimensions={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloString(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  ASSERT_TRUE(Match(
      root,
      Concatenate(Reshape(ConstantScalar(1)), Reshape(ConstantScalar(2)),
                  Reshape(ConstantScalar(3)), Reshape(ConstantScalar(4)))));
  ASSERT_FALSE(Match(
      root,
      Concatenate(Reshape(ConstantScalar(2)), Reshape(ConstantScalar(1)),
                  Reshape(ConstantScalar(3)), Reshape(ConstantScalar(4)))));
  ASSERT_FALSE(Match(
      root, Concatenate(Reshape(ConstantScalar(1)), Reshape(ConstantScalar(2)),
                        Reshape(ConstantScalar(3)))));
  ASSERT_FALSE(Match(
      root, Concatenate(Reshape(ConstantScalar(2)), Reshape(ConstantScalar(3)),
                        Reshape(ConstantScalar(4)))));
}

template <typename Pattern>
string Description(const Pattern& pattern) {
  std::stringstream ss;
  pattern.DescribeTo(&ss);
  return ss.str();
}

template <typename Elem, typename Pattern>
string Explanation(Elem* elem, const Pattern& pattern) {
  std::stringstream ss;
  MatchOption options{/*.capture=*/true, /*.explain_os=*/&ss};
  Match(elem, pattern, options);
  return ss.str();
}
template <typename Elem, typename Pattern>
string Explanation(const std::unique_ptr<Elem>& elem, const Pattern& pattern) {
  return Explanation(elem.get(), pattern);
}
template <typename Elem, typename Pattern>
string Explanation(const Elem& elem, const Pattern& pattern) {
  return Explanation(&elem, pattern);
}

// Helper macro for checking a pattern's description and the explanation printed
// when attempting to match (and presumably failing) on a given object.
//
// We use a macro rather than a function because we want good line numbers in
// errors.  We use this rather than writing a helper that returns a pair of
// (description, explanation) and doing something like
//
//   EXPECT_THAT(DescAndExplanation(...), ::testing::Pair(..., ...));
//
// because EXPECT_EQ prints a unified diff if multiline string comparison fails,
// while EXPECT_THAT does not.  This unified diff makes the errors much easier
// to read.
#define EXPECT_DESC_AND_EXPLANATION(elem, pattern, expected_desc,    \
                                    expected_explanation)            \
  do {                                                               \
    EXPECT_EQ(Description(pattern), (expected_desc));                \
    EXPECT_EQ(Explanation((elem), (pattern)), expected_explanation); \
  } while (0)

TEST(PatternMatcherTest, LayoutDescribeToAndExplain) {
  auto layout = LayoutUtil::MakeLayout({1, 2});
  auto layout2 = LayoutUtil::MakeLayout({2, 2});

  EXPECT_DESC_AND_EXPLANATION(static_cast<const Layout*>(nullptr), m::Layout(),
                              "a layout", "Layout is null");
  EXPECT_DESC_AND_EXPLANATION(layout2, m::Layout().EqualTo(&layout),
                              "a layout equal to {1,2}",
                              "Layout {2,2} is not equal to expected {1,2}");
  EXPECT_DESC_AND_EXPLANATION(layout2, m::Layout().WithSparseFormat(),
                              "a layout with format SPARSE",
                              "Layout has format DENSE but expected SPARSE");
  EXPECT_DESC_AND_EXPLANATION(layout,
                              m::Layout().EqualTo(&layout).WithSparseFormat(),
                              "a layout:\n"
                              " * equal to {1,2} AND\n"
                              " * with format SPARSE",
                              "Layout has format DENSE but expected SPARSE");
}

TEST(PatternMatcherTest, ShapeDescribeToAndExplain) {
  auto shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 2}, {0, 1});
  auto layout = shape.layout();

  EXPECT_DESC_AND_EXPLANATION(static_cast<const Shape*>(nullptr), m::Shape(),
                              "a shape", "Shape is null");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeShapeWithLayout(F32, {1, 2}, {1, 0}),
      m::Shape().EqualTo(&shape), "a shape equal to f32[1,2]{0,1}",
      "Shape not equal to f32[1,2]{0,1}\n"
      "in f32[1,2]{1,0}");
  EXPECT_DESC_AND_EXPLANATION(ShapeUtil::MakeShape(F32, {2, 2}),
                              m::Shape().CompatibleTo(&shape),
                              "a shape compatible with f32[1,2]",
                              "Shape not compatible with f32[1,2]\n"
                              "in f32[2,2]{1,0}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().WithElementType(F16),
                              "a shape with element type F16",
                              "Shape does not have element type F16\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().IsScalar(),
                              "a shape that represents a scalar",
                              "Shape is not a scalar\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(ShapeUtil::MakeNil(), m::Shape().IsArray(),
                              "a shape that represents an array",
                              "Shape is not an array\n"
                              "in ()");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().IsTuple(),
                              "a shape that represents a tuple",
                              "Shape is not a tuple\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().IsEffectiveScalar(),
                              "a shape that is an effective scalar",
                              "Shape is not an effective scalar\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().WithRank(42),
                              "a shape that has 42 dimensions",
                              "Shape does not have rank 42\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().WithRank(0),
                              "a shape that is a scalar",
                              "Shape is not a scalar\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape, m::Shape().WithRank(1).IsArray(),
                              "a shape:\n"
                              " * that has 1 dimension AND\n"
                              " * that represents an array",
                              "Shape does not have rank 1\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(ShapeUtil::MakeNil(),
                              m::Shape().IsArray().WithRank(1),
                              "a shape:\n"
                              " * that represents an array AND\n"
                              " * that has 1 dimension",
                              "Shape is not an array\n"
                              "in ()");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeShapeWithLayout(F32, {1, 2}, {1, 0}),
      m::Shape().WithLayoutEqualTo(&layout),
      "a shape with\n  a layout equal to {0,1}",
      "Layout {1,0} is not equal to expected {0,1}\n"
      "in f32[1,2]{1,0}");
  EXPECT_DESC_AND_EXPLANATION(
      shape, m::Shape().WithLayout(m::Layout().WithSparseFormat()),
      "a shape with\n  a layout with format SPARSE",
      "Layout has format DENSE but expected SPARSE\n"
      "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(shape,
                              m::Shape().WithSubshapeEqualTo({10}, &shape),
                              "a shape with subshape at index {10} which is\n"
                              "  a shape equal to f32[1,2]{0,1}",
                              "No subshape at {10}\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2, 2})}),
      m::Shape().WithSubshapeEqualTo({0}, &shape),
      "a shape with subshape at index {0} which is\n"
      "  a shape equal to f32[1,2]{0,1}",
      "Shape not equal to f32[1,2]{0,1}\n"
      "in f32[2,2]{1,0}\n"
      "in subshape at {0}\n"
      "in (f32[2,2])");
  EXPECT_DESC_AND_EXPLANATION(shape,
                              m::Shape().WithSubshapeCompatibleTo({10}, &shape),
                              "a shape with subshape at index {10} which is\n"
                              "  a shape compatible with f32[1,2]",
                              "No subshape at {10}\n"
                              "in f32[1,2]{0,1}");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2, 2})}),
      m::Shape().WithSubshapeCompatibleTo({0}, &shape),
      "a shape with subshape at index {0} which is\n"
      "  a shape compatible with f32[1,2]",
      "Shape not compatible with f32[1,2]\n"
      "in f32[2,2]{1,0}\n"
      "in subshape at {0}\n"
      "in (f32[2,2])");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTupleShape({shape})}),
      m::Shape().WithSubshape({0, 0}, m::Shape().IsScalar()),
      "a shape with subshape at index {0,0} which is\n"
      "  a shape that represents a scalar",
      "Shape is not a scalar\n"
      "in f32[1,2]{0,1}\n"
      "in subshape at {0,0}\n"
      "in ((f32[1,2]))");
}

std::unique_ptr<HloInstruction> SetName(absl::string_view name,
                                        std::unique_ptr<HloInstruction> instr) {
  instr->SetAndSanitizeName(string(name));
  return instr;
}

TEST(PatternMatcherTest, HloInstructionDescribeToAndExplain) {
  std::unique_ptr<HloInstruction> iota =
      SetName("i", HloInstruction::CreateIota(ShapeUtil::MakeShape(S32, {42}),
                                              /*iota_dimension=*/0));
  std::unique_ptr<HloInstruction> constant =
      SetName("c", HloInstruction::CreateConstant(LiteralUtil::CreateR0(0)));

  EXPECT_DESC_AND_EXPLANATION(static_cast<const HloInstruction*>(nullptr),
                              m::Op(), "an HloInstruction",
                              "HloInstruction* is null");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithName("foo"),
                              "an HloInstruction named \"foo\"",
                              "HloInstruction not named \"foo\"\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithOpcode(HloOpcode::kAdd),
                              "an HloInstruction with opcode add",
                              "HloInstruction doesn't have opcode add\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(
      constant, m::Op().IsNonConstant(),
      "an HloInstruction with any opcode other than constant",
      "HloInstruction has opcode constant, expected anything else\n"
      "in c = s32[] constant(0)");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithNumOperands(42),
                              "an HloInstruction with 42 operands",
                              "HloInstruction doesn't have 42 operands\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithShape(m::Shape().IsTuple()),
                              "an HloInstruction outputting\n"
                              "  a shape that represents a tuple",
                              "Shape is not a tuple\n"
                              "in s32[42]{0}\n"
                              "in output shape\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(
      iota, m::Op().WithOperand(2, m::Op().WithOpcode(HloOpcode::kAdd)),
      "an HloInstruction with operand 2 which is:\n"
      "  an HloInstruction with opcode add",
      "desired operand index 2 is out of bounds\n"
      "in i = s32[42]{0} iota(), iota_dimension=0");

  EXPECT_DESC_AND_EXPLANATION(
      SetName("a", HloInstruction::CreateBinary(ShapeUtil::MakeShape(S32, {}),
                                                HloOpcode::kAdd, constant.get(),
                                                constant.get())),
      m::Op().WithOperand(1, m::Op().IsNonConstant()),
      "an HloInstruction with operand 1 which is:\n"
      "  an HloInstruction with any opcode other than constant",
      "HloInstruction has opcode constant, expected anything else\n"
      "in c = s32[] constant(0)\n"
      "in operand 1\n"
      "in a = s32[] add(s32[] c, s32[] c)");
  EXPECT_DESC_AND_EXPLANATION(
      iota, m::Op().WithFusionKind(HloInstruction::FusionKind::kLoop),
      "an HloInstruction with fusion kind kLoop",
      "HloInstruction does not have fusion kind kLoop; it's not a fusion\n"
      "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(
      iota, m::Op().WithTupleIndex(42),
      "an HloInstruction which is a GTE with index 42",
      "HloInstruction is not a GTE with index 42; it's not a GTE at all\n"
      "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().IsConstantScalar(),
                              "an HloInstruction which is a constant scalar",
                              "HloInstruction is not a constant\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(
      SetName("c", HloInstruction::CreateConstant(
                       LiteralUtil::CreateR1<int>({1, 2}))),
      m::Op().IsConstantEffectiveScalar(),
      "an HloInstruction which is a constant effective scalar",
      "HloInstruction is not an effective scalar\n"
      "in c = s32[2]{0} constant({1, 2})");
  EXPECT_DESC_AND_EXPLANATION(
      SetName("c", HloInstruction::CreateConstant(LiteralUtil::CreateR0(10))),
      m::Op().IsConstantScalar(42),
      "an HloInstruction which is a constant scalar with value 42",
      "HloInstruction's constant value 10 did not match expected value 42\n"
      "in c = s32[] constant(10)");
  EXPECT_DESC_AND_EXPLANATION(
      SetName("c", HloInstruction::CreateConstant(LiteralUtil::CreateR0(2.25))),
      m::Op().IsConstantEffectiveScalar(1.25),
      "an HloInstruction which is a constant effective scalar with value 1.25",
      "HloInstruction's constant value 2.25 did not match expected value 1.25\n"
      "in c = f64[] constant(2.25)");
  EXPECT_DESC_AND_EXPLANATION(
      constant, m::Op().Is(iota.get()),
      absl::StrCat("an HloInstruction which is 0x", absl::Hex(iota.get()), " (",
                   iota->ToShortString(), ")"),
      absl::StrCat("HloInstruction 0x", absl::Hex(constant.get()), " is not 0x",
                   absl::Hex(iota.get()), " (", iota->ToShortString(), ")\n",
                   "in c = s32[] constant(0)"));
}

TEST(PatternMatcherTest, HloInstructionMatcherAnyOrderDescribeTo) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  EXPECT_DESC_AND_EXPLANATION(
      SetName("a", HloInstruction::CreateBinary(
                       scalar_s32, HloOpcode::kAdd,
                       SetName("b", HloInstruction::CreateConstant(
                                        LiteralUtil::CreateR0(0)))
                           .get(),
                       SetName("c", HloInstruction::CreateConstant(
                                        LiteralUtil::CreateR0(0)))
                           .get())),
      m::AddAnyOrder(m::Op().WithName("b"), m::Op().WithName("bar")),
      "an HloInstruction:\n"
      " * with opcode add AND\n"
      " * with two operands in either order:\n"
      "    - an HloInstruction named \"b\"\n"
      "    - an HloInstruction named \"bar\"",
      "HloInstruction's operands (ignoring order) did not match second "
      "matcher.  Specifically,\n"
      " - an HloInstruction named \"bar\"\n"
      "does not match LHS:\n"
      " - HloInstruction not named \"bar\"\n"
      "   in b = s32[] constant(0)\n"
      "does not match RHS:\n"
      " - HloInstruction not named \"bar\"\n"
      "   in c = s32[] constant(0)\n"
      "in a = s32[] add(s32[] b, s32[] c)");

  EXPECT_DESC_AND_EXPLANATION(
      SetName("a",
              HloInstruction::CreateBinary(
                  scalar_s32, HloOpcode::kAdd,
                  HloInstruction::CreateParameter(0, scalar_s32, "p").get(),
                  SetName("c", HloInstruction::CreateConstant(
                                   LiteralUtil::CreateR0(0)))
                      .get())),
      m::AddAnyOrder(m::Op().IsConstantScalar(), m::Op().IsConstant()),
      "an HloInstruction:\n"
      " * with opcode add AND\n"
      " * with two operands in either order:\n"
      "    - an HloInstruction which is a constant scalar\n"
      "    - an HloInstruction with opcode constant",
      "HloInstruction's LHS operand did not match either of the two matchers.  "
      "Specifically,\n"
      " - an HloInstruction which is a constant scalar\n"
      "does not match LHS:\n"
      " - HloInstruction is not a constant\n"
      "   in p = s32[] parameter(0)\n"
      "and\n"
      " - an HloInstruction with opcode constant\n"
      "does not match LHS:\n"
      " - HloInstruction doesn't have opcode constant\n"
      "   in p = s32[] parameter(0)\n"
      "in a = s32[] add(s32[] p, s32[] c)");
}

TEST(PatternMatcherTest, AnyOfMatcherDescribeToAndExplain) {
  EXPECT_DESC_AND_EXPLANATION(
      SetName("c", HloInstruction::CreateConstant(LiteralUtil::CreateR0(0))),
      m::AnyOf<HloInstruction>(m::Op().WithName("foo"),
                               m::Op().WithName("bar")),
      "any of:\n"
      " - an HloInstruction named \"foo\" OR\n"
      " - an HloInstruction named \"bar\"",
      "None of the following matchers succeeded:\n"
      "Matcher #1\n"
      " - an HloInstruction named \"foo\"\n"
      "failed with\n"
      " - HloInstruction not named \"foo\"\n"
      "   in c = s32[] constant(0)\n"
      "Matcher #2\n"
      " - an HloInstruction named \"bar\"\n"
      "failed with\n"
      " - HloInstruction not named \"bar\"\n"
      "   in c = s32[] constant(0)");
}

TEST(PatternMatcherTest, Parameter) {
  auto param =
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "p1");
  auto non_param =
      SetName("c", HloInstruction::CreateConstant(LiteralUtil::CreateR0(0)));
  EXPECT_FALSE(Match(param.get(), m::Parameter(0)));
  EXPECT_TRUE(Match(param.get(), m::Parameter()));
  EXPECT_TRUE(Match(param.get(), m::Parameter(1)));
  EXPECT_FALSE(Match(non_param.get(), m::Parameter()));
  EXPECT_FALSE(Match(non_param.get(), m::Parameter(1)));

  EXPECT_DESC_AND_EXPLANATION(non_param, m::Parameter(1),
                              "an HloInstruction:\n"
                              " * with opcode parameter AND\n"
                              " * which is parameter 1",
                              "HloInstruction doesn't have opcode parameter\n"
                              "in c = s32[] constant(0)");
  EXPECT_EQ(Explanation(HloInstruction::CreateParameter(
                            0, ShapeUtil::MakeShape(F32, {}), "p0"),
                        m::Parameter(1)),
            "HloInstruction is not parameter 1\n"
            "in p0 = f32[] parameter(0)");
}

}  // namespace
}  // namespace xla
