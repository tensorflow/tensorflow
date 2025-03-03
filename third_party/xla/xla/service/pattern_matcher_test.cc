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

#include "xla/service/pattern_matcher.h"

#include <memory>
#include <sstream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

namespace m = match;
using PatternMatcherTest = HloTestBase;

TEST_F(PatternMatcherTest, AddOp) {
  constexpr char kModuleStr[] = R"(HloModule two_plus_two_module
    ENTRY %two_plus_two_computation () -> f32[] {
      %two = f32[] constant(2)
      ROOT %two_plus_two = f32[] add(f32[] %two, f32[] %two)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  const HloInstruction* matched_inst;
  HloInstruction* matched_operand;
  Shape* matched_shape;

  ASSERT_TRUE(Match(
      hlo_module->entry_computation()->root_instruction(),
      match::Op(&matched_inst)
          .WithName("two_plus_two")
          .WithOpcode(HloOpcode::kAdd)
          .WithShape(match::Shape(&matched_shape).IsDenseArray())
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

TEST_F(PatternMatcherTest, ScalarShape) {
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

TEST_F(PatternMatcherTest, DenseArrayShape) {
  auto array_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape* matched_shape;
  EXPECT_TRUE(Match(&array_shape, match::Shape(&matched_shape).IsArray()));
  EXPECT_EQ(matched_shape, &array_shape);
  EXPECT_TRUE(Match(&array_shape, match::Shape().IsDenseArray()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsScalar()));
  EXPECT_FALSE(Match(&array_shape, match::Shape().IsTuple()));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithElementType(F32)));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithRank(3)));
  EXPECT_FALSE(
      Match(&array_shape, match::Shape().WithSubshape({0}, match::Shape())));
  EXPECT_TRUE(Match(&array_shape, match::Shape().WithLayout({2, 1, 0})));
  EXPECT_FALSE(Match(&array_shape, match::Shape().WithLayout({0, 1, 2})));
  Layout* matched_layout;
  EXPECT_TRUE(Match(&array_shape,
                    match::Shape().WithLayout(match::Layout(&matched_layout))));
  EXPECT_EQ(matched_layout, &array_shape.layout());
  EXPECT_TRUE(Match(&array_shape, match::Shape().IsDenseArray()));
}

TEST_F(PatternMatcherTest, DenseArrayShapeWithLayout) {
  auto array_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 2, 3}, {1, 2, 0});
  Shape* matched_shape;
  EXPECT_TRUE(
      Match(&array_shape, match::Shape(&matched_shape).WithLayout({1, 2, 0})));
  EXPECT_EQ(matched_shape, &array_shape);
  EXPECT_FALSE(Match(&array_shape, match::Shape().WithLayout({2, 0, 1})));
  Layout* matched_layout;
  EXPECT_TRUE(
      Match(&array_shape,
            match::Shape().WithLayout(
                match::Layout(&matched_layout).WithMinorToMajor({1, 2, 0}))));
  EXPECT_EQ(matched_layout, &array_shape.layout());
}

TEST_F(PatternMatcherTest, TupleShape) {
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

TEST_F(PatternMatcherTest, FusionKind) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    fused_computation {
      ROOT fp0 = f32[] parameter(0)
    }

    ENTRY while.v11 {
      p0 = f32[] parameter(0)
      ROOT fusion = f32[] fusion(p0), kind=kLoop, calls=fused_computation
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root, match::Op().WithFusionKind(HloInstruction::FusionKind::kLoop)));
  EXPECT_FALSE(Match(
      root, match::Op().WithFusionKind(HloInstruction::FusionKind::kInput)));
  EXPECT_FALSE(Match(root->operand(0), match::Op().WithFusionKind(
                                           HloInstruction::FusionKind::kLoop)));
}

TEST_F(PatternMatcherTest, GetTupleElement) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    ENTRY while.v11 {
      p0 = (f32[], f32[], f32[]) parameter(0)
      ROOT gte = f32[] get-tuple-element(p0), index=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_FALSE(Match(root, match::Op().WithTupleIndex(0)));
  EXPECT_TRUE(Match(root, match::Op().WithTupleIndex(1)));
  EXPECT_FALSE(Match(root, match::Op().WithTupleIndex(2)));
  EXPECT_FALSE(Match(root, match::GetTupleElement(match::Op(), 0)));
  EXPECT_TRUE(Match(root, match::GetTupleElement(match::Op(), 1)));
}

TEST_F(PatternMatcherTest, AnyOf) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(1) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, AnyOfInstructionIsInstructionPattern) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(1) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root, match::AnyOf<HloInstruction>(match::ConstantScalar(0),
                                               match::ConstantScalar(1))));
  EXPECT_FALSE(
      Match(root, match::AnyOf<HloInstruction>(match::ConstantScalar(0),
                                               match::ConstantScalar(1))
                      .WithName("foo")));
}

TEST_F(PatternMatcherTest, ConstantScalar) {
  using match::ConstantEffectiveScalar;
  using match::ConstantScalar;
  using match::Op;
  using match::Tuple;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      a = s32[] constant(1)
      b = s32[1,1] constant({{2}})
      c = s32[1,2] constant({{2,2}})
      d = f32[] constant(1)
      e = f32[] constant(1.25)
      ROOT tuple = (s32[], s32[1,1], s32[1,2], f32[], f32[]) tuple(a,b,c,d,e)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, MultiplyAnyOrder) {
  using match::ConstantScalar;
  using match::MultiplyAnyOrder;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      lhs = f16[] constant(42)
      rhs = f16[] constant(52)
      ROOT multiply = f16[] multiply(lhs, rhs)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, AnyOfShortCircuit) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, AllOf) {
  using match::AllOf;
  using match::Broadcast;
  using match::Constant;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test { ROOT constant = f16[] constant(1) })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, AllOfNoCaptureIfNotMatch) {
  using match::AllOf;
  using match::Broadcast;
  using match::Constant;
  using match::Op;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT v = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* constant = nullptr;
  ASSERT_FALSE(
      Match(root, AllOf<HloInstruction>(Constant(&constant), Broadcast(Op()))));
  EXPECT_EQ(nullptr, constant);
  ASSERT_TRUE(Match(root, Constant(&constant)));
  EXPECT_NE(nullptr, constant);
}

TEST_F(PatternMatcherTest, TestNoCapture) {
  using match::Constant;

  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT v = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  const HloInstruction* constant = nullptr;
  ASSERT_TRUE(Match(root, Constant(&constant), {/*capture=*/false}));
  EXPECT_EQ(nullptr, constant);
}

TEST_F(PatternMatcherTest, TestCaptureMatchedSubPatternForAnyOf) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, TestConcat) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
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

TEST_F(PatternMatcherTest, TestWithElementType) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT v = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Op().WithElementType(F16)));
  EXPECT_FALSE(Match(root, m::Op().WithElementType(F32)));
}

TEST_F(PatternMatcherTest, TestWithOperandIfPresent) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      a = f16[] constant(42)
      b = f16[] add(a, a)
      ROOT root = tuple(a, b)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  auto* a = root->operand(0);
  auto* b = root->operand(1);

  // No operand 0, but that's ok, still passes.
  EXPECT_TRUE(Match(a, m::Op().WithOperandIfPresent(0, m::Iota())));

  EXPECT_TRUE(Match(b, m::Op().WithOperandIfPresent(0, m::Constant())));
  EXPECT_TRUE(Match(b, m::Op().WithOperandIfPresent(1, m::Constant())));
  EXPECT_FALSE(Match(b, m::Op().WithOperandIfPresent(0, m::Iota())));
  // No operand 2/3, but that's ok, still passes.
  EXPECT_TRUE(Match(b, m::Op().WithOperandIfPresent(2, m::Iota())));
  EXPECT_TRUE(Match(b, m::Op().WithOperandIfPresent(3, m::Iota())));
}

TEST_F(PatternMatcherTest, TestWithPredicate) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      ROOT a = f16[] constant(42)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root, m::Op().WithPredicate([&](const HloInstruction* instr) {
        return instr == root;
      })));
  EXPECT_FALSE(
      Match(root, m::Op().WithPredicate([&](const HloInstruction* instr) {
        return instr != root;
      })));
}

template <typename Pattern>
std::string Description(const Pattern& pattern) {
  std::stringstream ss;
  pattern.DescribeTo(&ss);
  return ss.str();
}

template <typename Elem, typename Pattern>
std::string Explanation(Elem* elem, const Pattern& pattern,
                        bool single_user_only = false) {
  std::stringstream ss;
  MatchOption options{/*.capture=*/true,
                      /*.single_user_only=*/single_user_only,
                      /*.explain_os=*/&ss};
  Match(elem, pattern, options);
  return ss.str();
}
template <typename Elem, typename Pattern>
std::string Explanation(const std::unique_ptr<Elem>& elem,
                        const Pattern& pattern) {
  return Explanation(elem.get(), pattern);
}
template <typename Elem, typename Pattern>
std::string Explanation(const Elem& elem, const Pattern& pattern) {
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

TEST_F(PatternMatcherTest, LayoutDescribeToAndExplain) {
  auto layout = LayoutUtil::MakeLayout({1, 2});
  auto layout2 = LayoutUtil::MakeLayout({2, 2});

  EXPECT_DESC_AND_EXPLANATION(static_cast<const Layout*>(nullptr), m::Layout(),
                              "a layout", "Layout is null");
  EXPECT_DESC_AND_EXPLANATION(layout2, m::Layout().EqualTo(&layout),
                              "a layout equal to {1,2}",
                              "Layout {2,2} is not equal to expected {1,2}");
}

TEST_F(PatternMatcherTest, CustomCallTargetMatcherDescribeAndExplain) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    ENTRY test {
      ROOT out = f32[] custom-call(), custom_call_target="test_target"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, match::Op().WithCustomCallTarget({"test_target"})));
  EXPECT_TRUE(Match(
      root, match::Op().WithCustomCallTarget({"test_target", "other_target"})));
  EXPECT_TRUE(Match(
      root, match::Op().WithCustomCallTarget({"other_target", "test_target"})));
  EXPECT_FALSE(Match(root, match::Op().WithCustomCallTarget({"other_target"})));
  EXPECT_FALSE(Match(root, match::Op().WithCustomCallTarget(
                               {"other_target", "other_target2"})));

  EXPECT_DESC_AND_EXPLANATION(
      root, match::Op().WithCustomCallTarget({"other_target"}),
      "an HloInstruction custom call with target 'other_target'",
      "HloInstruction is not a custom call with a target 'other_target'\nin "
      "out = f32[] custom-call(), custom_call_target=\"test_target\"");

  EXPECT_DESC_AND_EXPLANATION(
      root, match::Op().WithCustomCallTarget({"other_target", "other_target2"}),
      "an HloInstruction custom call with target in {other_target, "
      "other_target2}",
      "HloInstruction is not a custom call with a target in {other_target, "
      "other_target2}\nin "
      "out = f32[] custom-call(), custom_call_target=\"test_target\"");
}

TEST_F(PatternMatcherTest, ShapeDescribeToAndExplain) {
  auto shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2}, {0, 1});
  auto layout = shape.layout();

  EXPECT_DESC_AND_EXPLANATION(static_cast<const Shape*>(nullptr), m::Shape(),
                              "a shape", "Shape is null");
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2}, {1, 0}),
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
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2}, {1, 0}),
      m::Shape().WithLayoutEqualTo(&layout),
      "a shape with\n  a layout equal to {0,1}",
      "Layout {1,0} is not equal to expected {0,1}\n"
      "in f32[1,2]{1,0}");
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
  instr->SetAndSanitizeName(name);
  return instr;
}

TEST_F(PatternMatcherTest, HloInstructionDescribeToAndExplain) {
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
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithShape(F32, {42}),
                              "an HloInstruction outputting\n"
                              "  a shape:\n"
                              "   * with element type F32 AND\n"
                              "   * with dimensions [42]",
                              "Shape does not have element type F32\n"
                              "in s32[42]{0}\n"
                              "in output shape\n"
                              "in i = s32[42]{0} iota(), iota_dimension=0");
  EXPECT_DESC_AND_EXPLANATION(iota, m::Op().WithShape(S32, {128}),
                              "an HloInstruction outputting\n"
                              "  a shape:\n"
                              "   * with element type S32 AND\n"
                              "   * with dimensions [128]",
                              "Shape does not have dimensions [128]\n"
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
      "in a = s32[] add(c, c)");
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
      absl::StrCat("an HloInstruction which is 0x", absl::Hex(iota.get()),
                   " (i = s32[42]{0} iota(), iota_dimension=0)"),
      absl::StrCat("HloInstruction 0x", absl::Hex(constant.get()), " is not 0x",
                   absl::Hex(iota.get()),
                   " (i = s32[42]{0} iota(), iota_dimension=0)\n"
                   "in c = s32[] constant(0)"));

  EXPECT_DESC_AND_EXPLANATION(
      SetName("a",
              HloInstruction::CreateBinary(constant->shape(), HloOpcode::kAdd,
                                           constant.get(), constant.get())),
      m::Op().WithOperandIfPresent(0, m::Iota()),  //
      "an HloInstruction either with fewer than 1 operand, or with an operand "
      "0 which is:\n"
      "  an HloInstruction with opcode iota",
      "HloInstruction doesn't have opcode iota\n"
      "in c = s32[] constant(0)\n"
      "in operand 0\n"
      "in a = s32[] add(c, c)");

  EXPECT_DESC_AND_EXPLANATION(
      constant, m::Op().WithPredicate(HloPredicateFalse),
      "an HloInstruction which matches a user-specified predicate",
      "HloInstruction does not match user-specified predicate\n"
      "in c = s32[] constant(0)");
}

TEST_F(PatternMatcherTest, HloInstructionMatcherAnyOrderDescribeTo) {
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
      "matcher. Specifically,\n"
      " - an HloInstruction named \"bar\"\n"
      "does not match LHS:\n"
      " - HloInstruction not named \"bar\"\n"
      "   in b = s32[] constant(0)\n"
      "does not match RHS:\n"
      " - HloInstruction not named \"bar\"\n"
      "   in c = s32[] constant(0)\n"
      "in a = s32[] add(b, c)");

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
      "HloInstruction's LHS operand did not match either of the two matchers. "
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
      "in a = s32[] add(p, c)");
}

TEST_F(PatternMatcherTest, AnyOfMatcherDescribeToAndExplain) {
  EXPECT_DESC_AND_EXPLANATION(
      ShapeUtil::MakeScalarShape(S32),
      m::AnyOf<Shape>(m::Shape().WithRank(1), m::Shape().WithElementType(F32)),
      "any of:\n"
      " - a shape that has 1 dimension OR\n"
      " - a shape with element type F32",
      "None of the following matchers succeeded:\n"
      "Matcher #1\n"
      " - a shape that has 1 dimension\n"
      "failed with\n"
      " - Shape does not have rank 1\n"
      "   in s32[]\n"
      "Matcher #2\n"
      " - a shape with element type F32\n"
      "failed with\n"
      " - Shape does not have element type F32\n"
      "   in s32[]");
}

TEST_F(PatternMatcherTest, Parameter) {
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

TEST_F(PatternMatcherTest, OneUseAndOneUser) {
  auto param =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0");

  EXPECT_FALSE(Match(param.get(), m::Op().WithOneUse()));
  EXPECT_DESC_AND_EXPLANATION(
      param, m::Op().WithOneUse(),
      "an HloInstruction which has exactly one use",
      "HloInstruction has 0 users, but expected exactly one.\n"
      "in p0 = f32[] parameter(0)");

  EXPECT_FALSE(Match(param.get(), m::Op().WithOneUser()));
  EXPECT_DESC_AND_EXPLANATION(
      param, m::Op().WithOneUser(),
      "an HloInstruction which has exactly one user (but possibly is used "
      "multiple times by that instruction)",
      "HloInstruction has 0 users, but expected exactly one.\n"
      "in p0 = f32[] parameter(0)");

  {
    auto reshape =
        SetName("r", HloInstruction::CreateReshape(
                         ShapeUtil::MakeShape(F32, {1}), param.get()));
    EXPECT_TRUE(Match(param.get(), m::Op().WithOneUse()));
    EXPECT_TRUE(Match(param.get(), m::Op().WithOneUser()));

    auto reshape1 =
        SetName("r1", HloInstruction::CreateReshape(
                          ShapeUtil::MakeShape(F32, {1}), param.get()));
    EXPECT_FALSE(Match(param.get(), m::Op().WithOneUse()));
    EXPECT_FALSE(Match(param.get(), m::Op().WithOneUser()));

    const char* kMultipleUserExplanation =
        "HloInstruction has 2 users, but expected exactly one.\n"
        "All users:\n"
        " - r = f32[1]{0} reshape(p0)\n"
        " - r1 = f32[1]{0} reshape(p0)\n"
        "in p0 = f32[] parameter(0)";
    EXPECT_EQ(Explanation(param.get(), m::Op().WithOneUse()),
              kMultipleUserExplanation);
    EXPECT_EQ(Explanation(param.get(), m::Op().WithOneUser()),
              kMultipleUserExplanation);
  }

  auto add = SetName("add", HloInstruction::CreateBinary(
                                ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd,
                                param.get(), param.get()));
  EXPECT_TRUE(Match(param.get(), m::Op().WithOneUser()));
  EXPECT_FALSE(Match(param.get(), m::Op().WithOneUse()));
  EXPECT_EQ(Explanation(param.get(), m::Op().WithOneUse()),
            "HloInstruction is used 2 times by its user, but is expected to be "
            "used just once: add = f32[] add(p0, p0)\n"
            "in p0 = f32[] parameter(0)");
}

TEST_F(PatternMatcherTest, MatchSingleUserOnlyUnaryOpOneUser) {
  auto param =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p");
  auto reshape =
      SetName("reshape", HloInstruction::CreateReshape(
                             ShapeUtil::MakeShape(F32, {1}), param.get()));
  EXPECT_TRUE(MatchSingleUserOnly(reshape.get(), m::Reshape(m::Op())));
  // Equivalent call of Match:
  EXPECT_TRUE(Match(reshape.get(), m::Reshape(m::Op().WithOneUser())));
}

TEST_F(PatternMatcherTest, MatchSingleUserOnlyUnaryOpTwoUsers) {
  auto param =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p");
  auto reshape =
      SetName("reshape", HloInstruction::CreateReshape(
                             ShapeUtil::MakeShape(F32, {1}), param.get()));
  auto bitcast =
      SetName("bitcast", HloInstruction::CreateBitcast(
                             ShapeUtil::MakeShape(F32, {1}), param.get()));
  EXPECT_TRUE(MatchSingleUserOnly(param.get(), m::Op()));
  // Equivalent call of Match:
  EXPECT_TRUE(Match(param.get(), m::Op()));

  EXPECT_TRUE(MatchSingleUserOnly(bitcast.get(), m::Bitcast()));
  EXPECT_TRUE(Match(bitcast.get(), m::Bitcast()));

  EXPECT_FALSE(MatchSingleUserOnly(bitcast.get(), m::Bitcast(m::Op())));
  EXPECT_FALSE(Match(bitcast.get(), m::Bitcast(m::Op().WithOneUser())));
  EXPECT_EQ(Explanation(bitcast.get(), m::Bitcast(m::Op()),
                        /*single_user_only=*/true),
            "Operand 0 of HloInstruction has 2 users. Expected 1.\nin bitcast "
            "= f32[1]{0} bitcast(p)");
}

TEST_F(PatternMatcherTest, MatchSingleUserOnlyBinaryOpOneUser) {
  auto param0 =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto add = SetName("add", HloInstruction::CreateBinary(
                                ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd,
                                param0.get(), param0.get()));
  EXPECT_TRUE(MatchSingleUserOnly(add.get(), m::Add(m::Op(), m::Op())));
  // Equivalent call of Match:
  EXPECT_TRUE(
      Match(add.get(), m::Add(m::Op().WithOneUser(), m::Op().WithOneUser())));
}

TEST_F(PatternMatcherTest, MatchSingleUserOnlyBinaryOpTwoUsers) {
  auto param0 =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto param1 =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p1");
  auto add = SetName("add", HloInstruction::CreateBinary(
                                ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd,
                                param0.get(), param0.get()));
  auto mul =
      SetName("mul", HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {}),
                                                  HloOpcode::kMultiply,
                                                  param1.get(), param0.get()));
  EXPECT_TRUE(MatchSingleUserOnly(mul.get(), m::Multiply()));
  // Equivalent call of Match:
  EXPECT_TRUE(Match(mul.get(), m::Multiply()));

  EXPECT_FALSE(MatchSingleUserOnly(mul.get(), m::Multiply(m::Op(), m::Op())));
  EXPECT_FALSE(Match(
      mul.get(), m::Multiply(m::Op().WithOneUser(), m::Op().WithOneUser())));
  EXPECT_EQ(Explanation(mul.get(), m::Multiply(m::Op(), m::Op()),
                        /*single_user_only=*/true),
            "Operand 1 of HloInstruction has 2 users. Expected 1.\nin mul = "
            "f32[] multiply(p1, p0)");

  EXPECT_FALSE(MatchSingleUserOnly(add.get(), m::Add(m::Op(), m::Op())));
  EXPECT_FALSE(
      Match(add.get(), m::Add(m::Op().WithOneUser(), m::Op().WithOneUser())));
  EXPECT_EQ(Explanation(add.get(), m::Add(m::Op(), m::Op()),
                        /*single_user_only=*/true),
            "Operand 0 of HloInstruction has 2 users. Expected 1.\nin add = "
            "f32[] add(p0, p0)");
}

TEST_F(PatternMatcherTest, MatchSingleUserOnlyBinaryOpTwoUsersLowerLevel) {
  auto param0 =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto param1 =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p1");
  auto add = SetName("add", HloInstruction::CreateBinary(
                                ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd,
                                param0.get(), param0.get()));
  auto mul =
      SetName("mul", HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {}),
                                                  HloOpcode::kMultiply,
                                                  param1.get(), param0.get()));
  auto div = SetName("div", HloInstruction::CreateBinary(
                                ShapeUtil::MakeShape(F32, {}),
                                HloOpcode::kDivide, add.get(), mul.get()));
  EXPECT_TRUE(
      MatchSingleUserOnly(div.get(), m::Divide(m::Add(), m::Multiply())));
  // Equivalent call of Match:
  EXPECT_TRUE(Match(div.get(), m::Divide(m::Add().WithOneUser(),
                                         m::Multiply().WithOneUser())));

  EXPECT_FALSE(MatchSingleUserOnly(
      div.get(), m::Divide(m::Add(m::Op(), m::Op()), m::Multiply())));
  EXPECT_FALSE(Match(
      div.get(),
      m::Divide(
          m::Add(m::Op().WithOneUser(), m::Op().WithOneUser()).WithOneUser(),
          m::Multiply().WithOneUser())));
  EXPECT_EQ(Explanation(add.get(), m::Add(m::Op(), m::Op()),
                        /*single_user_only=*/true),
            "Operand 0 of HloInstruction has 2 users. Expected 1.\nin add = "
            "f32[] add(p0, p0)");
}

TEST_F(PatternMatcherTest, Comparison) {
  auto shape = ShapeUtil::MakeShape(F32, {1});
  auto p0 = HloInstruction::CreateParameter(0, shape, "param.0");
  auto p1 = HloInstruction::CreateParameter(1, shape, "param.1");
  auto eq = HloInstruction::CreateCompare(shape, p0.get(), p1.get(),
                                          ComparisonDirection::kEq);
  auto ne = HloInstruction::CreateCompare(shape, p0.get(), p1.get(),
                                          ComparisonDirection::kNe);
  auto add =
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0.get(), p1.get());
  auto le = HloInstruction::CreateCompare(shape, p0.get(), add.get(),
                                          ComparisonDirection::kLe);

  EXPECT_TRUE(Match(eq.get(), m::Compare()));
  EXPECT_TRUE(Match(eq.get(), m::Eq()));
  EXPECT_TRUE(Match(eq.get(), m::Eq(m::Parameter(0), m::Parameter(1))));
  EXPECT_TRUE(Match(eq.get(), m::EqAnyOrder(m::Parameter(1), m::Parameter(0))));
  EXPECT_TRUE(Match(ne.get(), m::Compare()));
  EXPECT_TRUE(Match(ne.get(), m::Ne()));
  EXPECT_TRUE(Match(
      le.get(),
      m::Compare(m::Parameter(0), m::Add(m::Parameter(0), m::Parameter(1)))));
  EXPECT_TRUE(Match(le.get(), m::Le(m::Parameter(0),
                                    m::Add(m::Parameter(0), m::Parameter(1)))));

  EXPECT_FALSE(Match(eq.get(), m::Add()));
  EXPECT_FALSE(Match(eq.get(), m::Ne()));
  EXPECT_FALSE(
      Match(le.get(),
            m::Eq(m::Parameter(0), m::Add(m::Parameter(0), m::Parameter(1)))));
  EXPECT_FALSE(Match(eq.get(), m::Eq(m::Parameter(1), m::Parameter(0))));
  EXPECT_DESC_AND_EXPLANATION(
      eq, m::Ne().WithOneUser(),
      "an HloInstruction:\n"
      " * with opcode compare AND\n"
      " * which has comparison direction NE AND\n"
      " * which has exactly one user (but possibly is used "
      "multiple times by that instruction)",
      "HloInstruction is not comparison NE\n"
      "in compare = f32[1]{0} compare(param.0, param.1), direction=EQ");
}

TEST_F(PatternMatcherTest, ConvDnums) {
  TF_ASSERT_OK_AND_ASSIGN(ConvolutionDimensionNumbers dnums,
                          ParseConvolutionDimensionNumbers("bf01_oi01->bf01"));
  auto param =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto op = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                             /*operands=*/{},
                                             /*custom_call_target=*/"foo");
  op->set_convolution_dimension_numbers(dnums);

  EXPECT_TRUE(Match(op.get(), m::CustomCall().WithConvDnums(dnums)));
  EXPECT_TRUE(
      Match(op.get(), m::CustomCall().WithConvDnums("bf01_oi01->bf01")));
  TF_ASSERT_OK_AND_ASSIGN(ConvolutionDimensionNumbers different_dnums,
                          ParseConvolutionDimensionNumbers("b01f_oi01->bf01"));
  EXPECT_FALSE(Match(op.get(), m::CustomCall().WithConvDnums(different_dnums)));
  EXPECT_FALSE(
      Match(op.get(), m::CustomCall().WithConvDnums("b01f_oi01->bf01")));
  EXPECT_FALSE(
      Match(param.get(), m::CustomCall().WithConvDnums("b01f_oi01->bf01")));

  EXPECT_DESC_AND_EXPLANATION(
      op.get(), m::CustomCall().WithConvDnums("b01f_oi01->bf01"),
      "an HloInstruction:\n"
      " * with opcode custom-call AND\n"
      " * which has convolution dimension numbers b01f_oi01->bf01",
      "convolution_dimension_numbers bf01_oi01->bf01 don't match expected "
      "b01f_oi01->bf01\n"
      "in custom-call = f32[] custom-call(), dim_labels=bf01_oi01->bf01, "
      "custom_call_target=\"foo\"");
}

TEST_F(PatternMatcherTest, CustomCallMatchers) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module

    ENTRY test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT out = f32[] custom-call(p0, p1), custom_call_target="test_target"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_TRUE(Match(root, m::CustomCall()));
  EXPECT_TRUE(Match(root, m::CustomCall({"test_target"})));
  EXPECT_TRUE(Match(
      root, m::CustomCall({"test_target"}, m::Parameter(0), m::Parameter(1))));

  EXPECT_TRUE(Match(root, m::CustomCall({"test_target", "other_target"})));
  EXPECT_TRUE(Match(root, m::CustomCall({"other_target", "test_target"})));
  EXPECT_TRUE(Match(root, m::CustomCall({"test_target", "other_target"},
                                        m::Parameter(0), m::Parameter(1))));
  EXPECT_TRUE(Match(root, m::CustomCall({"other_target", "test_target"},
                                        m::Parameter(0), m::Parameter(1))));

  HloInstruction* instr;
  EXPECT_TRUE(Match(root, m::CustomCall(&instr)));
  EXPECT_TRUE(Match(root, m::CustomCall(&instr, {"test_target"})));
  EXPECT_TRUE(Match(root, m::CustomCall(&instr, {"test_target"},
                                        m::Parameter(0), m::Parameter(1))));

  const HloInstruction* const_instr;
  EXPECT_TRUE(Match(root, m::CustomCall(&const_instr)));
  EXPECT_TRUE(Match(root, m::CustomCall(&const_instr, {"test_target"})));
  EXPECT_TRUE(Match(root, m::CustomCall(&const_instr, {"test_target"},
                                        m::Parameter(0), m::Parameter(1))));

  EXPECT_FALSE(Match(root, m::CustomCall({"other_target"})));
  EXPECT_FALSE(Match(root, m::CustomCall({"other_target", "other_target2"})));
  EXPECT_FALSE(Match(
      root, m::CustomCall({"test_target"}, m::Parameter(1), m::Parameter(0))));
}

TEST_F(PatternMatcherTest, SharedSubpatternPreservesTheSemantics) {
  auto scalar0 = m::SharedSubpattern(m::ConstantScalar(0));
  auto pattern0 = m::AnyOf<HloInstruction>(m::Convert(scalar0), scalar0);

  auto scalar1 = m::SharedSubpattern(m::ConstantScalar(1));
  auto pattern1 = m::AnyOf<HloInstruction>(m::Convert(scalar1), scalar1);

  {
    constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test {
      ROOT constant = f16[] constant(0)
    })";
    TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                            ParseAndReturnVerifiedModule(kModuleStr));
    auto* root = hlo_module->entry_computation()->root_instruction();

    EXPECT_TRUE(Match(root, pattern0));
    EXPECT_FALSE(Match(root, pattern1));
  }

  {
    constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test {
      constant = f16[] constant(0)
      ROOT convert = f32[] convert(constant)
    })";
    TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                            ParseAndReturnVerifiedModule(kModuleStr));
    auto* root = hlo_module->entry_computation()->root_instruction();

    EXPECT_TRUE(Match(root, pattern0));
    EXPECT_FALSE(Match(root, pattern1));
  }
}

TEST_F(PatternMatcherTest, SharedSubpatternCanBeNested) {
  auto scalar0 = m::SharedSubpattern(match::ConstantScalar(0));
  auto subpattern0 = m::SharedSubpattern(
      m::AnyOf<HloInstruction>(m::Convert(scalar0), scalar0));
  auto pattern0 =
      m::AnyOf<HloInstruction>(m::Convert(subpattern0), subpattern0);

  auto scalar1 = m::SharedSubpattern(match::ConstantScalar(1));
  auto subpattern1 = m::SharedSubpattern(
      m::AnyOf<HloInstruction>(m::Convert(scalar1), scalar1));
  auto pattern1 =
      m::AnyOf<HloInstruction>(m::Convert(subpattern1), subpattern1);

  {
    constexpr char kModuleStr[] = R"(
    HloModule test_module ENTRY test {
      constant = f16[] constant(0)
      convert = f32[] convert(constant)
      ROOT convert1 = f32[] convert(convert)
    })";
    TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                            ParseAndReturnVerifiedModule(kModuleStr));
    auto* root = hlo_module->entry_computation()->root_instruction();

    EXPECT_TRUE(Match(root, pattern0));
    EXPECT_FALSE(Match(root, pattern1));
  }
}

TEST_F(PatternMatcherTest, TestWithContractingDims) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      %param1 = f32[2048,1024] parameter(0)
      %param2 = f32[1024,33708] parameter(1)
      ROOT %dot1 = f32[2048,33708]{1,0} dot(f32[2048,1024]{1,0} %param1,
                f32[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Dot().WithContractingDims({1}, {0})));
  EXPECT_FALSE(Match(root, m::Dot().WithContractingDims({0}, {1})));
  EXPECT_FALSE(Match(root, m::Dot().WithContractingDims({1}, {0, 1})));
  EXPECT_DESC_AND_EXPLANATION(
      root, m::Dot().WithContractingDims({1}, {0, 1}),
      "an HloInstruction:\n"
      " * with opcode dot AND\n"
      " * with lhs_contracting_dims {1} and rhs_contracting_dims {0,1}",
      "rhs_contracting_dimensions {0} don't match expected {0,1}\n"
      "in dot1 = f32[2048,33708]{1,0} dot(param1, param2), "
      "lhs_contracting_dims={1}, rhs_contracting_dims={0}");
}

TEST_F(PatternMatcherTest, TestWithReplicaGroups) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY test {
      input = f32[128,32]{0,1} parameter(0)
      ROOT all-reduce = f32[128,32]{0,1} all-reduce(input),
                        replica_groups={{0,1},{2,3}}, to_apply=add
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::AllReduce().WithReplicaGroups({{0, 1}, {2, 3}})));
  EXPECT_FALSE(Match(root, m::AllReduce().WithReplicaGroups({{}, {}})));
  EXPECT_FALSE(Match(root, m::AllReduce().WithReplicaGroups({{1, 0}, {3, 2}})));
  EXPECT_DESC_AND_EXPLANATION(
      root, m::AllReduce().WithReplicaGroups({{1, 0}, {3, 2}}),
      "an HloInstruction:\n"
      " * with opcode all-reduce AND\n"
      " * with replica_group {{1,0},{3,2}}",
      "replica_group {{0,1},{2,3}} don't match expected with replica_group "
      "{{1,0},{3,2}}\n"
      "in all-reduce = f32[128,32]{0,1} all-reduce(input), "
      "replica_groups={{0,1},{2,3}}, to_apply=add");
}

TEST_F(PatternMatcherTest, TestWithSharding) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      p0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
        sharding={devices=[1,2,2,1]0,1,2,3},
        metadata={op_name="test"}
      ROOT copy = f32[5,7,11,13]{3,2,1,0} copy(p0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* instruction = FindInstruction(hlo_module.get(), "p0");
  EXPECT_TRUE(
      Match(instruction, m::Op().WithSharding("{devices=[1,2,2,1]0,1,2,3}")));
  EXPECT_FALSE(
      Match(instruction, m::Op().WithSharding("{devices=[2,2,1,1]0,1,2,3}")));
  EXPECT_DESC_AND_EXPLANATION(
      instruction, m::Op().WithSharding("{devices=[2,2,1,1]0,1,2,3}"),
      "an HloInstruction with sharding {devices=[2,2,1,1]0,1,2,3}",
      "sharding {devices=[1,2,2,1]0,1,2,3} don't match expected "
      "{devices=[2,2,1,1]0,1,2,3}\n"
      "in p0 = f32[5,7,11,13]{3,2,1,0} parameter(0), "
      "sharding={devices=[1,2,2,1]0,1,2,3}");
}

TEST_F(PatternMatcherTest, TestWithControlDeps) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add = f32[4] add(p0, p1)
      mul = f32[4] multiply(p0, p1), control-predecessors={add}
      div = f32[4] divide(p0, p1), control-predecessors={mul}
      ROOT t = (f32[4], f32[4], f32[4]) tuple(add, mul, div)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* add = FindInstruction(hlo_module.get(), "add");
  auto* mul = FindInstruction(hlo_module.get(), "mul");
  auto* div = FindInstruction(hlo_module.get(), "div");

  EXPECT_TRUE(Match(add, m::Op().WithControlDeps({}, {mul})));
  EXPECT_TRUE(Match(mul, m::Op().WithControlDeps({add}, {div})));
  EXPECT_TRUE(Match(div, m::Op().WithControlDeps({mul}, {})));
  EXPECT_FALSE(Match(div, m::Op().WithControlDeps({mul}, {div})));
  EXPECT_DESC_AND_EXPLANATION(
      div, m::Op().WithControlDeps({mul}, {div}),
      "an HloInstruction with control predecessors {mul} and control "
      "successors {div}",
      "HloInstruction expected to have control successors {div} but has {}\n"
      "in div = f32[4]{0} divide(p0, p1), control-predecessors={mul}");
}

}  // namespace
}  // namespace xla
