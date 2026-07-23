/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tests/constraint_propagator.h"

#include <cmath>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tests/constraint_state.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class ConstraintPropagatorTest : public HloTestBase {};

TEST_F(ConstraintPropagatorTest, EmptyInterval) {
  ConstraintInterval a{0.0, 10.0, false};
  ConstraintInterval b{20.0, 30.0, false};
  ConstraintInterval c = a.Intersect(b);
  EXPECT_TRUE(c.IsEmpty());
}

TEST_F(ConstraintPropagatorTest, IntervalIsEmpty) {
  EXPECT_TRUE((ConstraintInterval{1.0, 0.0, false}).IsEmpty());
  EXPECT_TRUE((ConstraintInterval{0.0, 0.0, true}).IsEmpty());
  EXPECT_FALSE((ConstraintInterval{0.0, 0.0, false}).IsEmpty());
  EXPECT_FALSE((ConstraintInterval{-1.0, 1.0, false}).IsEmpty());
  EXPECT_FALSE((ConstraintInterval{-1.0, 1.0, true}).IsEmpty());
}

TEST_F(ConstraintPropagatorTest, IntervalSign) {
  EXPECT_TRUE(ConstraintInterval::Positive().IsPositive());
  EXPECT_FALSE(ConstraintInterval::Positive().IsPositiveStrict());
  EXPECT_TRUE(ConstraintInterval::StrictPositive().IsPositive());
  EXPECT_TRUE(ConstraintInterval::StrictPositive().IsPositiveStrict());
  EXPECT_FALSE(ConstraintInterval::StrictPositive().IsNegative());

  EXPECT_TRUE(ConstraintInterval::Negative().IsNegative());
  EXPECT_FALSE(ConstraintInterval::Negative().IsNegativeStrict());
  EXPECT_TRUE(ConstraintInterval::StrictNegative().IsNegative());
  EXPECT_TRUE(ConstraintInterval::StrictNegative().IsNegativeStrict());
  EXPECT_FALSE(ConstraintInterval::StrictNegative().IsPositive());

  ConstraintInterval zero_only{0.0, 0.0, false};
  EXPECT_TRUE(zero_only.IsPositive());
  EXPECT_FALSE(zero_only.IsPositiveStrict());
  EXPECT_TRUE(zero_only.IsNegative());
  EXPECT_FALSE(zero_only.IsNegativeStrict());

  ConstraintInterval positive_exclude_zero{0.0, 1.0, true};
  EXPECT_TRUE(positive_exclude_zero.IsPositive());
  EXPECT_TRUE(positive_exclude_zero.IsPositiveStrict());

  ConstraintInterval negative_exclude_zero{-1.0, 0.0, true};
  EXPECT_TRUE(negative_exclude_zero.IsNegative());
  EXPECT_TRUE(negative_exclude_zero.IsNegativeStrict());
}

TEST_F(ConstraintPropagatorTest, IntervalCrossesZero) {
  EXPECT_TRUE((ConstraintInterval{-1.0, 1.0, false}).CrossesZero());
  EXPECT_TRUE((ConstraintInterval{-1.0, 1.0, true}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{0.0, 1.0, false}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{0.0, 1.0, true}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{-1.0, 0.0, false}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{-1.0, 0.0, true}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{1.0, 2.0, false}).CrossesZero());
  EXPECT_FALSE((ConstraintInterval{-2.0, -1.0, false}).CrossesZero());
  EXPECT_FALSE(ConstraintInterval::StrictPositive().CrossesZero());
  EXPECT_FALSE(ConstraintInterval::Positive().CrossesZero());
  EXPECT_FALSE(ConstraintInterval::StrictNegative().CrossesZero());
  EXPECT_FALSE(ConstraintInterval::Negative().CrossesZero());
}

TEST_F(ConstraintPropagatorTest, IntervalIntersect) {
  ConstraintInterval a{0.0, 10.0, false};
  ConstraintInterval b{5.0, 15.0, false};
  EXPECT_EQ(a.Intersect(b), (ConstraintInterval{5.0, 10.0, false}));
  EXPECT_EQ(b.Intersect(a), (ConstraintInterval{5.0, 10.0, false}));

  ConstraintInterval e{0.0, 10.0, false};
  ConstraintInterval f{10.0, 20.0, false};
  EXPECT_EQ(e.Intersect(f), (ConstraintInterval{10.0, 10.0, false}));
  EXPECT_EQ(f.Intersect(e), (ConstraintInterval{10.0, 10.0, false}));

  ConstraintInterval g{-1.0, 1.0, false};
  ConstraintInterval h = ConstraintInterval::NonZero();
  EXPECT_EQ(g.Intersect(h), (ConstraintInterval{-1.0, 1.0, true}));

  ConstraintInterval i{-1.0, 1.0, false};
  ConstraintInterval j = ConstraintInterval::Positive();
  EXPECT_EQ(i.Intersect(j), (ConstraintInterval{0.0, 1.0, false}));

  ConstraintInterval k = ConstraintInterval::StrictPositive();
  ConstraintInterval l = ConstraintInterval::Negative();
  EXPECT_TRUE(k.Intersect(l).IsEmpty());

  ConstraintInterval zero_only{0.0, 0.0, false};
  ConstraintInterval non_zero = ConstraintInterval::NonZero();
  EXPECT_TRUE(zero_only.Intersect(non_zero).IsEmpty());

  ConstraintInterval o{-1.0, 1.0, true};
  ConstraintInterval p{0.0, 1.0, false};
  EXPECT_EQ(o.Intersect(p), (ConstraintInterval{0.0, 1.0, true}));

  EXPECT_TRUE(
      zero_only.Intersect(ConstraintInterval::StrictPositive()).IsEmpty());
  EXPECT_EQ(zero_only.Intersect(ConstraintInterval::Positive()), zero_only);
}

TEST_F(ConstraintPropagatorTest, SubtractPositive) {
  // Sqrt(Sub(x, y)): Sub(x,y) >= 0 => x >=0 and y<=0
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  sub = f32[] subtract(x, y)
  ROOT root = f32[] sqrt(sub)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto x_int = states[module->entry_computation()->parameter_instruction(0)]
                   .GetConstraintInterval();
  auto y_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();

  EXPECT_TRUE(x_int.IsPositive());
  EXPECT_FALSE(x_int.exclude_zero);
  EXPECT_TRUE(y_int.IsNegative());
  EXPECT_FALSE(y_int.exclude_zero);
}

TEST_F(ConstraintPropagatorTest, SubtractNegative) {
  // Sqrt(Sub(x, y)): Sub(x,y) <= 0 => x <0 and y>0
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  sub = f32[] subtract(x, y)
  negx = f32[] negate(sub)
  ROOT root = f32[] sqrt(negx)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto x_int = states[module->entry_computation()->parameter_instruction(0)]
                   .GetConstraintInterval();
  auto y_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();

  EXPECT_TRUE(x_int.IsNegative());
  EXPECT_FALSE(x_int.exclude_zero);
  EXPECT_TRUE(y_int.IsPositive());
  EXPECT_FALSE(y_int.exclude_zero);
}

TEST_F(ConstraintPropagatorTest, RsqrtMulForcedNegative) {
  // Rsqrt(Mul(x, y)) with x < 0 (forced via Log(-x))
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  negx = f32[] negate(x)
  log = f32[] log(negx)
  mul = f32[] multiply(x, y)
  rsqrt = f32[] rsqrt(mul)
  ROOT root = (f32[], f32[]) tuple(log, rsqrt)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto x_int = states[module->entry_computation()->parameter_instruction(0)]
                   .GetConstraintInterval();
  auto y_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();

  // log(negx) => negx > 0 => x < 0.
  // rsqrt(mul) => mul > 0. Since x < 0, y must be < 0.
  EXPECT_TRUE(x_int.IsNegativeStrict());
  EXPECT_TRUE(y_int.IsNegativeStrict());
}

TEST_F(ConstraintPropagatorTest, NestedDivExcludeZero) {
  // Div(a, Div(b, c))
  // c != 0, and Div(b,c) != 0 => b != 0
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] parameter(2)
  div_bc = f32[] divide(b, c)
  div_a = f32[] divide(a, div_bc)
  ROOT root = f32[] log(div_a) 
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto b_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();
  auto c_int = states[module->entry_computation()->parameter_instruction(2)]
                   .GetConstraintInterval();

  EXPECT_TRUE(b_int.exclude_zero);
  EXPECT_TRUE(c_int.exclude_zero);
}

TEST_F(ConstraintPropagatorTest, DivAddZeroCrossing) {
  // Div(x, Add(y, z)) -> Add(y,z) != 0 -> force same signs to avoid random
  // zero-crossing
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  z = f32[] parameter(2)
  add = f32[] add(y, z)
  ROOT root = f32[] divide(x, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto y_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();
  auto z_int = states[module->entry_computation()->parameter_instruction(2)]
                   .GetConstraintInterval();

  // add != 0 => y and z cant both be 0 and must have the same sign to avoid
  // zero-crossing.
  EXPECT_EQ(y_int.exclude_zero, true);
  EXPECT_EQ(z_int.exclude_zero, true);
  EXPECT_EQ(y_int.IsPositive(), true);
  EXPECT_EQ(z_int.IsPositive(), true);
}

TEST_F(ConstraintPropagatorTest, RsqrtAdd) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[2048,4] parameter(0)
  constant_1 = f32[] constant(0.00390625)
  broadcast_1 = f32[2048,4] broadcast(constant_1), dimensions={}
  mul = f32[2048,4] multiply(param_0, broadcast_1)
  constant_2 = f32[] constant(1e-06)
  broadcast_2 = f32[2048,4] broadcast(constant_2), dimensions={}
  add = f32[2048,4] add(mul, broadcast_2)
  ROOT rsqrt = f32[2048,4] rsqrt(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));
  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();
  EXPECT_TRUE(p0_int.IsPositive());
}

TEST_F(ConstraintPropagatorTest, RsqrtAddBitcast) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[2048,4] parameter(0)
  constant_1 = f32[] constant(0.00390625)
  broadcast_1 = f32[2048,4] broadcast(constant_1), dimensions={}
  mul = f32[2048,4] multiply(param_0, broadcast_1)
  constant_2 = f32[] constant(1e-06)
  broadcast_2 = f32[2048,4] broadcast(constant_2), dimensions={}
  add = f32[2048,4] add(mul, broadcast_2)
  bitcast_1 = f32[1,1,2048,4] bitcast(add)
  rsqrt = f32[1,1,2048,4] rsqrt(bitcast_1)
  ROOT bitcast_2 = f32[2048,4] bitcast(rsqrt)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();

  EXPECT_TRUE(p0_int.IsPositive());
}

TEST_F(ConstraintPropagatorTest, SqrtDivMax) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[256]{0} parameter(0)
  slice_0 = f32[4]{0} slice(param_0), slice={[161:165]}
  param_1 = s32[256]{0} parameter(1)
  slice_1 = s32[4]{0} slice(param_1), slice={[83:87]}
  constant_1 = s32[] constant(1)
  broadcast_1 = s32[4]{0} broadcast(constant_1), dimensions={}
  max = s32[4]{0} maximum(slice_1, broadcast_1)
  convert = f32[4]{0} convert(max)
  div = f32[4]{0} divide(slice_0, convert)
  ROOT sqrt = f32[4]{0} sqrt(div)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();

  // sqrt(div) => div >= 0.
  // div = slice_0 / convert => slice_0 >= 0, convert > 0.
  // slice_0 is part of param_0, so param_0 should be constrained to >= 0.
  EXPECT_GE(p0_int.min, 0.0);
  EXPECT_FALSE(p0_int.IsEmpty());
}

TEST_F(ConstraintPropagatorTest, TheoreticalLimitationConflictingConstraints) {
  // Test Log(x) vs Log(-Add(x,y))
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
  neg = f32[] negate(add)
  log1 = f32[] log(x)
  log2 = f32[] log(neg)
  ROOT root = (f32[], f32[]) tuple(log1, log2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto x_int = states[module->entry_computation()->parameter_instruction(0)]
                   .GetConstraintInterval();
  auto y_int = states[module->entry_computation()->parameter_instruction(1)]
                   .GetConstraintInterval();
  // log(x) => x > 0
  // log(-Add(x,y)) => -Add(x,y) > 0 => Add(x,y) < 0.
  // Since Add(x,y) < 0, we infer x < 0 and y < 0. This conflicts with x > 0
  // from log(x), so the ConstraintInterval for x is empty.
  EXPECT_TRUE(x_int.IsEmpty());
  EXPECT_TRUE(y_int.IsNegativeStrict());
}

TEST_F(ConstraintPropagatorTest, MultiplySquareReverseConstraintShapes) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = bf16[8,1024,3072] parameter(0)
  ROOT mul = bf16[8,1024,3072] multiply(param_0, param_0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();

  // Root bf16 output is seeded with [-65504.0, 65504.0].
  // Reverse propagation for multiply computes max_in = sqrt(65504.0) ~
  // 255.9375.
  EXPECT_FALSE(p0_int.IsEmpty());
  EXPECT_NEAR(p0_int.max, std::sqrt(65504.0), 1e-3);
  EXPECT_NEAR(p0_int.min, -std::sqrt(65504.0), 1e-3);
}

TEST_F(ConstraintPropagatorTest, ConvolutionReverseConstraintShapes) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = bf16[8,1024,3072,1] parameter(0)
  param_1 = bf16[24,3072,256,1] parameter(1)
  ROOT conv = bf16[8,1024,24,256] convolution(param_0, param_1),
    window={size=1x24 pad=0_0x23_23 rhs_reversal=0x1}, dim_labels=0bf1_1io0->0b1f
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();
  auto p1_int = states[module->entry_computation()->parameter_instruction(1)]
                    .GetConstraintInterval();

  // Contracting size K = 3072 * 1 * 24 = 73728 terms.
  // Root max_out = 65504.0.
  // max_in = sqrt(65504.0 / 73728) ~ 0.9426.
  double expected_max_in = std::sqrt(65504.0 / 73728.0);
  EXPECT_FALSE(p0_int.IsEmpty());
  EXPECT_NEAR(p0_int.max, expected_max_in, 1e-3);
  EXPECT_NEAR(p0_int.min, -expected_max_in, 1e-3);

  EXPECT_FALSE(p1_int.IsEmpty());
  EXPECT_NEAR(p1_int.max, expected_max_in, 1e-3);
  EXPECT_NEAR(p1_int.min, -expected_max_in, 1e-3);
}

TEST_F(ConstraintPropagatorTest, ReduceSumReverseConstraintShapes) {
  const char* hlo = R"(
HloModule TestModule
add_computation {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}
ENTRY main {
  param_0 = bf16[8,1024,24,256] parameter(0)
  init = bf16[] constant(0)
  ROOT reduce = bf16[8,1024] reduce(param_0, init), dimensions={2,3},
    to_apply=add_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();

  // Reduce sum across 24 * 256 = 6144 elements.
  // Root max_out = 65504.0.
  // max_in = 65504.0 / 6144 ~ 10.661458.
  double expected_max_in = 65504.0 / 6144.0;
  EXPECT_FALSE(p0_int.IsEmpty());
  EXPECT_NEAR(p0_int.max, expected_max_in, 1e-3);
  EXPECT_NEAR(p0_int.min, -expected_max_in, 1e-3);
}

TEST_F(ConstraintPropagatorTest, DotReverseConstraintShapes) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  lhs = bf16[8,1024,3072] parameter(0)
  rhs = bf16[3072,256] parameter(1)
  ROOT dot = bf16[8,1024,256] dot(lhs, rhs),
    lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto lhs_int = states[module->entry_computation()->parameter_instruction(0)]
                     .GetConstraintInterval();
  auto rhs_int = states[module->entry_computation()->parameter_instruction(1)]
                     .GetConstraintInterval();

  // Contracting size K = 3072 terms.
  // Root max_out = 65504.0.
  // max_in = sqrt(65504.0 / 3072) ~ 4.61735.
  double expected_max_in = std::sqrt(65504.0 / 3072.0);
  EXPECT_FALSE(lhs_int.IsEmpty());
  EXPECT_NEAR(lhs_int.max, expected_max_in, 1e-3);
  EXPECT_NEAR(lhs_int.min, -expected_max_in, 1e-3);

  EXPECT_FALSE(rhs_int.IsEmpty());
  EXPECT_NEAR(rhs_int.max, expected_max_in, 1e-3);
  EXPECT_NEAR(rhs_int.min, -expected_max_in, 1e-3);
}

TEST_F(ConstraintPropagatorTest, MultiplyReduceFusionSequence) {
  const char* hlo = R"(
HloModule TestModule
add_computation {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}
ENTRY main {
  param_0 = bf16[8,1024,3072,1] parameter(0)
  param_1 = bf16[24,3072,256,1] parameter(1)
  conv = bf16[8,1024,24,256] convolution(param_0, param_1),
    window={size=1x24 pad=0_0x23_23 rhs_reversal=0x1}, dim_labels=0bf1_1io0->0b1f
  square = bf16[8,1024,24,256] multiply(conv, conv)
  init = bf16[] constant(0)
  ROOT reduce = bf16[8,1024] reduce(square, init), dimensions={2,3},
    to_apply=add_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(auto states, ConstraintPropagator::Run(*module));

  auto p0_int = states[module->entry_computation()->parameter_instruction(0)]
                    .GetConstraintInterval();
  auto p1_int = states[module->entry_computation()->parameter_instruction(1)]
                    .GetConstraintInterval();

  // Sequence trace:
  // 1. reduce max_out = 65504.0 -> square bound max_in = 65504 / 6144
  // ~ 10.661458.
  // 2. square (conv * conv) -> conv bound max_in = sqrt(10.661458) ~ 3.265188.
  // 3. conv (K = 73728) -> param_0 and param_1 bound max_in = sqrt(3.265188 /
  // 73728) ~ 0.006654.
  double expected_max_in = std::sqrt(std::sqrt(65504.0 / 6144.0) / 73728.0);
  EXPECT_FALSE(p0_int.IsEmpty());
  EXPECT_NEAR(p0_int.max, expected_max_in, 1e-4);
  EXPECT_NEAR(p0_int.min, -expected_max_in, 1e-4);

  EXPECT_FALSE(p1_int.IsEmpty());
  EXPECT_NEAR(p1_int.max, expected_max_in, 1e-4);
  EXPECT_NEAR(p1_int.min, -expected_max_in, 1e-4);
}

}  // namespace
}  // namespace xla
