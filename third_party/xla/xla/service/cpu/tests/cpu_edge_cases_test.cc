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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class CpuEdgeCasesTest : public HloTestBase {};

// 1. Reversed convolution (rhs_reversal=1x0)
TEST_F(CpuEdgeCasesTest, ReversedConvolution) {
  const std::string hlo_text = R"(
HloModule module

ENTRY entry {
  lhs = f32[1,3,3,1]{3,2,1,0} constant({{{{1}, {2}, {3}}, {{4}, {5}, {6}}, {{7}, {8}, {9}}}})
  rhs = f32[2,2,1,1]{3,2,1,0} constant({{{{1}}, {{2}}}, {{{3}}, {{4}}}})
  ROOT conv = f32[1,2,2,1]{3,2,1,0} convolution(lhs, rhs), window={size=2x2 rhs_reversal=1x0}, dim_labels=b01f_01io->b01f
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result.shape(), ShapeUtil::MakeShape(F32, {1, 2, 2, 1}));
  absl::Span<const float> data = result.data<float>();
  ASSERT_EQ(data.size(), 4);
  EXPECT_EQ(data[0], 25.0f);
  EXPECT_EQ(data[1], 35.0f);
  EXPECT_EQ(data[2], 55.0f);
  EXPECT_EQ(data[3], 65.0f);
}

// 2. Non-canonical dimension numbers in a while body convolution
TEST_F(CpuEdgeCasesTest, WhileBodyNonCanonicalConv) {
  const std::string hlo_text = R"(
HloModule module

body {
  p = (s32[], f32[1,3,3,1]) parameter(0)
  iter = s32[] get-tuple-element(p), index=0
  input = f32[1,3,3,1] get-tuple-element(p), index=1
  one = s32[] constant(1)
  next_iter = s32[] add(iter, one)
  filter = f32[1,2,2,1]{3,2,1,0} constant({{{{1.0}, {2.0}}, {{3.0}, {4.0}}}})
  conv = f32[2,2,1,1]{0,1,3,2} convolution(input, filter), window={size=2x2}, dim_labels=f01b_i01o->01bf
  pad_conv = f32[3,3,1,1]{3,2,1,0} pad(conv, f32[] constant(0.0)), padding=0_1x0_1x0_0x0_0
  conv_reshaped = f32[1,3,3,1]{3,2,1,0} reshape(pad_conv)
  ROOT root = (s32[], f32[1,3,3,1]) tuple(next_iter, conv_reshaped)
}

cond {
  p = (s32[], f32[1,3,3,1]) parameter(0)
  iter = s32[] get-tuple-element(p), index=0
  max_iter = s32[] constant(2)
  ROOT cmp = pred[] compare(iter, max_iter), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  input = f32[1,3,3,1] constant({{{{1.0}, {2.0}, {3.0}}, {{4.0}, {5.0}, {6.0}}, {{7.0}, {8.0}, {9.0}}}})
  init = (s32[], f32[1,3,3,1]) tuple(zero, input)
  ROOT while_loop = (s32[], f32[1,3,3,1]) while(init), condition=cond, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  EXPECT_TRUE(result.shape().IsTuple());
  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 2);
  EXPECT_EQ(elements[0].data<int32_t>()[0], 2);
  absl::Span<const float> loop_out = elements[1].data<float>();
  ASSERT_EQ(loop_out.size(), 9);
}

// 3. Convolution with 4 spatial dimensions
TEST_F(CpuEdgeCasesTest, FourSpatialDimensionsConv) {
  const std::string hlo_text = R"(
HloModule module

ENTRY entry {
  c = f32[] constant(1.0)
  lhs = f32[1,2,2,2,2,1]{5,4,3,2,1,0} broadcast(c), dimensions={}
  rhs = f32[2,2,2,2,1,1]{5,4,3,2,1,0} broadcast(c), dimensions={}
  ROOT conv = f32[1,1,1,1,1,1]{5,4,3,2,1,0} convolution(lhs, rhs), window={size=2x2x2x2}, dim_labels=b0123f_0123oi->b0123f
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result.data<float>()[0], 16.0f);
}

// 4. Non-monotonic layout convolution / copy
TEST_F(CpuEdgeCasesTest, NonMonotonicLayoutCopyAndConv) {
  const std::string copy_hlo = R"(
HloModule module

ENTRY entry {
  in = u8[2,3,4]{0,2,1} constant({{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}})
  ROOT copy = u8[2,3,4]{1,0,2} copy(in)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module_copy,
                       ParseAndReturnVerifiedModule(copy_hlo));
  ASSERT_OK_AND_ASSIGN(const Literal result_copy,
                       Execute(std::move(module_copy), {}));
  absl::Span<const uint8_t> copy_data = result_copy.data<uint8_t>();
  ASSERT_EQ(copy_data.size(), 24);
  EXPECT_EQ(copy_data[0], 1);
  EXPECT_EQ(copy_data[1], 5);
  EXPECT_EQ(copy_data[23], 24);

  const std::string conv_hlo = R"(
HloModule module

ENTRY entry {
  lhs = f32[1,3,3,1]{0,2,1,3} constant({{{{1}, {2}, {3}}, {{4}, {5}, {6}}, {{7}, {8}, {9}}}})
  rhs = f32[2,2,1,1]{2,0,3,1} constant({{{{1}}, {{2}}}, {{{3}}, {{4}}}})
  ROOT conv = f32[1,2,2,1]{1,3,0,2} convolution(lhs, rhs), window={size=2x2}, dim_labels=b01f_01oi->b01f
}
)";

  ASSERT_OK_AND_ASSIGN(auto module_conv,
                       ParseAndReturnVerifiedModule(conv_hlo));
  ASSERT_OK_AND_ASSIGN(const Literal result_conv,
                       Execute(std::move(module_conv), {}));
  EXPECT_EQ(result_conv.shape(), ShapeUtil::MakeShapeWithDenseLayout(
                                     F32, {1, 2, 2, 1}, {1, 3, 0, 2}));
  absl::Span<const float> conv_data = result_conv.data<float>();
  ASSERT_EQ(conv_data.size(), 4);
  EXPECT_EQ(conv_data[0], 37.0f);
  EXPECT_EQ(conv_data[1], 67.0f);
  EXPECT_EQ(conv_data[2], 47.0f);
  EXPECT_EQ(conv_data[3], 77.0f);
}

// 5. S4 copy
TEST_F(CpuEdgeCasesTest, S4Copy) {
  const std::string hlo_text = R"(
HloModule module

ENTRY entry {
  in = s4[4,4]{1,0:E(4)} constant({{0, 1, 2, 3}, {4, 5, 6, 7}, {-8, -7, -6, -5}, {-4, -3, -2, -1}})
  copy = s4[4,4]{1,0:E(4)} copy(in)
  ROOT out = s32[4,4]{1,0} convert(copy)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  absl::Span<const int32_t> data = result.data<int32_t>();
  ASSERT_EQ(data.size(), 16);
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 1);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 3);
  EXPECT_EQ(data[4], 4);
  EXPECT_EQ(data[5], 5);
  EXPECT_EQ(data[6], 6);
  EXPECT_EQ(data[7], 7);
  EXPECT_EQ(data[8], -8);
  EXPECT_EQ(data[9], -7);
  EXPECT_EQ(data[10], -6);
  EXPECT_EQ(data[11], -5);
  EXPECT_EQ(data[12], -4);
  EXPECT_EQ(data[13], -3);
  EXPECT_EQ(data[14], -2);
  EXPECT_EQ(data[15], -1);
}

// 6. Mismatched-layout concatenate values
TEST_F(CpuEdgeCasesTest, MismatchedLayoutConcatenate) {
  const std::string hlo_text = R"(
HloModule module

ENTRY entry {
  p0 = s32[2,3]{1,0} constant({{1, 2, 3}, {4, 5, 6}})
  p1 = s32[2,3]{0,1} constant({{7, 8, 9}, {10, 11, 12}})
  ROOT concat = s32[4,3]{1,0} concatenate(p0, p1), dimensions={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  absl::Span<const int32_t> data = result.data<int32_t>();
  ASSERT_EQ(data.size(), 12);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 4);
  EXPECT_EQ(data[4], 5);
  EXPECT_EQ(data[5], 6);
  EXPECT_EQ(data[6], 7);
  EXPECT_EQ(data[7], 8);
  EXPECT_EQ(data[8], 9);
  EXPECT_EQ(data[9], 10);
  EXPECT_EQ(data[10], 11);
  EXPECT_EQ(data[11], 12);
}

// Regression tests for https://github.com/openxla/xla/issues/47370: in-place
// ops and loops inside a sort comparator used to crash the legacy IR emitter.
// Each comparator computes a value known to be true through the in-place op
// and returns xor(p0 >= p1, true), i.e. an ascending order.
class SortComparatorTest : public HloTestBase {
 protected:
  void RunAndExpectSorted(absl::string_view hlo_text) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
    Literal input = LiteralUtil::CreateR1<int32_t>({3, 1, 4, 1, 5, 9, 2});
    ASSERT_OK_AND_ASSIGN(const Literal result,
                         Execute(std::move(module), {&input}));
    EXPECT_EQ(result, LiteralUtil::CreateR1<int32_t>({1, 1, 2, 3, 4, 5, 9}));
  }
};

TEST_F(SortComparatorTest, ScatterInSortComparator) {
  RunAndExpectSorted(R"(
HloModule scatter_in_sort_comparator

overwrite {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  indices = s32[2,1] constant({{0}, {1}})
  updates = f32[2] constant({5, 2})
  scattered = f32[3] scatter(operand, indices, updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=overwrite
  a = f32[1] slice(scattered), slice={[0:1]}
  b = f32[1] slice(scattered), slice={[1:2]}
  a_scalar = f32[] reshape(a)
  b_scalar = f32[] reshape(b)
  flip = pred[] compare(a_scalar, b_scalar), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)");
}

TEST_F(SortComparatorTest, VariadicScatterInSortComparator) {
  RunAndExpectSorted(R"(
HloModule variadic_scatter_in_sort_comparator

overwrite2 {
  a0 = f32[] parameter(0)
  a1 = f32[] parameter(1)
  upd0 = f32[] parameter(2)
  upd1 = f32[] parameter(3)
  ROOT t = (f32[], f32[]) tuple(upd0, upd1)
}

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  indices = s32[2,1] constant({{0}, {1}})
  updates0 = f32[2] constant({5, 2})
  updates1 = f32[2] constant({1, 9})
  scattered = (f32[3], f32[3]) scatter(operand, operand, indices, updates0, updates1), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=overwrite2
  out1 = f32[3] get-tuple-element(scattered), index=1
  a = f32[1] slice(out1), slice={[0:1]}
  b = f32[1] slice(out1), slice={[1:2]}
  a_scalar = f32[] reshape(a)
  b_scalar = f32[] reshape(b)
  flip = pred[] compare(b_scalar, a_scalar), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)");
}

TEST_F(SortComparatorTest, DynamicUpdateSliceInSortComparator) {
  RunAndExpectSorted(R"(
HloModule dus_in_sort_comparator

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  update = f32[1] constant({7})
  zero = s32[] constant(0)
  two = s32[] constant(2)
  idx = s32[] clamp(zero, p0, two)
  updated = f32[3] dynamic-update-slice(operand, update, idx)
  picked = f32[1] dynamic-slice(updated, idx), dynamic_slice_sizes={1}
  picked_scalar = f32[] reshape(picked)
  three = f32[] constant(3)
  flip = pred[] compare(picked_scalar, three), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)");
}

TEST_F(SortComparatorTest, WhileInSortComparator) {
  RunAndExpectSorted(R"(
HloModule while_in_sort_comparator

body {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  buffer = f32[3] get-tuple-element(state), index=1
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  ten = s32[] constant(10)
  value = s32[] add(i, ten)
  value_f = f32[] convert(value)
  update = f32[1] reshape(value_f)
  next_buffer = f32[3] dynamic-update-slice(buffer, update, i)
  ROOT next_state = (s32[], f32[3]) tuple(next_i, next_buffer)
}

condition {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  three = s32[] constant(3)
  ROOT continue = pred[] compare(i, three), direction=LT
}

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  zero = s32[] constant(0)
  init = (s32[], f32[3]) tuple(zero, operand)
  loop = (s32[], f32[3]) while(init), condition=condition, body=body
  buffer_out = f32[3] get-tuple-element(loop), index=1
  a = f32[1] slice(buffer_out), slice={[0:1]}
  b = f32[1] slice(buffer_out), slice={[1:2]}
  a_scalar = f32[] reshape(a)
  b_scalar = f32[] reshape(b)
  flip = pred[] compare(b_scalar, a_scalar), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)");
}

// Unrolling the loop moves the scatter into the comparator, where it has to
// be expanded (and the resulting loop unrolled) in turn.
TEST_F(SortComparatorTest, ScatterInWhileInSortComparator) {
  RunAndExpectSorted(R"(
HloModule scatter_in_while_in_sort_comparator

overwrite {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

body {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  buffer = f32[3] get-tuple-element(state), index=1
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  indices = s32[2,1] constant({{0}, {1}})
  updates = f32[2] constant({5, 2})
  next_buffer = f32[3] scatter(buffer, indices, updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=overwrite
  ROOT next_state = (s32[], f32[3]) tuple(next_i, next_buffer)
}

condition {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  two = s32[] constant(2)
  ROOT continue = pred[] compare(i, two), direction=LT
}

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  zero = s32[] constant(0)
  init = (s32[], f32[3]) tuple(zero, operand)
  loop = (s32[], f32[3]) while(init), condition=condition, body=body
  buffer_out = f32[3] get-tuple-element(loop), index=1
  a = f32[1] slice(buffer_out), slice={[0:1]}
  b = f32[1] slice(buffer_out), slice={[1:2]}
  a_scalar = f32[] reshape(a)
  b_scalar = f32[] reshape(b)
  flip = pred[] compare(a_scalar, b_scalar), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)");
}

// A scatter with more indices than the unroll threshold stays a loop, which
// the nested emitter must reject with a status instead of a CHECK failure.
TEST_F(SortComparatorTest, LargeScatterInSortComparatorIsUnimplemented) {
  constexpr absl::string_view kHloText = R"(
HloModule large_scatter_in_sort_comparator

overwrite {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  c = f32[] convert(p1)
  operand = f32[3] broadcast(c), dimensions={}
  indices = s32[70,1] iota(), iota_dimension=0
  updates = f32[70] broadcast(c), dimensions={}
  scattered = f32[3] scatter(operand, indices, updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=overwrite
  a = f32[1] slice(scattered), slice={[0:1]}
  a_scalar = f32[] reshape(a)
  flip = pred[] compare(a_scalar, c), direction=GT
  ge = pred[] compare(p0, p1), direction=GE
  ROOT result = pred[] xor(ge, flip)
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, is_stable=true, to_apply=compare
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  Literal input = LiteralUtil::CreateR1<int32_t>({3, 1, 4, 1, 5, 9, 2});
  EXPECT_EQ(Execute(std::move(module), {&input}).status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace xla::cpu
