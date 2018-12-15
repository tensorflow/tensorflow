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

#include "tensorflow/compiler/xla/literal.h"

#include <vector>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class LiteralUtilTest : public ::testing::Test {
 protected:
  LiteralUtilTest() {
    Array4D<float> arr4d({
        // clang-format off
      {  // i0=0
          {  // i1=0
              {1, 2, 3},  // i2=0
              {4, 5, 6},  // i2=1
              {7, 8, 9},  // i2=2
          },
          {  // i1=1
              {11, 12, 13},
              {14, 15, 16},
              {17, 18, 19},
          },
      },
      {  // i0=1
          {  // i1=0
              {101, 102, 103},
              {104, 105, 106},
              {107, 108, 109},
          },
          {  // i1=1
              {201, 202, 203},  // i2=0
              {204, 205, 206},  // i2=1
              {207, 208, 209},  // i2=2
          },
      },
        // clang-format on
    });

    layout_r2_dim0major_ = LayoutUtil::MakeLayout({1, 0});
    layout_r2_dim0minor_ = LayoutUtil::MakeLayout({0, 1});
    layout_r3_dim0major_ = LayoutUtil::MakeLayout({2, 1, 0});
    layout_r3_dim0minor_ = LayoutUtil::MakeLayout({0, 1, 2});
    layout_r4_dim0major_ = LayoutUtil::MakeLayout({3, 2, 1, 0});
    layout_r4_dim0minor_ = LayoutUtil::MakeLayout({0, 1, 2, 3});

    literal_r4_2x2x3x3_dim0major_ =
        LiteralUtil::CreateR4FromArray4DWithLayout<float>(arr4d,
                                                          layout_r4_dim0major_);
    literal_r4_2x2x3x3_dim0minor_ =
        LiteralUtil::CreateR4FromArray4DWithLayout<float>(arr4d,
                                                          layout_r4_dim0minor_);
  }

  Layout layout_r2_dim0major_;
  Layout layout_r2_dim0minor_;
  Layout layout_r3_dim0major_;
  Layout layout_r3_dim0minor_;
  Layout layout_r4_dim0major_;
  Layout layout_r4_dim0minor_;
  Literal literal_r4_2x2x3x3_dim0major_;
  Literal literal_r4_2x2x3x3_dim0minor_;
};

TEST_F(LiteralUtilTest, LiteralScalarToString) {
  auto true_lit = LiteralUtil::CreateR0<bool>(true);
  EXPECT_EQ("pred[] true", true_lit.ToString());

  auto false_lit = LiteralUtil::CreateR0<bool>(false);
  EXPECT_EQ("pred[] false", false_lit.ToString());

  auto u32_lit = LiteralUtil::CreateR0<uint32>(42);
  EXPECT_EQ("u32[] 42", u32_lit.ToString());

  auto s32_lit = LiteralUtil::CreateR0<int32>(-999);
  EXPECT_EQ("s32[] -999", s32_lit.ToString());

  auto f32_lit = LiteralUtil::CreateR0<float>(3.14f);
  EXPECT_EQ("f32[] 3.14", f32_lit.ToString());

  auto f16_lit = LiteralUtil::CreateR0<half>(static_cast<half>(0.5f));
  EXPECT_EQ("f16[] 0.5", f16_lit.ToString());

  auto c64_lit = LiteralUtil::CreateR0<complex64>({3.14f, 2.78f});
  EXPECT_EQ("c64[] (3.14, 2.78)", c64_lit.ToString());

  auto bf16_lit = LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(0.5f));
  EXPECT_EQ("bf16[] 0.5", bf16_lit.ToString());

  // 3.14 will be rounded to 3.14062 in bfloat16 format.
  auto bf16_lit_truncated =
      LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(3.14f));
  ASSERT_EQ("bf16[] 3.14062", bf16_lit_truncated.ToString());

  auto bf16_lit_truncated2 =
      LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(9.001f));
  EXPECT_EQ("bf16[] 9", bf16_lit_truncated2.ToString());
}

TEST_F(LiteralUtilTest, LiteralVectorToString) {
  auto pred_vec = LiteralUtil::CreateR1<bool>({true, false, true});
  EXPECT_EQ("pred[3] {1, 0, 1}", pred_vec.ToString());
}

TEST_F(LiteralUtilTest, R2ToString) {
  const auto literal = LiteralUtil::CreateR2({{1, 2}, {3, 4}, {5, 6}});
  const string expected = R"(s32[3,2] {
  { 1, 2 },
  { 3, 4 },
  { 5, 6 }
})";
  EXPECT_EQ(expected, literal.ToString());
}

TEST_F(LiteralUtilTest, R3ToString) {
  const auto literal =
      LiteralUtil::CreateR3({{{1}, {2}}, {{3}, {4}}, {{5}, {6}}});
  const string expected = R"(s32[3,2,1] {
{
  {1},
  {2}
},
{
  {3},
  {4}
},
{
  {5},
  {6}
}
})";
  EXPECT_EQ(expected, literal.ToString());
}

TEST_F(LiteralUtilTest, R6ToString) {
  const auto literal =
      LiteralUtil::CreateFromDimensions(S32, {2, 2, 1, 1, 1, 2});
  const string expected = R"(s32[2,2,1,1,1,2] {
{ /*i0=0*/
{ /*i1=0*/
{ /*i2=0*/
{ /*i3=0*/
  { 0, 0 }
}
}
},
{ /*i1=1*/
{ /*i2=0*/
{ /*i3=0*/
  { 0, 0 }
}
}
}
},
{ /*i0=1*/
{ /*i1=0*/
{ /*i2=0*/
{ /*i3=0*/
  { 0, 0 }
}
}
},
{ /*i1=1*/
{ /*i2=0*/
{ /*i3=0*/
  { 0, 0 }
}
}
}
}
})";
  EXPECT_EQ(expected, literal.ToString());
}

TEST_F(LiteralUtilTest, TupleToString) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = LiteralUtil::MakeTuple({&scalar, &matrix});
  const string expected = R"((
f32[] 1,
f32[2,2] {
  { 1, 2 },
  { 3, 4 }
}
))";
  EXPECT_EQ(expected, tuple.ToString());
}

TEST_F(LiteralUtilTest, CreateR3FromArray3d) {
  // clang-format off
  Array3D<float> array_3d({
    {{1.0f, 2.0f},
     {3.0f, 4.0f},
     {5.0f, 6.0f}},
    {{7.0f, 8.0f},
     {9.0f, 10.0f},
     {11.0f, 12.0f}},
  });
  // clang-format on

  auto literal = LiteralUtil::CreateR3FromArray3D(array_3d);
  EXPECT_THAT(literal.shape().dimensions(), ElementsAre(2, 3, 2));
  string result = literal.ToString();
  const string expected = R"(f32[2,3,2] {
{
  { 1, 2 },
  { 3, 4 },
  { 5, 6 }
},
{
  { 7, 8 },
  { 9, 10 },
  { 11, 12 }
}
})";
  EXPECT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, CreateSparse) {
  std::vector<int64> dimensions = {8, 8, 8};
  Array2D<int64> indices = {
      {3, 4, 5},
      {1, 2, 3},
      {2, 3, 4},
      {3, 5, 6},
  };
  std::vector<int64> values = {7, 8, 9, 10};
  auto literal = LiteralUtil::CreateSparse<int64>(
      dimensions, SparseIndexArray(indices.n1() + 3, indices), values);

  Array2D<int64> expected_indices = {
      {1, 2, 3},
      {2, 3, 4},
      {3, 4, 5},
      {3, 5, 6},
  };
  std::vector<int64> expected_values = {8, 9, 7, 10};

  EXPECT_EQ(literal.sparse_indices()->data(),
            absl::Span<const int64>(expected_indices.data(),
                                    expected_indices.num_elements()));
  EXPECT_EQ(literal.data<int64>(), absl::Span<const int64>(expected_values));

  // Serialize then deserialize and verify the resulting literal.
  TF_ASSERT_OK_AND_ASSIGN(Literal literal_from_proto,
                          Literal::CreateFromProto(literal.ToProto()));

  EXPECT_EQ(literal_from_proto.sparse_indices()->data(),
            absl::Span<const int64>(expected_indices.data(),
                                    expected_indices.num_elements()));
  EXPECT_EQ(literal_from_proto.data<int64>(),
            absl::Span<const int64>(expected_values));
}

TEST_F(LiteralUtilTest, LiteralR4F32ProjectedStringifies) {
  // clang-format off
  auto literal = LiteralUtil::CreateR4Projected<float>({
    {1, 2},
    {1001, 1002},
    {2001, 2002},
  }, /*projection_p=*/1, /*projection_z=*/2);
  // clang-format on
  EXPECT_THAT(literal.shape().dimensions(), ElementsAre(1, 2, 3, 2));
  string result = literal.ToString();
  const string expected = R"(f32[1,2,3,2] {
{ /*i0=0*/
{ /*i1=0*/
  { 1, 2 },
  { 1001, 1002 },
  { 2001, 2002 }
},
{ /*i1=1*/
  { 1, 2 },
  { 1001, 1002 },
  { 2001, 2002 }
}
}
})";
  EXPECT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, LiteralR4F32Stringifies) {
  EXPECT_THAT(literal_r4_2x2x3x3_dim0major_.shape().dimensions(),
              ElementsAre(2, 2, 3, 3));
  string result = literal_r4_2x2x3x3_dim0major_.ToString();
  const string expected = R"(f32[2,2,3,3] {
{ /*i0=0*/
{ /*i1=0*/
  { 1, 2, 3 },
  { 4, 5, 6 },
  { 7, 8, 9 }
},
{ /*i1=1*/
  { 11, 12, 13 },
  { 14, 15, 16 },
  { 17, 18, 19 }
}
},
{ /*i0=1*/
{ /*i1=0*/
  { 101, 102, 103 },
  { 104, 105, 106 },
  { 107, 108, 109 }
},
{ /*i1=1*/
  { 201, 202, 203 },
  { 204, 205, 206 },
  { 207, 208, 209 }
}
}
})";
  EXPECT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, EachCellR2F32) {
  // clang-format off
  auto literal = LiteralUtil::CreateR2<float>({
    {3.1f, 4.2f},
    {9.3f, 12.4f},
  });
  // clang-format on
  std::vector<std::tuple<int64, int64, string>> seen;
  literal.EachCellAsString(
      [&seen](absl::Span<const int64> indices, const string& value) {
        seen.emplace_back(indices[0], indices[1], value);
      });

  using Elem = std::tuple<int64, int64, string>;
  std::vector<Elem> expected = {Elem(0, 0, "3.1"), Elem(0, 1, "4.2"),
                                Elem(1, 0, "9.3"), Elem(1, 1, "12.4")};
  EXPECT_EQ(expected, seen);
}

TEST_F(LiteralUtilTest, ScalarEquality) {
  // Test equality with scalars.
  auto f32_42 = LiteralUtil::CreateR0<float>(42.0);
  auto f32_42_clone = LiteralUtil::CreateR0<float>(42.0);

  EXPECT_EQ(f32_42, f32_42);
  EXPECT_EQ(f32_42, f32_42_clone);

  auto f32_123 = LiteralUtil::CreateR0<float>(123.0);
  EXPECT_NE(f32_42, f32_123);

  auto f64_42 = LiteralUtil::CreateR0<double>(42.0);
  EXPECT_NE(f32_42, f64_42);
}

TEST_F(LiteralUtilTest, NonScalarEquality) {
  // Test equality with nonscalars.
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_clone = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_different =
      LiteralUtil::CreateR2<float>({{4.0, 3.0}, {1.0, 2.0}});
  auto vector_literal = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  Literal nil(ShapeUtil::MakeNil());

  EXPECT_EQ(matrix, matrix);
  EXPECT_EQ(matrix, matrix_clone);
  EXPECT_NE(matrix, matrix_different);
  EXPECT_NE(matrix, vector_literal);
  EXPECT_NE(matrix, scalar);
  EXPECT_NE(matrix, nil);
  EXPECT_EQ(nil, nil);
}

TEST_F(LiteralUtilTest, TokenEquality) {
  auto token0 = LiteralUtil::CreateToken();
  auto token1 = LiteralUtil::CreateToken();
  auto scalar = LiteralUtil::CreateR0<float>(1.0);

  EXPECT_EQ(token0, token1);
  EXPECT_NE(token0, scalar);

  EXPECT_EQ(LiteralUtil::MakeTuple({&token0}),
            LiteralUtil::MakeTuple({&token0}));
  EXPECT_EQ(LiteralUtil::MakeTuple({&token0, &scalar}),
            LiteralUtil::MakeTuple({&token1, &scalar}));
  EXPECT_NE(LiteralUtil::MakeTuple({&token0, &scalar}),
            LiteralUtil::MakeTuple({&scalar, &token1}));
}

TEST_F(LiteralUtilTest, DifferentLayoutEquality) {
  // Test equality with literals which have different layouts.
  Literal colmajor(ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {0, 1}));
  colmajor.Set<float>({0, 0}, 1.0);
  colmajor.Set<float>({0, 1}, 2.0);
  colmajor.Set<float>({1, 0}, 3.0);
  colmajor.Set<float>({1, 1}, 4.0);

  Literal rowmajor(ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {1, 0}));
  rowmajor.Set<float>({0, 0}, 1.0);
  rowmajor.Set<float>({0, 1}, 2.0);
  rowmajor.Set<float>({1, 0}, 3.0);
  rowmajor.Set<float>({1, 1}, 4.0);

  EXPECT_EQ(rowmajor, colmajor);
}

TEST_F(LiteralUtilTest, TupleEquality) {
  // Test equality with tuples.
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple1 = LiteralUtil::MakeTuple({&scalar, &matrix});

  // Tuple with the same elements. One element is shared with the original
  // tuple, the other is a clone of the element in the original tuple.
  auto scalar_clone = LiteralUtil::CreateR0<float>(1.0);
  auto tuple2 = LiteralUtil::MakeTuple({&scalar_clone, &matrix});
  EXPECT_EQ(tuple1, tuple2);

  // Tuple with elements reversed.
  auto reversed_tuple = LiteralUtil::MakeTuple({&matrix, &scalar});
  EXPECT_NE(tuple1, reversed_tuple);

  // Tuple with different value.
  auto scalar_42 = LiteralUtil::CreateR0<float>(42.0);
  auto different_tuple = LiteralUtil::MakeTuple({&scalar_42, &matrix});
  EXPECT_NE(tuple1, different_tuple);
}

TEST_F(LiteralUtilTest, C64Equality) {
  // Test equality with tuples.
  auto vector = LiteralUtil::CreateR1<complex64>({{1.0, 2.0}, {3.0, 4.0}});

  // Tuple with the same elements. One element is shared with the original
  // tuple, the other is a clone of the element in the original tuple.
  auto vector_clone =
      LiteralUtil::CreateR1<complex64>({{1.0, 2.0}, {3.0, 4.0}});
  EXPECT_EQ(vector, vector_clone);

  auto vector_reversed =
      LiteralUtil::CreateR1<complex64>({{3.0, 4.0}, {1.0, 2.0}});
  EXPECT_NE(vector, vector_reversed);
}

TEST_F(LiteralUtilTest, IsAllTuple) {
  auto element1 = LiteralUtil::CreateR0<float>(0.0);
  auto element2 = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});
  auto tuple = LiteralUtil::MakeTuple({&element1, &element1});

  // Tuples should always return false for IsAll.
  EXPECT_FALSE(tuple.IsAll(0));
  EXPECT_FALSE(tuple.IsAll(1));
}

// Verifies that CreateFromShape works for tuples.
TEST_F(LiteralUtilTest, CreateFromShapeTuple) {
  auto scalar = LiteralUtil::CreateR0<float>(0.0);
  auto matrix = LiteralUtil::CreateR2<int32>({{0, 0}, {0, 0}});
  auto tuple = LiteralUtil::MakeTuple({&scalar, &matrix});

  auto x = Literal::CreateFromShape(tuple.shape());
  EXPECT_EQ(tuple, x);
}

TEST_F(LiteralUtilTest, IsAll) {
  EXPECT_TRUE(LiteralUtil::CreateR0<bool>(false).IsAll(0));
  EXPECT_TRUE(LiteralUtil::CreateR0<bool>(true).IsAll(1));
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(false).IsAll(1));
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(false).IsAll(2));
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(true).IsAll(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(true).IsAll(2));
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(true).IsAll(-1));

  // We shouldn't reinterpret int8_min as an unsigned type and then decide that
  // it is equal to 255.
  auto int8_min = std::numeric_limits<int8>::min();
  EXPECT_FALSE(LiteralUtil::CreateR0<uint8>(255).IsAll(int8_min));

  EXPECT_TRUE(LiteralUtil::CreateR0<float>(42.0).IsAll(42));
  EXPECT_FALSE(LiteralUtil::CreateR0<float>(42.0001).IsAll(42));

  EXPECT_TRUE(LiteralUtil::CreateR1<int>({100, 100, 100}).IsAll(100));
  EXPECT_FALSE(LiteralUtil::CreateR1<double>({100, 100, 100.001}).IsAll(100));

  EXPECT_TRUE(LiteralUtil::CreateR2<uint64>({{8, 8}, {8, 8}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<uint64>({{8, 8}, {8, 9}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<uint64>({{9, 8}, {8, 8}}).IsAll(8));

  half h8(8.0f);
  half h9(9.0f);
  EXPECT_TRUE(LiteralUtil::CreateR2<half>({{h8}, {h8}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<half>({{h8}, {h9}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<half>({{h9}, {h8}}).IsAll(8));

  bfloat16 b8(8.0f);
  bfloat16 b9(9.0f);

  EXPECT_TRUE(LiteralUtil::CreateR2<bfloat16>({{b8}, {b8}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<bfloat16>({{b8}, {b9}}).IsAll(8));
  EXPECT_FALSE(LiteralUtil::CreateR2<bfloat16>({{b9}, {b8}}).IsAll(8));

  // 9.001 will be truncated to 9.0
  bfloat16 b91(9.001f);
  bfloat16 b90(9.00f);
  EXPECT_TRUE(LiteralUtil::CreateR2<bfloat16>({{b91}, {b90}}).IsAll(9.0));

  complex64 c8_9 = {8, 9};
  EXPECT_FALSE(LiteralUtil::CreateR2<complex64>({{c8_9}, {c8_9}}).IsAll(8));

  auto uint64_max = std::numeric_limits<uint64>::max();
  EXPECT_FALSE(LiteralUtil::CreateR2<uint64>(
                   {{uint64_max, uint64_max}, {uint64_max, uint64_max}})
                   .IsAll(-1));
}

TEST_F(LiteralUtilTest, IsAllFloat) {
  // IsAllFloat always returns false when the literal is not floating-point.
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(false).IsAllFloat(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<int8>(0).IsAllFloat(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<uint8>(0).IsAllFloat(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<int>(0).IsAllFloat(0));

  EXPECT_TRUE(LiteralUtil::CreateR0<float>(0).IsAllFloat(0));
  EXPECT_TRUE(LiteralUtil::CreateR0<float>(.5).IsAllFloat(.5));
  EXPECT_TRUE(LiteralUtil::CreateR0<float>(-.5).IsAllFloat(-.5));
  EXPECT_FALSE(LiteralUtil::CreateR0<float>(-.5).IsAllFloat(-.49));
  EXPECT_FALSE(
      LiteralUtil::CreateR2<float>({{0, 0, 0}, {0, .1, 0}}).IsAllFloat(0));
  EXPECT_TRUE(LiteralUtil::CreateR2<float>({{.5, .5, .5}, {.5, .5, .5}})
                  .IsAllFloat(.5));

  EXPECT_TRUE(LiteralUtil::CreateR0<double>(0).IsAllFloat(0));
  EXPECT_TRUE(LiteralUtil::CreateR0<double>(.5).IsAllFloat(.5));
  EXPECT_TRUE(LiteralUtil::CreateR0<double>(-.5).IsAllFloat(-.5));
  EXPECT_FALSE(LiteralUtil::CreateR0<double>(-.5).IsAllFloat(-.49));
  EXPECT_FALSE(
      LiteralUtil::CreateR2<double>({{0, 0, 0}, {0, .1, 0}}).IsAllFloat(0));
}

TEST_F(LiteralUtilTest, IsAllComplex) {
  // IsAllComplex always returns false when the literal is not complex.
  EXPECT_FALSE(LiteralUtil::CreateR0<bool>(false).IsAllComplex(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<int8>(0).IsAllComplex(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<uint8>(0).IsAllComplex(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<int>(0).IsAllComplex(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<float>(0).IsAllComplex(0));
  EXPECT_FALSE(LiteralUtil::CreateR0<double>(0).IsAllComplex(0));

  complex64 c8_9 = {8, 9};
  complex64 c7_9 = {7, 9};
  EXPECT_TRUE(LiteralUtil::CreateR2<complex64>({{c8_9}, {c8_9}})
                  .IsAllComplex({8.0f, 9.0f}));
  EXPECT_FALSE(LiteralUtil::CreateR2<complex64>({{c7_9}, {c8_9}})
                   .IsAllComplex({8.0f, 9.0f}));
  EXPECT_FALSE(LiteralUtil::CreateR2<complex64>({{c8_9}, {c7_9}})
                   .IsAllComplex({8.0f, 9.0f}));
}

TEST_F(LiteralUtilTest, IsAllFirst) {
  // IsAllComplex always returns false when the literal is not complex.
  EXPECT_FALSE(LiteralUtil::CreateR1<bool>({false, true}).IsAllFirst());
  EXPECT_TRUE(LiteralUtil::CreateR1<bool>({false, false}).IsAllFirst());
  EXPECT_FALSE(LiteralUtil::CreateR1<int8>({1, 1, 2}).IsAllFirst());
  EXPECT_TRUE(LiteralUtil::CreateR1<int8>({5, 5, 5, 5}).IsAllFirst());
  EXPECT_FALSE(LiteralUtil::CreateR1<uint8>({1, 1, 2}).IsAllFirst());
  EXPECT_TRUE(LiteralUtil::CreateR1<int32>({5, 5, 5, 5}).IsAllFirst());
  EXPECT_FALSE(LiteralUtil::CreateR1<int32>({1, 1, 2}).IsAllFirst());
  EXPECT_TRUE(LiteralUtil::CreateR1<uint32>({5, 5, 5, 5}).IsAllFirst());
  EXPECT_FALSE(LiteralUtil::CreateR1<uint32>({1, 1, 2}).IsAllFirst());

  complex64 c8_9 = {8, 9};
  complex64 c7_9 = {7, 9};
  EXPECT_TRUE(LiteralUtil::CreateR2<complex64>({{c8_9}, {c8_9}}).IsAllFirst());
  EXPECT_FALSE(LiteralUtil::CreateR2<complex64>({{c7_9}, {c8_9}}).IsAllFirst());
}

TEST_F(LiteralUtilTest, IsZero) {
  auto scalar_zero = LiteralUtil::CreateR0<float>(0.0f);
  auto scalar_one = LiteralUtil::CreateR0<float>(1.0f);
  EXPECT_TRUE(scalar_zero.IsZero({}));
  EXPECT_FALSE(scalar_one.IsZero({}));

  auto array = LiteralUtil::CreateR2<uint32>({{1, 2, 0, 3}, {1, 0, 1, 2}});
  EXPECT_FALSE(array.IsZero({0, 1}));
  EXPECT_TRUE(array.IsZero({0, 2}));
  EXPECT_TRUE(array.IsZero({1, 1}));
  EXPECT_FALSE(array.IsZero({1, 2}));

  auto complex_zero = LiteralUtil::CreateR0<complex64>(0.0f);
  auto complex_nonzero = LiteralUtil::CreateR0<complex64>(0.5f);
  EXPECT_TRUE(complex_zero.IsZero({}));
  EXPECT_FALSE(complex_nonzero.IsZero({}));
}

template <typename T>
class LiteralUtilTestTemplated : public ::testing::Test {};

using TestedTypes = ::testing::Types<float, int32, uint32, complex64>;
TYPED_TEST_CASE(LiteralUtilTestTemplated, TestedTypes);

TYPED_TEST(LiteralUtilTestTemplated, Relayout2x2) {
  // Make a non-integer for floating point types.
  TypeParam half = TypeParam(1) / TypeParam(2);
  auto data = LiteralUtil::CreateR2<TypeParam>({{half, 2}, {3, 4}});
  const Layout layout01 = LayoutUtil::MakeLayout({0, 1});
  const Layout layout10 = LayoutUtil::MakeLayout({1, 0});

  auto data01 = data.Relayout(layout01);
  EXPECT_TRUE(LayoutUtil::Equal(data01.shape().layout(), layout01));
  EXPECT_EQ(data, data01);

  auto data10 = data.Relayout(layout10);
  EXPECT_TRUE(LayoutUtil::Equal(data10.shape().layout(), layout10));
  EXPECT_EQ(data, data10);
}

TEST_F(LiteralUtilTest, ReshapeR0) {
  auto original = LiteralUtil::CreateR0<float>(1.7f);
  auto reshape = original.Reshape(/*dimensions=*/{}).ConsumeValueOrDie();
  EXPECT_EQ(original, reshape);
}

TEST_F(LiteralUtilTest, ReshapeR4) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = LiteralUtil::CreateR4WithLayout<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  // F32[1x3x4x2]
  auto expected = LiteralUtil::CreateR3WithLayout<float>({
    {{10, 11}, {12, 13}, {14, 15}, {16, 17}},
    {{18, 19}, {20, 21}, {22, 23}, {24, 25}},
    {{26, 27}, {28, 29}, {30, 31}, {32, 33}},
  }, layout_r3_dim0major_);
  // clang-format on
  auto reshape = original.Reshape({3, 4, 2}).ConsumeValueOrDie();

  EXPECT_EQ(expected, reshape);
}

TEST_F(LiteralUtilTest, ReshapeR4Dim0Minor) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = LiteralUtil::CreateR4WithLayout<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0minor_);
  // F32[1x3x4x2]
  auto expected = LiteralUtil::CreateR3WithLayout<float>({
    {{10, 11}, {12, 13}, {14, 15}, {16, 17}},
    {{18, 19}, {20, 21}, {22, 23}, {24, 25}},
    {{26, 27}, {28, 29}, {30, 31}, {32, 33}},
  }, layout_r3_dim0major_);
  // clang-format on
  auto reshape = original.Reshape({3, 4, 2}).ConsumeValueOrDie();

  EXPECT_EQ(expected, reshape);
}

TEST_F(LiteralUtilTest, TransposeR0) {
  auto original = LiteralUtil::CreateR0<float>(1.7f);
  auto reshape = original.Transpose(/*permutation=*/{});
  EXPECT_EQ(original, reshape);
}

TEST_F(LiteralUtilTest, TransposeR4) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = LiteralUtil::CreateR4<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }});
  // clang-format on
  auto reshape = original.Transpose(/*permutation=*/{2, 3, 0, 1});

  reshape.EachCell<float>([&](absl::Span<const int64> indices, float value) {
    EXPECT_EQ(value, original.Get<float>(
                         {indices[2], indices[3], indices[0], indices[1]}));
  });
}

TEST_F(LiteralUtilTest, TestR4RelayoutEquivalence) {
  // Tests that using Relayout on an array is equivalent to creating it in the
  // target layout in the first place.
  auto dim0minor_relaid_to_dim0major =
      literal_r4_2x2x3x3_dim0minor_.Relayout(layout_r4_dim0major_);
  EXPECT_EQ(literal_r4_2x2x3x3_dim0major_, dim0minor_relaid_to_dim0major);

  auto dim0major_relaid_to_dim0minor =
      literal_r4_2x2x3x3_dim0major_.Relayout(layout_r4_dim0minor_);
  EXPECT_EQ(literal_r4_2x2x3x3_dim0minor_, dim0major_relaid_to_dim0minor);
}

TEST_F(LiteralUtilTest, TestR2LinearLayout) {
  // Test expected memory layout of R2 dim0-minor (column-major) literal.
  auto mat_dim0minor = LiteralUtil::CreateR2WithLayout<int32>(
      {{1, 2, 3}, {4, 5, 6}}, layout_r2_dim0minor_);
  EXPECT_EQ(mat_dim0minor.element_count(), 6);
  EXPECT_THAT(mat_dim0minor.data<int32>(), ElementsAre(1, 4, 2, 5, 3, 6));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_mat_to_dim0major = mat_dim0minor.Relayout(layout_r2_dim0major_);
  EXPECT_THAT(relaid_mat_to_dim0major.data<int32>(),
              ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout of R2 created with dim0-major (row-major).
  auto mat_dim0major = LiteralUtil::CreateR2WithLayout<int32>(
      {{1, 2, 3}, {4, 5, 6}}, layout_r2_dim0major_);
  EXPECT_EQ(mat_dim0major.element_count(), 6);
  EXPECT_THAT(mat_dim0major.data<int32>(), ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_mat_to_dim0minor = mat_dim0major.Relayout(layout_r2_dim0minor_);
  EXPECT_THAT(relaid_mat_to_dim0minor.data<int32>(),
              ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST_F(LiteralUtilTest, TestR3LinearLayout) {
  // Test expected memory layout of R3 dim0-minor (column-major) literal.
  Array3D<int> arr3d(
      // clang-format off
        {
          {
            {1, 2, 3},
            {4, 5, 6},
          },
          {
            {7, 8, 9},
            {10, 11, 12},
          },
      });  // clang-format on
  auto lit_dim0minor = LiteralUtil::CreateR3FromArray3DWithLayout<int>(
      arr3d, layout_r3_dim0minor_);

  EXPECT_EQ(lit_dim0minor.element_count(), 12);
  std::vector<int> expected_dim0minor{1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12};
  EXPECT_THAT(lit_dim0minor.data<int32>(),
              testing::ElementsAreArray(expected_dim0minor));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_lit_to_dim0major = lit_dim0minor.Relayout(layout_r3_dim0major_);
  std::vector<int> expected_dim0major{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_THAT(relaid_lit_to_dim0major.data<int32>(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout of R3 created with dim0-major (row-major).
  auto lit_dim0major = LiteralUtil::CreateR3FromArray3DWithLayout<int>(
      arr3d, layout_r3_dim0major_);
  EXPECT_EQ(lit_dim0major.element_count(), 12);
  EXPECT_THAT(lit_dim0major.data<int32>(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_lit_to_dim0minor = lit_dim0major.Relayout(layout_r3_dim0minor_);
  EXPECT_THAT(relaid_lit_to_dim0minor.data<int32>(),
              testing::ElementsAreArray(expected_dim0minor));
}

TEST_F(LiteralUtilTest, SliceR0S32) {
  auto input = LiteralUtil::CreateR0<int32>(1);
  auto result = input.Slice({}, {});
  EXPECT_EQ(input, result);
}

TEST_F(LiteralUtilTest, SliceR1F32) {
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0, 5.0});
  auto result = input.Slice({3}, {4});
  auto expected = LiteralUtil::CreateR1<float>({4.0});
  EXPECT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, SliceR2U32) {
  auto input_3x4 = LiteralUtil::CreateR2<uint32>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto result = input_3x4.Slice({0, 2}, {2, 4});
  auto expected = LiteralUtil::CreateR2<uint32>({{3, 4}, {7, 8}});
  EXPECT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, SliceR3U32Full) {
  auto input_2x3x2 = LiteralUtil::CreateR3<uint32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  auto result = input_2x3x2.Slice({0, 0, 0}, {2, 3, 2});
  EXPECT_EQ(input_2x3x2, result);
}

TEST_F(LiteralUtilTest, PopulateR1S64) {
  Literal output(ShapeUtil::MakeShape(S64, {1}));
  output.PopulateR1<int64>({77});
  auto expected = LiteralUtil::CreateR1<int64>({77});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateR1U64) {
  Literal output(ShapeUtil::MakeShape(U64, {2}));
  output.PopulateR1<uint64>({{77, 88}});
  auto expected = LiteralUtil::CreateR1<uint64>({{77, 88}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateR1C64) {
  Literal output(ShapeUtil::MakeShape(C64, {1}));
  output.PopulateR1<complex64>({{77, 88}});
  auto expected = LiteralUtil::CreateR1<complex64>({{77, 88}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateR2C64) {
  Literal output(ShapeUtil::MakeShape(C64, {2, 2}));
  output.PopulateR2<complex64>({{{7, 8}, {9, 10}}, {{1, 2}, {3, 4}}});
  auto expected =
      LiteralUtil::CreateR2<complex64>({{{7, 8}, {9, 10}}, {{1, 2}, {3, 4}}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0BF16) {
  Literal output(ShapeUtil::MakeShape(BF16, {}));
  bfloat16 h(0.25f);
  output.PopulateWithValue<bfloat16>(h);
  auto expected = LiteralUtil::CreateR0<bfloat16>(h);
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1BF16) {
  Literal output(ShapeUtil::MakeShape(BF16, {3}));
  bfloat16 h(0.5f);
  output.PopulateWithValue<bfloat16>(h);
  auto expected = LiteralUtil::CreateR1<bfloat16>({h, h, h});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2BF16) {
  Literal output(ShapeUtil::MakeShape(BF16, {2, 2}));
  bfloat16 h(2.0f);
  output.PopulateWithValue<bfloat16>(h);
  auto expected = LiteralUtil::CreateR2<bfloat16>({{h, h}, {h, h}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F32) {
  Literal output(ShapeUtil::MakeShape(F32, {}));
  output.PopulateWithValue<float>(2.5f);
  auto expected = LiteralUtil::CreateR0<float>(2.5f);
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1S64) {
  Literal output(ShapeUtil::MakeShape(S64, {3}));
  output.PopulateWithValue<int64>(-7);
  auto expected = LiteralUtil::CreateR1<int64>({-7, -7, -7});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2U64) {
  Literal output(ShapeUtil::MakeShape(U64, {2, 2}));
  output.PopulateWithValue<uint64>(42);
  auto expected = LiteralUtil::CreateR2<uint64>({{42, 42}, {42, 42}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2C64) {
  Literal output(ShapeUtil::MakeShape(C64, {2, 2}));
  output.PopulateWithValue<complex64>({4, 2});
  auto expected =
      LiteralUtil::CreateR2<complex64>({{{4, 2}, {4, 2}}, {{4, 2}, {4, 2}}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F16) {
  Literal output(ShapeUtil::MakeShape(F16, {}));
  half h(0.25f);
  output.PopulateWithValue<half>(h);
  auto expected = LiteralUtil::CreateR0<half>(h);
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1F16) {
  Literal output(ShapeUtil::MakeShape(F16, {3}));
  half h(0.5f);
  output.PopulateWithValue<half>(h);
  auto expected = LiteralUtil::CreateR1<half>({h, h, h});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2F16) {
  Literal output(ShapeUtil::MakeShape(F16, {2, 2}));
  half h(2.0f);
  output.PopulateWithValue<half>(h);
  auto expected = LiteralUtil::CreateR2<half>({{h, h}, {h, h}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, ReplicateR2U32) {
  auto input = LiteralUtil::CreateR2<uint32>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto output = input.Replicate<uint32>(3);
  auto expected = LiteralUtil::CreateR3<uint32>(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}});
  EXPECT_EQ(output, expected);
}

TEST_F(LiteralUtilTest, CopySliceFrom) {
  const int64 dimensions[] = {17, 15, 34, 21};
  const int64 layouts[][4] = {
      {3, 2, 1, 0}, {0, 2, 1, 3}, {0, 1, 2, 3}, {2, 0, 3, 1}, {1, 3, 0, 2}};
  for (const auto& layout : layouts) {
    Shape shape = ShapeUtil::MakeShapeWithLayout(
        primitive_util::NativeToPrimitiveType<uint32>(), dimensions, layout);

    auto source = Literal::CreateFromShape(shape);
    const int64 zero_base[] = {0, 0, 0, 0};
    const int64 step[] = {1, 1, 1, 1};
    uint32 seqnr = 0;
    auto init_proc = [&](absl::Span<const int64> indexes) {
      source.Set(indexes, ++seqnr);
      return true;
    };
    ShapeUtil::ForEachIndex(source.shape(), zero_base, dimensions, step,
                            init_proc);

    auto blank = Literal::CreateFromShape(shape);
    const int64 src_base[] = {3, 1, 5, 7};
    const int64 dest_base[] = {6, 4, 12, 2};
    const int64 copy_size[] = {7, 8, 11, 9};
    TF_EXPECT_OK(blank.CopySliceFrom(source, src_base, dest_base, copy_size));

    std::vector<int64> source_indexes(TF_ARRAYSIZE(dimensions), 0);
    std::vector<int64> blank_indexes(TF_ARRAYSIZE(dimensions), 0);
    bool matched = true;
    auto check_proc = [&](absl::Span<const int64> indexes) {
      std::copy(indexes.begin(), indexes.end(), source_indexes.begin());
      std::transform(source_indexes.begin(), source_indexes.end(), src_base,
                     source_indexes.begin(), std::plus<int64>());
      std::copy(indexes.begin(), indexes.end(), blank_indexes.begin());
      std::transform(blank_indexes.begin(), blank_indexes.end(), dest_base,
                     blank_indexes.begin(), std::plus<int64>());
      auto bval = blank.Get<uint32>(blank_indexes);
      matched = (bval != 0 && bval == source.Get<uint32>(source_indexes));
      return matched;
    };

    ShapeUtil::ForEachIndex(source.shape(), zero_base, copy_size, step,
                            check_proc);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, CopyFromScalars) {
  auto zero = LiteralUtil::CreateR0<uint32>(0);
  auto nine = LiteralUtil::CreateR0<uint32>(9);
  TF_EXPECT_OK(zero.CopyFrom(nine));
  EXPECT_EQ(zero, nine);

  auto vect = LiteralUtil::CreateR1<uint32>({3, 4, 9, 12, 5, 17, 21});
  TF_EXPECT_OK(zero.CopySliceFrom(vect, {5}, {}, {}));
  EXPECT_EQ(zero.Get<uint32>({}), 17);
  TF_EXPECT_OK(vect.CopySliceFrom(zero, {}, {4}, {}));
  EXPECT_EQ(vect.Get<uint32>({4}), 17);
}

TEST_F(LiteralUtilTest, CopyFromAndToZeroElement) {
  const Shape empty_r1_shape = ShapeUtil::MakeShape(F32, {0});
  const auto const_nine = LiteralUtil::CreateR1<float>({9});
  const auto const_empty = Literal::CreateFromShape(empty_r1_shape);

  {
    // Source contains dimension with zero elements.
    const auto empty = Literal::CreateFromShape(empty_r1_shape);
    auto nine = LiteralUtil::CreateR1<float>({9});

    TF_EXPECT_OK(nine.CopySliceFrom(empty, {0}, {0}, {0}));
    EXPECT_EQ(nine, const_nine);
  }

  {
    // Copy 0 element to destination with zero elements.
    auto empty = Literal::CreateFromShape(empty_r1_shape);
    auto nine = LiteralUtil::CreateR1<float>({9});

    TF_EXPECT_OK(empty.CopySliceFrom(nine, {0}, {0}, {0}));
    EXPECT_EQ(empty, const_empty);
  }
}

TEST_F(LiteralUtilTest, CopyFromNilShape) {
  Literal nil_literal0(ShapeUtil::MakeNil());
  Literal nil_literal1(ShapeUtil::MakeNil());
  // This doesn't actually do any copying, but it should succeed.
  TF_ASSERT_OK(nil_literal0.CopyFrom(nil_literal1));
}

TEST_F(LiteralUtilTest, CopyFromArrays) {
  auto scalar_42 = LiteralUtil::CreateR0<float>(42.0);
  auto scalar_123 = LiteralUtil::CreateR0<float>(123.0);
  EXPECT_NE(scalar_42, scalar_123);
  TF_ASSERT_OK(scalar_42.CopyFrom(scalar_123, /*dest_shape_index=*/{},
                                  /*src_shape_index=*/{}));
  EXPECT_EQ(scalar_42, scalar_123);
  EXPECT_EQ(scalar_42.Get<float>({}), 123.0f);

  auto matrix_1234 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_5678 = LiteralUtil::CreateR2<float>({{5.0, 6.0}, {7.0, 8.0}});
  EXPECT_NE(matrix_1234, matrix_5678);
  EXPECT_EQ(matrix_1234.Get<float>({0, 0}), 1.0f);
  TF_ASSERT_OK(matrix_1234.CopyFrom(matrix_5678, /*dest_shape_index=*/{},
                                    /*src_shape_index=*/{}));
  EXPECT_EQ(matrix_1234, matrix_5678);
  EXPECT_EQ(matrix_1234.Get<float>({0, 0}), 5.0f);
}

TEST_F(LiteralUtilTest, CopyFromTuples) {
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal nil_literal(ShapeUtil::MakeNil());
  Literal inner_elements[] = {LiteralUtil::CreateR0<int32>(42),
                              LiteralUtil::CreateR1<double>({23.0, 44.0})};
  Literal inner_tuple = LiteralUtil::MakeTuple(
      {&inner_elements[0], &inner_elements[1], &nil_literal});
  Literal nested_tuple = LiteralUtil::MakeTuple({&matrix, &inner_tuple});
  // Create a tuple the same shape as the inner tuple of nested_tuple but with
  // different values..
  Literal int32_minus5 = LiteralUtil::CreateR0<int32>(-5);
  Literal double_2_4 = LiteralUtil::CreateR1<double>({2.0, 4.0});
  Literal tuple =
      LiteralUtil::MakeTuple({&int32_minus5, &double_2_4, &nil_literal});

  EXPECT_EQ(matrix, LiteralSlice(nested_tuple, {0}));
  EXPECT_EQ(nested_tuple.Get<int32>({}, {1, 0}), 42);
  EXPECT_EQ(nested_tuple.Get<double>({0}, {1, 1}), 23.0);
  EXPECT_EQ(nested_tuple.Get<double>({1}, {1, 1}), 44.0);

  // Overwrite the inner tuple element of nested_tuple with the contents of
  // 'tuple'.
  TF_ASSERT_OK(nested_tuple.CopyFrom(tuple, /*dest_shape_index=*/{1},
                                     /*src_shape_index=*/{}));

  // The matrix element should be unchanged.
  EXPECT_EQ(matrix, LiteralSlice(nested_tuple, {0}));

  // The tuple element should have been copied from 'tuple'.
  EXPECT_EQ(nested_tuple.Get<int32>({}, {1, 0}), -5);
  EXPECT_EQ(nested_tuple.Get<double>({0}, {1, 1}), 2.0);
  EXPECT_EQ(nested_tuple.Get<double>({1}, {1, 1}), 4.0);
}
TEST_F(LiteralUtilTest, CopyBetweenSameTuple) {
  Literal elements[] = {LiteralUtil::CreateR0<int32>(-2),
                        LiteralUtil::CreateR0<int32>(4)};
  Literal tuple = LiteralUtil::MakeTuple({&elements[0], &elements[1]});

  EXPECT_EQ(tuple.Get<int32>({}, {0}), -2);
  EXPECT_EQ(tuple.Get<int32>({}, {1}), 4);

  // Copy from one element to the other.
  TF_ASSERT_OK(tuple.CopyFrom(tuple, /*dest_shape_index=*/{1},
                              /*src_shape_index=*/{0}));

  EXPECT_EQ(tuple.Get<int32>({}, {0}), -2);
  EXPECT_EQ(tuple.Get<int32>({}, {1}), -2);
}

TEST_F(LiteralUtilTest, CopyFromDifferentShapes) {
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto vector = LiteralUtil::CreateR1<float>({5.0, 7.0});
  Status status = matrix.CopyFrom(vector);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Destination subshape incompatible"));
}

TEST_F(LiteralUtilTest, F16) {
  // Verify that the internal data views are consistent and that they
  // are in little endian format
  // TODO - modify if we make the data format machine endianess dependent
  Literal m1 = Literal::CreateFromShape(ShapeUtil::MakeShape(F16, {2, 2}));
  const char* d1 = reinterpret_cast<const char*>(m1.data<half>().data());
  EXPECT_EQ(d1[0], 0);
  EXPECT_EQ(d1[1], 0);
  EXPECT_EQ(d1[2], 0);
  EXPECT_EQ(d1[3], 0);
  EXPECT_EQ(d1[4], 0);
  EXPECT_EQ(d1[5], 0);
  EXPECT_EQ(d1[6], 0);
  EXPECT_EQ(d1[7], 0);

  half h1(1.0f);
  half h2(2.0f);
  auto m2 = LiteralUtil::CreateR2<half>({{h1, h2}, {h2, h1}});
  const char* d2 = reinterpret_cast<const char*>(m2.data<half>().data());
  EXPECT_EQ(d2[0], 0);
  EXPECT_EQ(d2[1], 0x3C);
  EXPECT_EQ(d2[2], 0);
  EXPECT_EQ(d2[3], 0x40);
  EXPECT_EQ(d2[4], 0);
  EXPECT_EQ(d2[5], 0x40);
  EXPECT_EQ(d2[6], 0);
  EXPECT_EQ(d2[7], 0x3C);
}

TEST_F(LiteralUtilTest, Populate) {
  struct PopulateData {
    std::vector<int64> dimensions;
    std::vector<int64> layout;
  } populate_data[] = {
      {{}, {}},
      {{0}, {0}},
      {{16}, {0}},
      {{2, 0}, {1, 0}},
      {{4, 16}, {1, 0}},
      {{21, 12}, {0, 1}},
      {{6, 11, 17}, {2, 0, 1}},
      {{6, 11, 5, 17}, {3, 2, 0, 1}},
  };
  for (const auto& data : populate_data) {
    Shape shape = ShapeUtil::MakeShapeWithLayout(
        primitive_util::NativeToPrimitiveType<uint32>(), data.dimensions,
        data.layout);
    Literal literal(shape);
    auto generator = [&](absl::Span<const int64> indexes) -> uint32 {
      // Offsets from linear index just to avoid R0 literals to be initialized
      // with zero.
      return IndexUtil::MultidimensionalIndexToLinearIndex(literal.shape(),
                                                           indexes) +
             17;
    };
    TF_EXPECT_OK(literal.Populate<uint32>(generator));

    std::vector<int64> zero_base(data.dimensions.size(), 0);
    std::vector<int64> step(data.dimensions.size(), 1);
    bool matched = true;
    auto check_function = [&](absl::Span<const int64> indexes) {
      auto value = literal.Get<uint32>(indexes);
      matched = matched && (value == generator(indexes));
      return matched;
    };
    ShapeUtil::ForEachIndex(literal.shape(), zero_base, data.dimensions, step,
                            check_function);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, PopulateParallel) {
  struct PopulateData {
    std::vector<int64> dimensions;
    std::vector<int64> layout;
  } populate_data[] = {
      {{}, {}},
      {{0}, {0}},
      {{16}, {0}},
      {{2, 0}, {1, 0}},
      {{4, 16}, {1, 0}},
      {{21, 12}, {0, 1}},
      {{6, 11, 17}, {2, 0, 1}},
      {{6, 11, 5, 17}, {3, 2, 0, 1}},
  };
  for (const auto& data : populate_data) {
    Shape shape = ShapeUtil::MakeShapeWithLayout(
        primitive_util::NativeToPrimitiveType<uint32>(), data.dimensions,
        data.layout);
    Literal literal(shape);
    auto generator = [&](absl::Span<const int64> indexes) -> uint32 {
      // Offsets from linear index just to avoid R0 literals to be initialized
      // with zero.
      return IndexUtil::MultidimensionalIndexToLinearIndex(literal.shape(),
                                                           indexes) +
             17;
    };
    TF_EXPECT_OK(literal.PopulateParallel<uint32>(generator));

    std::vector<int64> zero_base(data.dimensions.size(), 0);
    std::vector<int64> step(data.dimensions.size(), 1);
    bool matched = true;
    auto check_function = [&](absl::Span<const int64> indexes) {
      auto value = literal.Get<uint32>(indexes);
      matched = matched && (value == generator(indexes));
      return matched;
    };
    ShapeUtil::ForEachIndex(literal.shape(), zero_base, data.dimensions, step,
                            check_function);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, ConvertR4) {
  // clang-format off
  auto original = LiteralUtil::CreateR4WithLayout<int8>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  auto expected = LiteralUtil::CreateR4WithLayout<uint32>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  // clang-format on
  TF_ASSERT_OK_AND_ASSIGN(Literal converted, original.Convert(U32));

  EXPECT_EQ(expected, converted);
}

TEST_F(LiteralUtilTest, ConvertIfTypesMatch) {
  // clang-format off
  auto s8 = LiteralUtil::CreateR4WithLayout<int8>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto s32 = LiteralUtil::CreateR4WithLayout<int32>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto u32 = LiteralUtil::CreateR4WithLayout<uint32>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto s64 = LiteralUtil::CreateR4WithLayout<int64>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto u64 = LiteralUtil::CreateR4WithLayout<uint64>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto pred = LiteralUtil::CreateR4WithLayout<bool>({{
    {{true, false, true, false}, {false, true, false, true}},
    {{false, true, false, true}, {true, false, true, false}},
    {{true, false, true, false}, {false, true, false, true}},
  }}, layout_r4_dim0major_);
  auto int32_pred = LiteralUtil::CreateR4WithLayout<int32>({{
    {{1, 0, 1, 0}, {0, 1, 0, 1}},
    {{0, 1, 0, 1}, {1, 0, 1, 0}},
    {{1, 0, 1, 0}, {0, 1, 0, 1}},
  }}, layout_r4_dim0major_);
  auto f16 = LiteralUtil::CreateR4WithLayout<half>({{
    {{half(10.0), half(0.0), half(12.0), half(0.0)},
     {half(0.0), half(15.0), half(0.0), half(17.0)}},
    {{half(0.0), half(19.0), half(0.0), half(21.0)},
     {half(22.0), half(0.0), half(24.0), half(0.0)}},
    {{half(26.0), half(0.0), half(28.0), half(0.0)},
     {half(0.0), half(31.0), half(0.0), half(33.0)}},
  }}, layout_r4_dim0major_);
  auto bf16 = LiteralUtil::CreateR4WithLayout<bfloat16>({{
    {{bfloat16(10.0), bfloat16(0.0), bfloat16(12.0), bfloat16(0.0)},
     {bfloat16(0.0), bfloat16(15.0), bfloat16(0.0), bfloat16(17.0)}},
    {{bfloat16(0.0), bfloat16(19.0), bfloat16(0.0), bfloat16(21.0)},
     {bfloat16(22.0), bfloat16(0.0), bfloat16(24.0), bfloat16(0.0)}},
    {{bfloat16(26.0), bfloat16(0.0), bfloat16(28.0), bfloat16(0.0)},
     {bfloat16(0.0), bfloat16(31.0), bfloat16(0.0), bfloat16(33.0)}},
  }}, layout_r4_dim0major_);
  auto f32 = LiteralUtil::CreateR4WithLayout<float>({{
    {{10.0f, 0.0f, 12.0f, 0.0f}, {0.0f, 15.0f, 0.0f, 17.0f}},
    {{0.0f, 19.0f, 0.0f, 21.0f}, {22.0f, 0.0f, 24.0f, 0.0f}},
    {{26.0f, 0.0f, 28.0f, 0.0f}, {0.0f, 31.0f, 0.0f, 33.0f}},
  }}, layout_r4_dim0major_);
  auto f64 = LiteralUtil::CreateR4WithLayout<double>({{
    {{10.0, 0.0, 12.0, 0.0}, {0.0, 15.0, 0.0, 17.0}},
    {{0.0, 19.0, 0.0, 21.0}, {22.0, 0.0, 24.0, 0.0}},
    {{26.0, 0.0, 28.0, 0.0}, {0.0, 31.0, 0.0, 33.0}},
  }}, layout_r4_dim0major_);
  auto c64 = LiteralUtil::CreateR4WithLayout<complex64>({{
    {{10.0f, 0.0f, 12.0f, 0.0f}, {0.0f, 15.0f, 0.0f, 17.0f}},
    {{0.0f, 19.0f, 0.0f, 21.0f}, {22.0f, 0.0f, 24.0f, 0.0f}},
    {{26.0f, 0.0f, 28.0f, 0.0f}, {0.0f, 31.0f, 0.0f, 33.0f}},
  }}, layout_r4_dim0major_);
  // clang-format on
  Literal conv;

  conv = s8.Convert(U32).ConsumeValueOrDie();
  EXPECT_EQ(conv, u32);

  conv = s8.Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(conv, s32);

  conv = s8.Convert(U64).ConsumeValueOrDie();
  EXPECT_EQ(conv, u64);

  conv = s8.Convert(S64).ConsumeValueOrDie();
  EXPECT_EQ(conv, s64);

  conv = s8.Convert(PRED).ConsumeValueOrDie();
  EXPECT_EQ(conv, pred);

  conv = bf16.Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(conv, s32);

  conv = bf16.Convert(F32).ConsumeValueOrDie();
  EXPECT_EQ(conv, f32);

  conv = pred.Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(conv, int32_pred);

  conv = f32.Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(conv, s32);

  conv = f64.Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(conv, s32);

  conv = s32.Convert(F32).ConsumeValueOrDie();
  EXPECT_EQ(conv, f32);

  conv = f32.Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(conv, f16);

  conv = f64.Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(conv, f16);

  conv = s32.Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(conv, f16);

  conv = u32.Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(conv, f16);

  conv = s32.Convert(C64).ConsumeValueOrDie();
  EXPECT_EQ(conv, c64);

  conv = f16.Convert(C64).ConsumeValueOrDie();
  EXPECT_EQ(conv, c64);

  EXPECT_EQ(s32.Convert(TUPLE).status().code(),
            tensorflow::error::UNIMPLEMENTED);
  EXPECT_EQ(s32.Convert(S16).status().code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_EQ(s32.Convert(U16).status().code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_EQ(c64.Convert(F32).status().code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_EQ(c64.Convert(S32).status().code(), tensorflow::error::UNIMPLEMENTED);
}

TEST_F(LiteralUtilTest, BitcastConvert) {
  auto original = LiteralUtil::CreateR1<uint32>(
      {absl::bit_cast<uint32>(2.5f), absl::bit_cast<uint32>(-42.25f),
       absl::bit_cast<uint32>(100.f), 0xbeef});
  auto expected = LiteralUtil::CreateR1<float>(
      {2.5f, -42.25f, 100.0f, absl::bit_cast<float>(0xbeef)});
  TF_ASSERT_OK_AND_ASSIGN(Literal converted, original.BitcastConvert(F32));
}

TEST_F(LiteralUtilTest, BitcastConvertBetweenInvalidTypes) {
  auto literal = LiteralUtil::CreateR0<uint32>(1234);
  Status status = literal.BitcastConvert(F64).status();
  EXPECT_NE(Status::OK(), status);
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "bit widths are different"));
}

// Sets the layout of the given ShapeProto to the default.
void SetDefaultLayoutOnProto(ShapeProto* shape_proto) {
  CHECK(ShapeUtil::IsArrayPrimitiveType(shape_proto->element_type()));
  shape_proto->mutable_layout()->set_format(DENSE);
  auto* minor_to_major =
      shape_proto->mutable_layout()->mutable_minor_to_major();
  minor_to_major->Resize(shape_proto->dimensions_size(), 0);
  const int64 size = minor_to_major->size();
  for (int64 i = 0; i < size; ++i) {
    minor_to_major->Set(i, size - 1 - i);
  }
}

TEST_F(LiteralUtilTest, CopyFromProto_Bool) {
  LiteralProto p;
  p.mutable_shape()->set_element_type(PRED);
  for (int len = 0; len < 25; ++len) {
    p.mutable_shape()->clear_dimensions();
    p.mutable_shape()->add_dimensions(len);
    SetDefaultLayoutOnProto(p.mutable_shape());
    p.clear_preds();
    for (int i = 0; i < len; ++i) {
      p.add_preds((i % 2) == (len % 2));
    }

    TF_ASSERT_OK_AND_ASSIGN(Literal literal, Literal::CreateFromProto(p));
    ASSERT_EQ(len, literal.data<bool>().size());
    int i = 0;
    for (bool value : literal.data<bool>()) {
      EXPECT_EQ((i % 2) == (len % 2), value);
      ++i;
    }
  }
}

// Note that f16 is currently stored in a byte array in little endian byte order
TEST_F(LiteralUtilTest, ToProto_f16) {
  half h1(1.0f);
  half h2(2.0f);

  auto m = LiteralUtil::CreateR2<half>({{h1, h2}, {h2, h1}});
  EXPECT_EQ(4, ShapeUtil::ElementsIn(m.shape()));
  EXPECT_EQ(4, m.data<half>().size());

  LiteralProto p = m.ToProto();
  EXPECT_EQ(4, ShapeUtil::ElementsIn(Shape(p.shape())));
  EXPECT_EQ(8, p.f16s().size());
  const char* d = p.f16s().data();
  EXPECT_EQ(d[0], 0);
  EXPECT_EQ(d[1], 0x3C);
  EXPECT_EQ(d[2], 0);
  EXPECT_EQ(d[3], 0x40);
  EXPECT_EQ(d[4], 0);
  EXPECT_EQ(d[5], 0x40);
  EXPECT_EQ(d[6], 0);
  EXPECT_EQ(d[7], 0x3C);
}

// Note that f16 is currently stored in a byte array in little endian byte order
TEST_F(LiteralUtilTest, CopyFromProto_f16) {
  half h1(1.0f);
  half h2(2.0f);

  const char half_vals[8] = {0x00, 0x3C, 0x00, 0x40, 0x00, 0x40, 0x00, 0x3C};
  LiteralProto p;
  p.mutable_shape()->set_element_type(F16);
  p.mutable_shape()->clear_dimensions();
  p.mutable_shape()->add_dimensions(4);
  SetDefaultLayoutOnProto(p.mutable_shape());
  p.clear_f16s();
  p.set_f16s(half_vals, 8);
  TF_ASSERT_OK_AND_ASSIGN(Literal literal, Literal::CreateFromProto(p));
  auto r = literal.data<half>();
  ASSERT_EQ(4, r.size());
  EXPECT_EQ(h1, r[0]);
  EXPECT_EQ(h2, r[1]);
  EXPECT_EQ(h2, r[2]);
  EXPECT_EQ(h1, r[3]);
}

TEST_F(LiteralUtilTest, CopyFromProto_u16) {
  uint16 u1(0xabcd);
  uint16 u2(0x1234);

  const unsigned char uint16_vals[8] = {0xcd, 0xab, 0x34, 0x12,
                                        0x34, 0x12, 0xcd, 0xab};
  LiteralProto p;
  p.mutable_shape()->set_element_type(U16);
  p.mutable_shape()->clear_dimensions();
  p.mutable_shape()->add_dimensions(4);
  SetDefaultLayoutOnProto(p.mutable_shape());
  p.clear_u16s();
  p.set_u16s(uint16_vals, 8);
  TF_ASSERT_OK_AND_ASSIGN(Literal literal, Literal::CreateFromProto(p));
  auto r = literal.data<uint16>();
  ASSERT_EQ(4, r.size());
  EXPECT_EQ(u1, r[0]);
  EXPECT_EQ(u2, r[1]);
  EXPECT_EQ(u2, r[2]);
  EXPECT_EQ(u1, r[3]);
}

TEST_F(LiteralUtilTest, LiteralSliceTest) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = LiteralUtil::MakeTuple({&scalar, &matrix});
  auto nested_tuple = LiteralUtil::MakeTuple({&tuple, &scalar});
  Literal nil(ShapeUtil::MakeNil());

  EXPECT_EQ(LiteralSlice(scalar, {}), scalar);
  EXPECT_EQ(LiteralSlice(matrix, {}), matrix);
  EXPECT_EQ(LiteralSlice(tuple, {}), tuple);
  EXPECT_EQ(LiteralSlice(nested_tuple, {}), nested_tuple);
  EXPECT_EQ(LiteralSlice(nil, {}), nil);

  EXPECT_EQ(LiteralSlice(tuple, {0}), scalar);
  EXPECT_EQ(LiteralSlice(tuple, {1}), matrix);

  EXPECT_EQ(LiteralSlice(nested_tuple, {0}), tuple);
  EXPECT_EQ(LiteralSlice(nested_tuple, {0, 0}), scalar);
  EXPECT_EQ(LiteralSlice(nested_tuple, {0, 1}), matrix);
  EXPECT_EQ(LiteralSlice(nested_tuple, {1}), scalar);
}

TEST_F(LiteralUtilTest, MutatingLiteralSlice) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = LiteralUtil::MakeTuple({&scalar, &matrix});
  auto nested_tuple = LiteralUtil::MakeTuple({&tuple, &scalar});
  // Verify that changing the underlying data beneath the view changes the
  // data of the view itself.
  const auto nested_tuple_view = LiteralSlice(nested_tuple);
  EXPECT_EQ(nested_tuple.Get<float>(/*multi_index=*/{}, /*shape_index=*/{0, 0}),
            1.0f);
  EXPECT_EQ(nested_tuple_view.Get<float>(/*multi_index=*/{},
                                         /*shape_index=*/{0, 0}),
            1.0f);
  nested_tuple.Set<float>(/*multi_index=*/{}, /*shape_index=*/{0, 0}, 555.0f);
  EXPECT_EQ(nested_tuple.Get<float>(/*multi_index=*/{}, /*shape_index=*/{0, 0}),
            555.0f);
  EXPECT_EQ(nested_tuple_view.Get<float>(/*multi_index=*/{},
                                         /*shape_index=*/{0, 0}),
            555.0f);
}

TEST_F(LiteralUtilTest, LiteralSliceOfALiteralSlice) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = LiteralUtil::MakeTuple({&scalar, &matrix});
  auto nested_tuple = LiteralUtil::MakeTuple({&tuple, &scalar});

  const auto nested_tuple_view = LiteralSlice(nested_tuple);
  const auto tuple_view = LiteralSlice(nested_tuple_view, /*view_root=*/{0});
  const auto matrix_view = LiteralSlice(tuple_view, /*view_root=*/{1});
  EXPECT_EQ(matrix_view,
            LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}));
}

TEST_F(LiteralUtilTest, BorrowingLiteralFromOneBufferPtr) {
  std::vector<int64> int64_values = {1, 2, 3};
  const Shape literal_shape = ShapeUtil::MakeShape(S64, {3});

  BorrowingLiteral literal(reinterpret_cast<const char*>(int64_values.data()),
                           literal_shape);

  EXPECT_EQ(literal.Get<int64>({0}), 1);
  EXPECT_EQ(literal.Get<int64>({1}), 2);
  EXPECT_EQ(literal.Get<int64>({2}), 3);
}

TEST_F(LiteralUtilTest, BorrowingLiteralFromMultipleBufferPtrs) {
  std::vector<int64> one_two_three = {1, 2, 3};
  const Shape one_two_three_shape = ShapeUtil::MakeShape(S64, {3});

  std::vector<int64> hundred = {100};
  const Shape hundred_shape = ShapeUtil::MakeShape(S64, {1});

  std::vector<const char*> src_buf_ptrs;
  src_buf_ptrs.emplace_back(
      reinterpret_cast<const char*>(one_two_three.data()));
  src_buf_ptrs.emplace_back(reinterpret_cast<const char*>(hundred.data()));
  auto literal_tuple = BorrowingLiteral(
      src_buf_ptrs,
      ShapeUtil::MakeTupleShape({one_two_three_shape, hundred_shape}));

  EXPECT_EQ(literal_tuple.Get<int64>(/*multi_index=*/{0}, /*shape_index=*/{0}),
            1);
  EXPECT_EQ(literal_tuple.Get<int64>(/*multi_index=*/{0}, /*shape_index=*/{1}),
            100);

  EXPECT_EQ(literal_tuple.Get<int64>(/*multi_index=*/{1}, /*shape_index=*/{0}),
            2);

  EXPECT_EQ(literal_tuple.Get<int64>(/*multi_index=*/{2}, /*shape_index=*/{0}),
            3);
}

TEST_F(LiteralUtilTest, LiteralMove) {
  Literal matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal literal(std::move(matrix));

  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {2, 2}), literal.shape()));
  EXPECT_EQ(literal.Get<float>({0, 0}), 1.0);
  EXPECT_EQ(literal.Get<float>({0, 1}), 2.0);
  EXPECT_EQ(literal.Get<float>({1, 0}), 3.0);
  EXPECT_EQ(literal.Get<float>({1, 1}), 4.0);
}

TEST_F(LiteralUtilTest, DecomposeTuple) {
  Literal nil_literal(ShapeUtil::MakeNil());
  Literal inner_elements[] = {
      LiteralUtil::CreateR0<int32>(42),
      LiteralUtil::CreateR1<double>({23.0, 44.0}),
  };
  Literal tuple_elements[] = {
      LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}}),
      LiteralUtil::MakeTuple(
          {&inner_elements[0], &inner_elements[1], &nil_literal}),
  };
  Literal nested_tuple = LiteralUtil::MakeTuple(
      {&tuple_elements[0], &tuple_elements[1], &nil_literal});

  EXPECT_FALSE(ShapeUtil::IsEmptyTuple(nested_tuple.shape()));
  std::vector<Literal> elements = nested_tuple.DecomposeTuple();
  EXPECT_TRUE(ShapeUtil::IsEmptyTuple(nested_tuple.shape()));

  ASSERT_EQ(elements.size(), 3);

  EXPECT_TRUE(ShapeUtil::Compatible(elements[0].shape(),
                                    ShapeUtil::MakeShape(S32, {2, 2})));
  EXPECT_EQ(elements[0].Get<int32>({0, 0}), 1);
  EXPECT_EQ(elements[0].Get<int32>({0, 1}), 2);
  EXPECT_EQ(elements[0].Get<int32>({1, 0}), 3);
  EXPECT_EQ(elements[0].Get<int32>({1, 1}), 4);

  EXPECT_TRUE(ShapeUtil::Compatible(
      elements[1].shape(),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {}),
                                 ShapeUtil::MakeShape(F64, {2}),
                                 ShapeUtil::MakeNil()})));
  EXPECT_EQ(elements[1].Get<int32>({}, /*shape_index=*/{0}), 42);
  EXPECT_EQ(elements[1].Get<double>({0}, /*shape_index=*/{1}), 23.0);
  EXPECT_EQ(elements[1].Get<double>({1}, /*shape_index=*/{1}), 44.0);

  EXPECT_TRUE(ShapeUtil::Compatible(elements[2].shape(), ShapeUtil::MakeNil()));
}

TEST_F(LiteralUtilTest, DecomposeEmptyTuple) {
  Literal nil_literal(ShapeUtil::MakeNil());
  std::vector<Literal> elements = nil_literal.DecomposeTuple();
  EXPECT_EQ(elements.size(), 0);
}

TEST_F(LiteralUtilTest, MoveIntoTuple) {
  std::vector<Literal> elements;
  elements.push_back(LiteralUtil::CreateR0<float>(1.0));
  elements.push_back(LiteralUtil::CreateR1<int32>({4, 8}));
  std::vector<Literal> inner_elements;
  inner_elements.push_back(LiteralUtil::CreateR0<int32>(42));
  inner_elements.push_back(LiteralUtil::CreateR1<double>({23.0, 44.0}));
  elements.push_back(
      LiteralUtil::MakeTuple({&inner_elements[0], &inner_elements[1]}));

  Literal literal = Literal::MoveIntoTuple(absl::MakeSpan(elements));
  ASSERT_TRUE(ShapeUtil::IsTuple(literal.shape()));
  ASSERT_EQ(ShapeUtil::TupleElementCount(literal.shape()), 3);

  EXPECT_EQ(literal.Get<float>({}, /*shape_index=*/{0}), 1.0);
  EXPECT_EQ(literal.Get<int32>({0}, /*shape_index=*/{1}), 4);
  EXPECT_EQ(literal.Get<int32>({1}, /*shape_index=*/{1}), 8);
  EXPECT_EQ(literal.Get<int32>({}, /*shape_index=*/{2, 0}), 42);
  EXPECT_EQ(literal.Get<double>({0}, /*shape_index=*/{2, 1}), 23.0);
  EXPECT_EQ(literal.Get<double>({1}, /*shape_index=*/{2, 1}), 44.0);

  for (const Literal& element : elements) {
    EXPECT_TRUE(ShapeUtil::IsEmptyTuple(element.shape()));
  }
}

TEST_F(LiteralUtilTest, MoveIntoEmptyTuple) {
  Literal literal = Literal::MoveIntoTuple({});
  ASSERT_TRUE(ShapeUtil::IsTuple(literal.shape()));
  EXPECT_EQ(ShapeUtil::TupleElementCount(literal.shape()), 0);
}

TEST_F(LiteralUtilTest, LiteralMoveAssignment) {
  Literal literal;
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeNil(), literal.shape()));

  Literal matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  literal = std::move(matrix);

  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {2, 2}), literal.shape()));
  EXPECT_EQ(literal.Get<float>({0, 0}), 1.0);
  EXPECT_EQ(literal.Get<float>({0, 1}), 2.0);
  EXPECT_EQ(literal.Get<float>({1, 0}), 3.0);
  EXPECT_EQ(literal.Get<float>({1, 1}), 4.0);
}

TEST_F(LiteralUtilTest, LiteralSliceCopy) {
  Literal matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  const auto matrix_view = LiteralSlice(matrix);
  LiteralSlice matrix_view_copy(matrix_view);

  EXPECT_EQ(matrix_view_copy.Get<float>({0, 0}), 1.0);
  EXPECT_EQ(matrix_view_copy.Get<float>({0, 1}), 2.0);
  EXPECT_EQ(matrix_view_copy.Get<float>({1, 0}), 3.0);
  EXPECT_EQ(matrix_view_copy.Get<float>({1, 1}), 4.0);
}

TEST_F(LiteralUtilTest, GetSetTuple) {
  Literal elements[] = {
      LiteralUtil::CreateR0<float>(42.0),
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}),
  };
  auto tuple = LiteralUtil::MakeTuple({&elements[0], &elements[1]});
  EXPECT_EQ(tuple.Get<float>(/*multi_index=*/{}, /*shape_index=*/{0}), 42.0);
  tuple.Set<float>(/*multi_index=*/{}, /*shape_index=*/{0}, -5.0);
  EXPECT_EQ(tuple.Get<float>(/*multi_index=*/{}, /*shape_index=*/{0}), -5.0);

  EXPECT_EQ(tuple.Get<float>(/*multi_index=*/{1, 0}, /*shape_index=*/{1}), 3.0);
  tuple.Set<float>(/*multi_index=*/{1, 0}, /*shape_index=*/{1}, -4.0);
  EXPECT_EQ(tuple.Get<float>(/*multi_index=*/{1, 0}, /*shape_index=*/{1}),
            -4.0);
}

TEST_F(LiteralUtilTest, CreateFromShapeZeroInitialized) {
  // Literals constructed using CreateFromShape should be zero initialized.
  Literal scalar_f32 = Literal::CreateFromShape(ShapeUtil::MakeShape(F32, {}));
  EXPECT_EQ(scalar_f32.Get<float>({}), 0.0);
  EXPECT_TRUE(scalar_f32.IsAll(0));

  Literal vector_s32 = Literal::CreateFromShape(ShapeUtil::MakeShape(S32, {3}));
  EXPECT_EQ(vector_s32.Get<int32>({0}), 0);
  EXPECT_EQ(vector_s32.Get<int32>({1}), 0);
  EXPECT_EQ(vector_s32.Get<int32>({2}), 0);
  EXPECT_TRUE(vector_s32.IsAll(0));

  Literal tuple = Literal::CreateFromShape(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F64, {}), ShapeUtil::MakeShape(PRED, {2}),
       ShapeUtil::MakeShape(U64, {2, 1}), ShapeUtil::MakeShape(C64, {})}));

  EXPECT_EQ(tuple.Get<double>({}, {0}), 0.0);
  EXPECT_EQ(tuple.Get<bool>({0}, {1}), false);
  EXPECT_EQ(tuple.Get<bool>({1}, {1}), false);
  EXPECT_EQ(tuple.Get<uint64>({0, 0}, {2}), 0);
  EXPECT_EQ(tuple.Get<uint64>({1, 0}, {2}), 0);
  EXPECT_EQ(tuple.Get<complex64>({}, {3}), complex64(0.0f, 0.0f));
}

TEST_F(LiteralUtilTest, ProtoRoundTrip) {
  // Test serializing then deserializing a Literal through a proto.
  auto one_f32 = LiteralUtil::CreateR0<float>(1.0);
  auto two_f32 = LiteralUtil::CreateR0<float>(2.0);
  auto vector_int8 = LiteralUtil::CreateR1<int8>({-128, 0, 2, 4, 7, 56, 127});
  auto vector_uint8 = LiteralUtil::CreateR1<uint8>({128, 0, 2, 56, 127, 255});
  auto vector_c64 = LiteralUtil::CreateR1<complex64>({{1.0, 2.0}, {3.0, 4.0}});
  auto vector_bfloat16 = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16{-1.0}, bfloat16{2.0}, bfloat16{-3.0}});
  auto vector_half =
      LiteralUtil::CreateR1<half>({half{10.0}, half{20.0}, half{-30.0}});
  auto matrix_pred =
      LiteralUtil::CreateR2<bool>({{true, false, true}, {false, false, true}});
  auto tuple = LiteralUtil::MakeTuple(
      {&one_f32, &vector_half, &matrix_pred, &matrix_pred});
  Literal nil_literal(ShapeUtil::MakeNil());
  auto nested_tuple =
      LiteralUtil::MakeTuple({&tuple, &vector_bfloat16, &tuple, &nil_literal});

  auto to_from_proto = [](const Literal& literal) -> Literal {
    return Literal::CreateFromProto(literal.ToProto()).ValueOrDie();
  };

  EXPECT_EQ(one_f32, to_from_proto(one_f32));
  EXPECT_EQ(vector_int8, to_from_proto(vector_int8));
  EXPECT_EQ(vector_uint8, to_from_proto(vector_uint8));
  EXPECT_EQ(vector_c64, to_from_proto(vector_c64));
  EXPECT_EQ(vector_bfloat16, to_from_proto(vector_bfloat16));
  EXPECT_EQ(matrix_pred, to_from_proto(matrix_pred));
  EXPECT_EQ(tuple, to_from_proto(tuple));
  EXPECT_EQ(nested_tuple, to_from_proto(nested_tuple));
  EXPECT_EQ(nil_literal, to_from_proto(nil_literal));

  EXPECT_NE(one_f32, two_f32);
  EXPECT_NE(one_f32, to_from_proto(two_f32));
}

TEST_F(LiteralUtilTest, InvalidProtoNoValues) {
  // Proto contains a shape, but no values.
  LiteralProto proto;
  *proto.mutable_shape() = ShapeUtil::MakeShape(F32, {3}).ToProto();
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected 3 elements in LiteralProto"));
}

TEST_F(LiteralUtilTest, InvalidProtoNoShape) {
  // Proto contains values, but no shape.
  LiteralProto proto;
  proto.add_preds(false);
  proto.add_preds(true);
  proto.add_preds(false);
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("LiteralProto has no shape"));
}

TEST_F(LiteralUtilTest, InvalidProtoWrongContainer) {
  // Proto contains values in wrong container.
  LiteralProto proto;
  *proto.mutable_shape() = ShapeUtil::MakeShape(F32, {3}).ToProto();
  proto.add_preds(false);
  proto.add_preds(true);
  proto.add_preds(false);
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected 3 elements in LiteralProto"));
}

TEST_F(LiteralUtilTest, InvalidProtoTooFewValues) {
  // Proto contains too few values.
  LiteralProto proto;
  *proto.mutable_shape() = ShapeUtil::MakeShape(F32, {42, 2}).ToProto();
  proto.add_f32s(1.0);
  proto.add_f32s(2.0);
  proto.add_f32s(3.0);
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected 84 elements in LiteralProto"));
}

TEST_F(LiteralUtilTest, InvalidProtoTooManyValues) {
  // Proto contains too many values.
  LiteralProto proto;
  *proto.mutable_shape() = ShapeUtil::MakeShape(S32, {2}).ToProto();
  proto.add_s32s(42);
  proto.add_s32s(-10);
  proto.add_s32s(100);
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected 2 elements in LiteralProto"));
}

TEST_F(LiteralUtilTest, InvalidProtoMissingLayout) {
  // Proto shape missing layout.
  LiteralProto proto;
  *proto.mutable_shape() = ShapeUtil::MakeShape(PRED, {2, 2}).ToProto();
  proto.mutable_shape()->clear_layout();
  proto.add_preds(true);
  proto.add_preds(false);
  proto.add_preds(true);
  proto.add_preds(false);
  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("LiteralProto has no layout"));
}

TEST_F(LiteralUtilTest, InvalidProtoTooFewTupleElements) {
  // Proto has the too few tuple elements.
  LiteralProto proto;
  *proto.mutable_shape() =
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(PRED, {2}), ShapeUtil::MakeShape(F32, {})})
          .ToProto();
  LiteralProto* element0 = proto.add_tuple_literals();
  *element0->mutable_shape() =
      ShapeUtil::GetTupleElementShape(Shape(proto.shape()), 0).ToProto();
  element0->add_preds(false);
  element0->add_preds(true);

  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Expected 2 tuple elements"));
}

TEST_F(LiteralUtilTest, InvalidProtoTooManyTupleElements) {
  // Proto has the too many tuple elements.
  LiteralProto proto;
  *proto.mutable_shape() =
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(PRED, {2}), ShapeUtil::MakeShape(F32, {})})
          .ToProto();
  LiteralProto* element0 = proto.add_tuple_literals();
  *element0->mutable_shape() =
      ShapeUtil::GetTupleElementShape(Shape(proto.shape()), 0).ToProto();
  element0->add_preds(false);
  element0->add_preds(true);
  LiteralProto* element1 = proto.add_tuple_literals();
  *element1->mutable_shape() =
      ShapeUtil::GetTupleElementShape(Shape(proto.shape()), 1).ToProto();
  element1->add_f32s(42.0);
  LiteralProto* element2 = proto.add_tuple_literals();
  *element2->mutable_shape() = ShapeUtil::MakeShape(F32, {}).ToProto();
  element2->add_f32s(123.0);

  Status status = Literal::CreateFromProto(proto).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Expected 2 tuple elements"));
}

TEST_F(LiteralUtilTest, SortSparseElements) {
  auto literal = LiteralUtil::CreateSparse<float>({10, 10, 10},
                                                  SparseIndexArray(10, 3), {});
  literal.AppendSparseElement<float>({2, 3, 4}, 2.0);
  literal.AppendSparseElement<float>({3, 4, 5}, 3.0);
  literal.AppendSparseElement<float>({1, 2, 3}, 1.0);
  literal.SortSparseElements();
  EXPECT_EQ(literal.ToString(),
            "f32[10,10,10]{[1, 2, 3]: 1, [2, 3, 4]: 2, [3, 4, 5]: 3}");
}

TEST_F(LiteralUtilTest, GetSparseElementAsString) {
  std::vector<int64> dimensions = {10, 10, 10};
  SparseIndexArray indices(10, {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  EXPECT_EQ(
      LiteralUtil::CreateSparse<bool>(dimensions, indices, {true, false, true})
          .GetSparseElementAsString(1),
      "false");
  EXPECT_EQ(LiteralUtil::CreateSparse<int64>(dimensions, indices, {1, 2, 3})
                .GetSparseElementAsString(1),
            absl::StrCat(int64{2}));
  EXPECT_EQ(
      LiteralUtil::CreateSparse<double>(dimensions, indices, {1.0, 2.0, 3.0})
          .GetSparseElementAsString(1),
      absl::StrCat(double{2.0}));
  EXPECT_EQ(LiteralUtil::CreateSparse<half>(dimensions, indices,
                                            {half{1.0}, half{2.0}, half{3.0}})
                .GetSparseElementAsString(1),
            absl::StrCat(static_cast<float>(half{2.0})));
  EXPECT_EQ(LiteralUtil::CreateSparse<complex64>(
                dimensions, indices,
                std::vector<complex64>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}})
                .GetSparseElementAsString(1),
            absl::StrCat("(", float{3.0}, ", ", float{4.0}, ")"));
}

TEST_F(LiteralUtilTest, BroadcastVectorToMatrix0) {
  Literal literal = LiteralUtil::CreateR1<int64>({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal broadcasted_literal,
      literal.Broadcast(/*result_shape=*/ShapeUtil::MakeShape(S64, {2, 2}),
                        /*dimensions=*/{0}));
  EXPECT_EQ(broadcasted_literal,
            LiteralUtil::CreateR2<int64>({{1, 1}, {2, 2}}));
}

TEST_F(LiteralUtilTest, BroadcastVectorToMatrix1) {
  Literal literal = LiteralUtil::CreateR1<int64>({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal broadcasted_literal,
      literal.Broadcast(/*result_shape=*/ShapeUtil::MakeShape(S64, {2, 2}),
                        /*dimensions=*/{1}));
  EXPECT_EQ(broadcasted_literal,
            LiteralUtil::CreateR2<int64>({{1, 2}, {1, 2}}));
}

TEST_F(LiteralUtilTest, BroadcastScalarToMatrix) {
  Literal literal = LiteralUtil::CreateR0<int32>(9);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal broadcasted_literal,
      literal.Broadcast(/*result_shape=*/ShapeUtil::MakeShape(S32, {2, 2}),
                        /*dimensions=*/{}));
  EXPECT_EQ(broadcasted_literal,
            LiteralUtil::CreateR2<int32>({{9, 9}, {9, 9}}));
}

}  // namespace
}  // namespace xla
