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

#include "tensorflow/compiler/xla/literal_util.h"

#include <vector>

#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

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
        Literal::CreateR4FromArray4DWithLayout<float>(arr4d,
                                                      layout_r4_dim0major_);
    literal_r4_2x2x3x3_dim0minor_ =
        Literal::CreateR4FromArray4DWithLayout<float>(arr4d,
                                                      layout_r4_dim0minor_);
  }

  Layout layout_r2_dim0major_;
  Layout layout_r2_dim0minor_;
  Layout layout_r3_dim0major_;
  Layout layout_r3_dim0minor_;
  Layout layout_r4_dim0major_;
  Layout layout_r4_dim0minor_;
  std::unique_ptr<Literal> literal_r4_2x2x3x3_dim0major_;
  std::unique_ptr<Literal> literal_r4_2x2x3x3_dim0minor_;
};

TEST_F(LiteralUtilTest, LiteralScalarToString) {
  auto true_lit = Literal::CreateR0<bool>(true);
  ASSERT_EQ("true", true_lit->ToString());

  auto false_lit = Literal::CreateR0<bool>(false);
  ASSERT_EQ("false", false_lit->ToString());

  auto u32_lit = Literal::CreateR0<uint32>(42);
  ASSERT_EQ("42", u32_lit->ToString());

  auto s32_lit = Literal::CreateR0<int32>(-999);
  ASSERT_EQ("-999", s32_lit->ToString());

  auto f32_lit = Literal::CreateR0<float>(3.14f);
  ASSERT_EQ("3.14", f32_lit->ToString());

  auto f16_lit = Literal::CreateR0<half>(static_cast<half>(0.5f));
  ASSERT_EQ("0.5", f16_lit->ToString());

  auto c64_lit = Literal::CreateR0<complex64>({3.14f, 2.78f});
  ASSERT_EQ("(3.14, 2.78)", c64_lit->ToString());

  auto bf16_lit = Literal::CreateR0<bfloat16>(static_cast<bfloat16>(0.5f));
  ASSERT_EQ("0.5", bf16_lit->ToString());

  // 3.14 will be rounded to 3.125 in bfloat16 format (Round to nearest even).
  auto bf16_lit_truncated =
      Literal::CreateR0<bfloat16>(static_cast<bfloat16>(3.14f));
  ASSERT_EQ("3.140625", bf16_lit_truncated->ToString());

  auto bf16_lit_truncated2 =
      Literal::CreateR0<bfloat16>(static_cast<bfloat16>(9.001f));
  ASSERT_EQ("9", bf16_lit_truncated2->ToString());
}

TEST_F(LiteralUtilTest, LiteralVectorToString) {
  auto pred_vec = Literal::CreateR1<bool>({true, false, true});
  ASSERT_EQ("{101}", pred_vec->ToString());
}

TEST_F(LiteralUtilTest, R2ToString) {
  const auto literal = Literal::CreateR2({{1, 2}, {3, 4}, {5, 6}});
  const string expected = R"(s32[3,2] {
  { 1, 2 },
  { 3, 4 },
  { 5, 6 }
})";
  ASSERT_EQ(expected, literal->ToString());
}

TEST_F(LiteralUtilTest, R3ToString) {
  const auto literal = Literal::CreateR3({{{1}, {2}}, {{3}, {4}}, {{5}, {6}}});
  const string expected = R"(s32[3,2,1] {
{ { 1 },
  { 2 } },
{ { 3 },
  { 4 } },
{ { 5 },
  { 6 } }
})";
  ASSERT_EQ(expected, literal->ToString());
}

TEST_F(LiteralUtilTest, TupleToString) {
  auto scalar = Literal::CreateR0<float>(1.0);
  auto matrix = Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = Literal::MakeTuple({scalar.get(), matrix.get()});
  const string expected = R"((f32[], f32[2,2]) (
1,
f32[2,2] {
  { 1, 2 },
  { 3, 4 }
}
))";
  ASSERT_EQ(expected, tuple->ToString());
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

  auto literal = Literal::CreateR3FromArray3D(array_3d);
  EXPECT_THAT(literal->shape().dimensions(), ElementsAre(2, 3, 2));
  string result = literal->ToString();
  const string expected = R"(f32[2,3,2] {
{ { 1, 2 },
  { 3, 4 },
  { 5, 6 } },
{ { 7, 8 },
  { 9, 10 },
  { 11, 12 } }
})";
  ASSERT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, LiteralR4F32ProjectedStringifies) {
  // clang-format off
  auto literal = Literal::CreateR4Projected<float>({
    {1, 2},
    {1001, 1002},
    {2001, 2002},
  }, /*projection_p=*/1, /*projection_z=*/2);
  // clang-format on
  EXPECT_THAT(literal->shape().dimensions(), ElementsAre(1, 2, 3, 2));
  string result = literal->ToString();
  const string expected = R"(f32[1,2,3,2] {
  {  /*i0=0*/
    {  /*i1=0*/
      {1, 2},
      {1001, 1002},
      {2001, 2002}
    },
    {  /*i1=1*/
      {1, 2},
      {1001, 1002},
      {2001, 2002}
    }
  }
})";
  ASSERT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, LiteralR4F32Stringifies) {
  EXPECT_THAT(literal_r4_2x2x3x3_dim0major_->shape().dimensions(),
              ElementsAre(2, 2, 3, 3));
  string result = literal_r4_2x2x3x3_dim0major_->ToString();
  const string expected = R"(f32[2,2,3,3] {
  {  /*i0=0*/
    {  /*i1=0*/
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9}
    },
    {  /*i1=1*/
      {11, 12, 13},
      {14, 15, 16},
      {17, 18, 19}
    }
  },
  {  /*i0=1*/
    {  /*i1=0*/
      {101, 102, 103},
      {104, 105, 106},
      {107, 108, 109}
    },
    {  /*i1=1*/
      {201, 202, 203},
      {204, 205, 206},
      {207, 208, 209}
    }
  }
})";
  ASSERT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, EachCellR2F32) {
  // clang-format off
  auto literal = Literal::CreateR2<float>({
    {3.1f, 4.2f},
    {9.3f, 12.4f},
  });
  // clang-format on
  std::vector<std::tuple<int64, int64, string>> seen;
  literal->EachCellAsString(
      [&seen](tensorflow::gtl::ArraySlice<int64> indices, const string& value) {
        seen.emplace_back(indices[0], indices[1], value);
      });

  using Elem = std::tuple<int64, int64, string>;
  std::vector<Elem> expected = {Elem(0, 0, "3.1"), Elem(0, 1, "4.2"),
                                Elem(1, 0, "9.3"), Elem(1, 1, "12.4")};
  EXPECT_EQ(expected, seen);
}

TEST_F(LiteralUtilTest, ScalarEquality) {
  // Test equality with scalars.
  auto f32_42 = Literal::CreateR0<float>(42.0);
  auto f32_42_clone = Literal::CreateR0<float>(42.0);

  EXPECT_EQ(*f32_42, *f32_42);
  EXPECT_EQ(*f32_42, *f32_42_clone);

  auto f32_123 = Literal::CreateR0<float>(123.0);
  EXPECT_NE(*f32_42, *f32_123);

  auto f64_42 = Literal::CreateR0<double>(42.0);
  EXPECT_NE(*f32_42, *f64_42);
}

TEST_F(LiteralUtilTest, NonScalarEquality) {
  // Test equality with nonscalars.
  auto matrix = Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_clone = Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_different = Literal::CreateR2<float>({{4.0, 3.0}, {1.0, 2.0}});
  auto vector_literal = Literal::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto scalar = Literal::CreateR0<float>(1.0);

  EXPECT_EQ(*matrix, *matrix);
  EXPECT_EQ(*matrix, *matrix_clone);
  EXPECT_NE(*matrix, *matrix_different);
  EXPECT_NE(*matrix, *vector_literal);
  EXPECT_NE(*matrix, *scalar);
}

TEST_F(LiteralUtilTest, DifferentLayoutEquality) {
  // Test equality with literals which have different layouts.
  auto colmajor = MakeUnique<Literal>();
  *colmajor->mutable_shape() = ShapeUtil::MakeShape(F32, {2, 2});
  *colmajor->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  colmajor->Reserve(4);
  colmajor->Set<float>({0, 0}, 1.0);
  colmajor->Set<float>({0, 1}, 2.0);
  colmajor->Set<float>({1, 0}, 3.0);
  colmajor->Set<float>({1, 1}, 4.0);

  auto rowmajor = MakeUnique<Literal>();
  *rowmajor->mutable_shape() = ShapeUtil::MakeShape(F32, {2, 2});
  *rowmajor->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  rowmajor->Reserve(4);
  rowmajor->Set<float>({0, 0}, 1.0);
  rowmajor->Set<float>({0, 1}, 2.0);
  rowmajor->Set<float>({1, 0}, 3.0);
  rowmajor->Set<float>({1, 1}, 4.0);

  EXPECT_EQ(*rowmajor, *colmajor);
}

TEST_F(LiteralUtilTest, TupleEquality) {
  // Test equality with tuples.
  auto scalar = Literal::CreateR0<float>(1.0);
  auto matrix = Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple1 = Literal::MakeTuple({scalar.get(), matrix.get()});

  // Tuple with the same elements. One element is shared with the original
  // tuple, the other is a clone of the element in the original tuple.
  auto scalar_clone = Literal::CreateR0<float>(1.0);
  auto tuple2 = Literal::MakeTuple({scalar_clone.get(), matrix.get()});
  EXPECT_EQ(*tuple1, *tuple2);

  // Tuple with elements reversed.
  auto reversed_tuple = Literal::MakeTuple({matrix.get(), scalar.get()});
  EXPECT_NE(*tuple1, *reversed_tuple);

  // Tuple with different value.
  auto scalar_42 = Literal::CreateR0<float>(42.0);
  auto different_tuple = Literal::MakeTuple({scalar_42.get(), matrix.get()});
  EXPECT_NE(*tuple1, *different_tuple);
}

TEST_F(LiteralUtilTest, C64Equality) {
  // Test equality with tuples.
  auto vector = Literal::CreateR1<complex64>({{1.0, 2.0}, {3.0, 4.0}});

  // Tuple with the same elements. One element is shared with the original
  // tuple, the other is a clone of the element in the original tuple.
  auto vector_clone = Literal::CreateR1<complex64>({{1.0, 2.0}, {3.0, 4.0}});
  EXPECT_EQ(*vector, *vector_clone);

  auto vector_reversed = Literal::CreateR1<complex64>({{3.0, 4.0}, {1.0, 2.0}});
  EXPECT_NE(*vector, *vector_reversed);
}

TEST_F(LiteralUtilTest, IsAllTuple) {
  auto element1 = Literal::CreateR0<float>(0.0);
  auto element2 = Literal::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});
  auto tuple = Literal::MakeTuple({element1.get(), element1.get()});

  // Tuples should always return false for IsAll.
  EXPECT_FALSE(tuple->IsAll(0));
  EXPECT_FALSE(tuple->IsAll(1));
}

// Verifies that CreateFromShape works for tuples.
TEST_F(LiteralUtilTest, CreateFromShapeTuple) {
  auto scalar = Literal::CreateR0<float>(0.0);
  auto matrix = Literal::CreateR2<int32>({{0, 0}, {0, 0}});
  auto tuple = Literal::MakeTuple({scalar.get(), matrix.get()});

  auto x = Literal::CreateFromShape(tuple->shape());
  EXPECT_EQ(*tuple, *x);
}

TEST_F(LiteralUtilTest, IsAll) {
  EXPECT_TRUE(Literal::CreateR0<bool>(false)->IsAll(0));
  EXPECT_TRUE(Literal::CreateR0<bool>(true)->IsAll(1));
  EXPECT_FALSE(Literal::CreateR0<bool>(false)->IsAll(1));
  EXPECT_FALSE(Literal::CreateR0<bool>(false)->IsAll(2));
  EXPECT_FALSE(Literal::CreateR0<bool>(true)->IsAll(0));
  EXPECT_FALSE(Literal::CreateR0<bool>(true)->IsAll(2));
  EXPECT_FALSE(Literal::CreateR0<bool>(true)->IsAll(-1));

  // We shouldn't reinterpret int8_min as an unsigned type and then decide that
  // it is equal to 255.
  auto int8_min = std::numeric_limits<int8>::min();
  EXPECT_FALSE(Literal::CreateR0<uint8>(255)->IsAll(int8_min));

  EXPECT_TRUE(Literal::CreateR0<float>(42.0)->IsAll(42));
  EXPECT_FALSE(Literal::CreateR0<float>(42.0001)->IsAll(42));

  EXPECT_TRUE(Literal::CreateR1<int>({100, 100, 100})->IsAll(100));
  EXPECT_FALSE(Literal::CreateR1<double>({100, 100, 100.001})->IsAll(100));

  EXPECT_TRUE(Literal::CreateR2<uint64>({{8, 8}, {8, 8}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<uint64>({{8, 8}, {8, 9}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<uint64>({{9, 8}, {8, 8}})->IsAll(8));

  half h8(8.0f);
  half h9(9.0f);
  EXPECT_TRUE(Literal::CreateR2<half>({{h8}, {h8}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<half>({{h8}, {h9}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<half>({{h9}, {h8}})->IsAll(8));

  bfloat16 b8(8.0f);
  bfloat16 b9(9.0f);

  EXPECT_TRUE(Literal::CreateR2<bfloat16>({{b8}, {b8}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<bfloat16>({{b8}, {b9}})->IsAll(8));
  EXPECT_FALSE(Literal::CreateR2<bfloat16>({{b9}, {b8}})->IsAll(8));

  // 9.001 will be truncated to 9.0
  bfloat16 b91(9.001f);
  bfloat16 b90(9.00f);
  EXPECT_TRUE(Literal::CreateR2<bfloat16>({{b91}, {b90}})->IsAll(9.0));

  complex64 c8_9 = {8, 9};
  EXPECT_FALSE(Literal::CreateR2<complex64>({{c8_9}, {c8_9}})->IsAll(8));

  auto uint64_max = std::numeric_limits<uint64>::max();
  EXPECT_FALSE(Literal::CreateR2<uint64>(
                   {{uint64_max, uint64_max}, {uint64_max, uint64_max}})
                   ->IsAll(-1));
}

TEST_F(LiteralUtilTest, IsAllFloat) {
  // IsAllFloat always returns false when the literal is not floating-point.
  EXPECT_FALSE(Literal::CreateR0<bool>(false)->IsAllFloat(0));
  EXPECT_FALSE(Literal::CreateR0<int8>(0)->IsAllFloat(0));
  EXPECT_FALSE(Literal::CreateR0<uint8>(0)->IsAllFloat(0));
  EXPECT_FALSE(Literal::CreateR0<int>(0)->IsAllFloat(0));

  EXPECT_TRUE(Literal::CreateR0<float>(0)->IsAllFloat(0));
  EXPECT_TRUE(Literal::CreateR0<float>(.5)->IsAllFloat(.5));
  EXPECT_TRUE(Literal::CreateR0<float>(-.5)->IsAllFloat(-.5));
  EXPECT_FALSE(Literal::CreateR0<float>(-.5)->IsAllFloat(-.49));
  EXPECT_FALSE(
      Literal::CreateR2<float>({{0, 0, 0}, {0, .1, 0}})->IsAllFloat(0));
  EXPECT_TRUE(
      Literal::CreateR2<float>({{.5, .5, .5}, {.5, .5, .5}})->IsAllFloat(.5));

  EXPECT_TRUE(Literal::CreateR0<double>(0)->IsAllFloat(0));
  EXPECT_TRUE(Literal::CreateR0<double>(.5)->IsAllFloat(.5));
  EXPECT_TRUE(Literal::CreateR0<double>(-.5)->IsAllFloat(-.5));
  EXPECT_FALSE(Literal::CreateR0<double>(-.5)->IsAllFloat(-.49));
  EXPECT_FALSE(
      Literal::CreateR2<double>({{0, 0, 0}, {0, .1, 0}})->IsAllFloat(0));
}

TEST_F(LiteralUtilTest, IsAllComplex) {
  // IsAllComplex always returns false when the literal is not complex.
  EXPECT_FALSE(Literal::CreateR0<bool>(false)->IsAllComplex(0));
  EXPECT_FALSE(Literal::CreateR0<int8>(0)->IsAllComplex(0));
  EXPECT_FALSE(Literal::CreateR0<uint8>(0)->IsAllComplex(0));
  EXPECT_FALSE(Literal::CreateR0<int>(0)->IsAllComplex(0));
  EXPECT_FALSE(Literal::CreateR0<float>(0)->IsAllComplex(0));
  EXPECT_FALSE(Literal::CreateR0<double>(0)->IsAllComplex(0));

  complex64 c8_9 = {8, 9};
  complex64 c7_9 = {7, 9};
  EXPECT_TRUE(Literal::CreateR2<complex64>({{c8_9}, {c8_9}})
                  ->IsAllComplex({8.0f, 9.0f}));
  EXPECT_FALSE(Literal::CreateR2<complex64>({{c7_9}, {c8_9}})
                   ->IsAllComplex({8.0f, 9.0f}));
  EXPECT_FALSE(Literal::CreateR2<complex64>({{c8_9}, {c7_9}})
                   ->IsAllComplex({8.0f, 9.0f}));
}

TEST_F(LiteralUtilTest, IsZero) {
  auto scalar_zero = Literal::CreateR0<float>(0.0f);
  auto scalar_one = Literal::CreateR0<float>(1.0f);
  EXPECT_TRUE(scalar_zero->IsZero({}));
  EXPECT_FALSE(scalar_one->IsZero({}));

  auto array = Literal::CreateR2<uint32>({{1, 2, 0, 3}, {1, 0, 1, 2}});
  EXPECT_FALSE(array->IsZero({0, 1}));
  EXPECT_TRUE(array->IsZero({0, 2}));
  EXPECT_TRUE(array->IsZero({1, 1}));
  EXPECT_FALSE(array->IsZero({1, 2}));

  auto complex_zero = Literal::CreateR0<complex64>(0.0f);
  auto complex_nonzero = Literal::CreateR0<complex64>(0.5f);
  EXPECT_TRUE(complex_zero->IsZero({}));
  EXPECT_FALSE(complex_nonzero->IsZero({}));
}

template <typename T>
class LiteralUtilTestTemplated : public ::testing::Test {};

using TestedTypes = ::testing::Types<float, int32, uint32, complex64>;
TYPED_TEST_CASE(LiteralUtilTestTemplated, TestedTypes);

TYPED_TEST(LiteralUtilTestTemplated, Relayout2x2) {
  // Make a non-integer for floating point types.
  TypeParam half = TypeParam(1) / TypeParam(2);
  auto data = Literal::CreateR2<TypeParam>({{half, 2}, {3, 4}});
  const Layout layout01 = LayoutUtil::MakeLayout({0, 1});
  const Layout layout10 = LayoutUtil::MakeLayout({1, 0});

  auto data01 = data->Relayout(layout01);
  EXPECT_TRUE(LayoutUtil::Equal(data01->shape().layout(), layout01));
  EXPECT_EQ(*data, *data01);

  auto data10 = data->Relayout(layout10);
  EXPECT_TRUE(LayoutUtil::Equal(data10->shape().layout(), layout10));
  EXPECT_EQ(*data, *data10);
}

TEST_F(LiteralUtilTest, ReshapeR0) {
  auto original = Literal::CreateR0<float>(1.7f);
  auto reshape = original->Reshape(/*shape=*/{}).ConsumeValueOrDie();
  EXPECT_EQ(*original, *reshape);
}

TEST_F(LiteralUtilTest, ReshapeR4) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = Literal::CreateR4WithLayout<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  // F32[1x3x4x2]
  auto expected = Literal::CreateR3WithLayout<float>({
    {{10, 11}, {12, 13}, {14, 15}, {16, 17}},
    {{18, 19}, {20, 21}, {22, 23}, {24, 25}},
    {{26, 27}, {28, 29}, {30, 31}, {32, 33}},
  }, layout_r3_dim0major_);
  // clang-format on
  auto reshape = original->Reshape({3, 4, 2}).ConsumeValueOrDie();

  EXPECT_EQ(*expected, *reshape);
}

TEST_F(LiteralUtilTest, ReshapeR4Dim0Minor) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = Literal::CreateR4WithLayout<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0minor_);
  // F32[1x3x4x2]
  auto expected = Literal::CreateR3WithLayout<float>({
    {{10, 11}, {12, 13}, {14, 15}, {16, 17}},
    {{18, 19}, {20, 21}, {22, 23}, {24, 25}},
    {{26, 27}, {28, 29}, {30, 31}, {32, 33}},
  }, layout_r3_dim0major_);
  // clang-format on
  auto reshape = original->Reshape({3, 4, 2}).ConsumeValueOrDie();

  EXPECT_EQ(*expected, *reshape);
}

TEST_F(LiteralUtilTest, TransposeR0) {
  auto original = Literal::CreateR0<float>(1.7f);
  auto reshape = original->Transpose(/*permutation=*/{});
  EXPECT_EQ(*original, *reshape);
}

TEST_F(LiteralUtilTest, TransposeR4) {
  // clang-format off
  // F32[1x3x2x4]
  auto original = Literal::CreateR4<float>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }});
  // clang-format on
  auto reshape = original->Transpose(/*permutation=*/{2, 3, 0, 1});

  reshape->EachCell<float>(
      [&](tensorflow::gtl::ArraySlice<int64> indices, float value) {
        EXPECT_EQ(value, original->Get<float>(
                             {indices[2], indices[3], indices[0], indices[1]}));
      });
}

TEST_F(LiteralUtilTest, TestR4RelayoutEquivalence) {
  // Tests that using Relayout on an array is equivalent to creating it in the
  // target layout in the first place.
  auto dim0minor_relaid_to_dim0major =
      literal_r4_2x2x3x3_dim0minor_->Relayout(layout_r4_dim0major_);
  EXPECT_EQ(*literal_r4_2x2x3x3_dim0major_, *dim0minor_relaid_to_dim0major);

  auto dim0major_relaid_to_dim0minor =
      literal_r4_2x2x3x3_dim0major_->Relayout(layout_r4_dim0minor_);
  EXPECT_EQ(*literal_r4_2x2x3x3_dim0minor_, *dim0major_relaid_to_dim0minor);
}

TEST_F(LiteralUtilTest, TestR2LinearLayout) {
  // Test expected memory layout of R2 dim0-minor (column-major) literal.
  auto mat_dim0minor = Literal::CreateR2WithLayout<int>({{1, 2, 3}, {4, 5, 6}},
                                                        layout_r2_dim0minor_);
  EXPECT_EQ(mat_dim0minor->s32s_size(), 6);
  EXPECT_THAT(mat_dim0minor->s32s(), ElementsAre(1, 4, 2, 5, 3, 6));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_mat_to_dim0major = mat_dim0minor->Relayout(layout_r2_dim0major_);
  EXPECT_THAT(relaid_mat_to_dim0major->s32s(), ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout of R2 created with dim0-major (row-major).
  auto mat_dim0major = Literal::CreateR2WithLayout<int>({{1, 2, 3}, {4, 5, 6}},
                                                        layout_r2_dim0major_);
  EXPECT_EQ(mat_dim0major->s32s_size(), 6);
  EXPECT_THAT(mat_dim0major->s32s(), ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_mat_to_dim0minor = mat_dim0major->Relayout(layout_r2_dim0minor_);
  EXPECT_THAT(relaid_mat_to_dim0minor->s32s(), ElementsAre(1, 4, 2, 5, 3, 6));
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
  auto lit_dim0minor =
      Literal::CreateR3FromArray3DWithLayout<int>(arr3d, layout_r3_dim0minor_);

  EXPECT_EQ(lit_dim0minor->s32s_size(), 12);
  std::vector<int> expected_dim0minor{1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12};
  EXPECT_THAT(lit_dim0minor->s32s(),
              testing::ElementsAreArray(expected_dim0minor));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_lit_to_dim0major = lit_dim0minor->Relayout(layout_r3_dim0major_);
  std::vector<int> expected_dim0major{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_THAT(relaid_lit_to_dim0major->s32s(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout of R3 created with dim0-major (row-major).
  auto lit_dim0major =
      Literal::CreateR3FromArray3DWithLayout<int>(arr3d, layout_r3_dim0major_);
  EXPECT_EQ(lit_dim0major->s32s_size(), 12);
  EXPECT_THAT(lit_dim0major->s32s(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_lit_to_dim0minor = lit_dim0major->Relayout(layout_r3_dim0minor_);
  EXPECT_THAT(relaid_lit_to_dim0minor->s32s(),
              testing::ElementsAreArray(expected_dim0minor));
}

TEST_F(LiteralUtilTest, SliceR0S32) {
  auto input = Literal::CreateR0<int32>(1);
  auto result = input->Slice({}, {});
  EXPECT_EQ(*input, *result);
}

TEST_F(LiteralUtilTest, SliceR1F32) {
  auto input = Literal::CreateR1<float>({1.0, 2.0, 3.0, 4.0, 5.0});
  auto result = input->Slice({3}, {4});
  auto expected = Literal::CreateR1<float>({4.0});
  EXPECT_EQ(*expected, *result);
}

TEST_F(LiteralUtilTest, SliceR2U32) {
  auto input_3x4 =
      Literal::CreateR2<uint32>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto result = input_3x4->Slice({0, 2}, {2, 4});
  auto expected = Literal::CreateR2<uint32>({{3, 4}, {7, 8}});
  EXPECT_EQ(*expected, *result);
}

TEST_F(LiteralUtilTest, SliceR3U32Full) {
  auto input_2x3x2 = Literal::CreateR3<uint32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  auto result = input_2x3x2->Slice({0, 0, 0}, {2, 3, 2});
  EXPECT_EQ(*input_2x3x2, *result);
}

TEST_F(LiteralUtilTest, PopulateR1S64) {
  Literal output;
  output.PopulateR1<int64>({77});
  auto expected = Literal::CreateR1<int64>({77});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateR1U64) {
  Literal output;
  output.PopulateR1<uint64>({{77, 88}});
  auto expected = Literal::CreateR1<uint64>({{77, 88}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateR1C64) {
  Literal output;
  output.PopulateR1<complex64>({{77, 88}});
  auto expected = Literal::CreateR1<complex64>({{77, 88}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateR2C64) {
  Literal output;
  output.PopulateR2<complex64>({{{7, 8}, {9, 10}}, {{1, 2}, {3, 4}}});
  auto expected =
      Literal::CreateR2<complex64>({{{7, 8}, {9, 10}}, {{1, 2}, {3, 4}}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0BF16) {
  Literal output;
  bfloat16 h(0.25f);
  output.PopulateWithValue<bfloat16>(h, {});
  auto expected = Literal::CreateR0<bfloat16>(h);
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1BF16) {
  Literal output;
  bfloat16 h(0.5f);
  output.PopulateWithValue<bfloat16>(h, {3});
  auto expected = Literal::CreateR1<bfloat16>({h, h, h});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2BF16) {
  Literal output;
  bfloat16 h(2.0f);
  output.PopulateWithValue<bfloat16>(h, {2, 2});
  auto expected = Literal::CreateR2<bfloat16>({{h, h}, {h, h}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F32) {
  Literal output;
  output.PopulateWithValue<float>(2.5f, {});
  auto expected = Literal::CreateR0<float>(2.5f);
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1S64) {
  Literal output;
  output.PopulateWithValue<int64>(-7, {3});
  auto expected = Literal::CreateR1<int64>({-7, -7, -7});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2U64) {
  Literal output;
  output.PopulateWithValue<uint64>(42, {2, 2});
  auto expected = Literal::CreateR2<uint64>({{42, 42}, {42, 42}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2C64) {
  Literal output;
  output.PopulateWithValue<complex64>({4, 2}, {2, 2});
  auto expected =
      Literal::CreateR2<complex64>({{{4, 2}, {4, 2}}, {{4, 2}, {4, 2}}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F16) {
  Literal output;
  half h(0.25f);
  output.PopulateWithValue<half>(h, {});
  auto expected = Literal::CreateR0<half>(h);
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR1F16) {
  Literal output;
  half h(0.5f);
  output.PopulateWithValue<half>(h, {3});
  auto expected = Literal::CreateR1<half>({h, h, h});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, PopulateWithValueR2F16) {
  Literal output;
  half h(2.0f);
  output.PopulateWithValue<half>(h, {2, 2});
  auto expected = Literal::CreateR2<half>({{h, h}, {h, h}});
  EXPECT_EQ(output, *expected);
}

TEST_F(LiteralUtilTest, ReplicateR2U32) {
  auto input =
      Literal::CreateR2<uint32>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto output = input->Replicate<uint32>(3);
  auto expected = Literal::CreateR3<uint32>(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}});
  EXPECT_EQ(*output, *expected);
}

TEST_F(LiteralUtilTest, Copy) {
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
    auto init_proc = [&](const std::vector<int64>& indexes) {
      source->Set(indexes, ++seqnr);
      return true;
    };
    ShapeUtil::ForEachIndex(source->shape(), zero_base, dimensions, step,
                            init_proc);

    auto blank = Literal::CreateFromShape(shape);
    const int64 src_base[] = {3, 1, 5, 7};
    const int64 dest_base[] = {6, 4, 12, 2};
    const int64 copy_size[] = {7, 8, 11, 9};
    TF_EXPECT_OK(blank->Copy(*source, src_base, dest_base, copy_size));

    std::vector<int64> source_indexes(TF_ARRAYSIZE(dimensions), 0);
    std::vector<int64> blank_indexes(TF_ARRAYSIZE(dimensions), 0);
    bool matched = true;
    auto check_proc = [&](const std::vector<int64>& indexes) {
      std::copy(indexes.begin(), indexes.end(), source_indexes.begin());
      std::transform(source_indexes.begin(), source_indexes.end(), src_base,
                     source_indexes.begin(), std::plus<int64>());
      std::copy(indexes.begin(), indexes.end(), blank_indexes.begin());
      std::transform(blank_indexes.begin(), blank_indexes.end(), dest_base,
                     blank_indexes.begin(), std::plus<int64>());
      auto bval = blank->Get<uint32>(blank_indexes);
      matched = (bval != 0 && bval == source->Get<uint32>(source_indexes));
      return matched;
    };

    ShapeUtil::ForEachIndex(source->shape(), zero_base, copy_size, step,
                            check_proc);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, CopyScalars) {
  auto zero = Literal::CreateR0<uint32>(0);
  auto nine = Literal::CreateR0<uint32>(9);
  TF_EXPECT_OK(zero->Copy(*nine, {}, {}, {}));
  EXPECT_EQ(*zero, *nine);

  auto vect = Literal::CreateR1<uint32>({3, 4, 9, 12, 5, 17, 21});
  TF_EXPECT_OK(zero->Copy(*vect, {5}, {}, {}));
  EXPECT_EQ(zero->Get<uint32>({}), 17);
  TF_EXPECT_OK(vect->Copy(*zero, {}, {4}, {}));
  EXPECT_EQ(vect->Get<uint32>({4}), 17);
}

TEST_F(LiteralUtilTest, CopyFromAndToZeroElement) {
  const Shape empty_r1_shape = ShapeUtil::MakeShape(F32, {0});
  const auto const_nine = Literal::CreateR1<float>({9});
  const auto const_empty = Literal::CreateFromShape(empty_r1_shape);

  {
    // Source contains dimension with zero elements.
    const auto empty = Literal::CreateFromShape(empty_r1_shape);
    auto nine = Literal::CreateR1<float>({9});

    TF_EXPECT_OK(nine->Copy(*empty, {0}, {0}, {0}));
    EXPECT_EQ(*nine, *const_nine);
  }

  {
    // Copy 0 element to destination with zero elements.
    const auto empty = Literal::CreateFromShape(empty_r1_shape);
    auto nine = Literal::CreateR1<float>({9});

    TF_EXPECT_OK(empty->Copy(*nine, {0}, {0}, {0}));
    EXPECT_EQ(*empty, *const_empty);
  }
}

TEST_F(LiteralUtilTest, F16) {
  // Verify that the internal data views are consistent and that they
  // are in little endian format
  // TODO - modify if we make the data format machine endianess dependent
  auto m1 = Literal::CreateFromShape(ShapeUtil::MakeShape(F16, {2, 2}));
  Literal* l1 = m1.get();
  const char* d1 = static_cast<const char*>(l1->InternalData());
  EXPECT_EQ(d1[0], 0);
  EXPECT_EQ(d1[1], 0);
  EXPECT_EQ(d1[2], 0);
  EXPECT_EQ(d1[3], 0);
  EXPECT_EQ(d1[4], 0);
  EXPECT_EQ(d1[5], 0);
  EXPECT_EQ(d1[6], 0);
  EXPECT_EQ(d1[7], 0);
  EXPECT_EQ(l1->InternalData(), l1->MutableInternalData());

  half h1(1.0f);
  half h2(2.0f);
  auto m2 = Literal::CreateR2<half>({{h1, h2}, {h2, h1}});
  Literal* l2 = m2.get();
  const char* d2 = static_cast<const char*>(l2->InternalData());
  EXPECT_EQ(d2[0], 0);
  EXPECT_EQ(d2[1], 0x3C);
  EXPECT_EQ(d2[2], 0);
  EXPECT_EQ(d2[3], 0x40);
  EXPECT_EQ(d2[4], 0);
  EXPECT_EQ(d2[5], 0x40);
  EXPECT_EQ(d2[6], 0);
  EXPECT_EQ(d2[7], 0x3C);
  EXPECT_EQ(l2->InternalData(), l2->MutableInternalData());
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
    auto literal = Literal::CreateFromShape(shape);
    auto generator = [&](tensorflow::gtl::ArraySlice<int64> indexes) -> uint32 {
      // Offsets from linear index just to avoid R0 literals to be initialized
      // with zero.
      return literal->LinearIndex(indexes) + 17;
    };
    TF_EXPECT_OK(literal->Populate<uint32>(generator));

    std::vector<int64> zero_base(data.dimensions.size(), 0);
    std::vector<int64> step(data.dimensions.size(), 1);
    bool matched = true;
    auto check_function = [&](const std::vector<int64>& indexes) {
      auto value = literal->Get<uint32>(indexes);
      matched = matched && (value == generator(indexes));
      return matched;
    };
    ShapeUtil::ForEachIndex(literal->shape(), zero_base, data.dimensions, step,
                            check_function);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, ConvertR4) {
  // clang-format off
  auto original = Literal::CreateR4WithLayout<int8>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  auto expected = Literal::CreateR4WithLayout<uint32>({{
     {{10, 11, 12, 13}, {14, 15, 16, 17}},
     {{18, 19, 20, 21}, {22, 23, 24, 25}},
     {{26, 27, 28, 29}, {30, 31, 32, 33}},
  }}, layout_r4_dim0major_);
  // clang-format on
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> converted,
                          original->Convert(U32));

  EXPECT_EQ(*expected, *converted);
}

TEST_F(LiteralUtilTest, ConvertIfTypesMatch) {
  // clang-format off
  auto s8 = Literal::CreateR4WithLayout<int8>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto s32 = Literal::CreateR4WithLayout<int32>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto u32 = Literal::CreateR4WithLayout<uint32>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto s64 = Literal::CreateR4WithLayout<int64>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto u64 = Literal::CreateR4WithLayout<uint64>({{
    {{10, 0, 12, 0}, {0, 15, 0, 17}},
    {{0, 19, 0, 21}, {22, 0, 24, 0}},
    {{26, 0, 28, 0}, {0, 31, 0, 33}},
  }}, layout_r4_dim0major_);
  auto pred = Literal::CreateR4WithLayout<bool>({{
    {{true, false, true, false}, {false, true, false, true}},
    {{false, true, false, true}, {true, false, true, false}},
    {{true, false, true, false}, {false, true, false, true}},
  }}, layout_r4_dim0major_);
  auto int32_pred = Literal::CreateR4WithLayout<int32>({{
    {{1, 0, 1, 0}, {0, 1, 0, 1}},
    {{0, 1, 0, 1}, {1, 0, 1, 0}},
    {{1, 0, 1, 0}, {0, 1, 0, 1}},
  }}, layout_r4_dim0major_);
  auto f16 = Literal::CreateR4WithLayout<half>({{
    {{half(10.0), half(0.0), half(12.0), half(0.0)},
     {half(0.0), half(15.0), half(0.0), half(17.0)}},
    {{half(0.0), half(19.0), half(0.0), half(21.0)},
     {half(22.0), half(0.0), half(24.0), half(0.0)}},
    {{half(26.0), half(0.0), half(28.0), half(0.0)},
     {half(0.0), half(31.0), half(0.0), half(33.0)}},
  }}, layout_r4_dim0major_);
  auto bf16 = Literal::CreateR4WithLayout<bfloat16>({{
    {{bfloat16(10.0), bfloat16(0.0), bfloat16(12.0), bfloat16(0.0)},
     {bfloat16(0.0), bfloat16(15.0), bfloat16(0.0), bfloat16(17.0)}},
    {{bfloat16(0.0), bfloat16(19.0), bfloat16(0.0), bfloat16(21.0)},
     {bfloat16(22.0), bfloat16(0.0), bfloat16(24.0), bfloat16(0.0)}},
    {{bfloat16(26.0), bfloat16(0.0), bfloat16(28.0), bfloat16(0.0)},
     {bfloat16(0.0), bfloat16(31.0), bfloat16(0.0), bfloat16(33.0)}},
  }}, layout_r4_dim0major_);
  auto f32 = Literal::CreateR4WithLayout<float>({{
    {{10.0f, 0.0f, 12.0f, 0.0f}, {0.0f, 15.0f, 0.0f, 17.0f}},
    {{0.0f, 19.0f, 0.0f, 21.0f}, {22.0f, 0.0f, 24.0f, 0.0f}},
    {{26.0f, 0.0f, 28.0f, 0.0f}, {0.0f, 31.0f, 0.0f, 33.0f}},
  }}, layout_r4_dim0major_);
  auto f64 = Literal::CreateR4WithLayout<double>({{
    {{10.0, 0.0, 12.0, 0.0}, {0.0, 15.0, 0.0, 17.0}},
    {{0.0, 19.0, 0.0, 21.0}, {22.0, 0.0, 24.0, 0.0}},
    {{26.0, 0.0, 28.0, 0.0}, {0.0, 31.0, 0.0, 33.0}},
  }}, layout_r4_dim0major_);
  auto c64 = Literal::CreateR4WithLayout<complex64>({{
    {{10.0f, 0.0f, 12.0f, 0.0f}, {0.0f, 15.0f, 0.0f, 17.0f}},
    {{0.0f, 19.0f, 0.0f, 21.0f}, {22.0f, 0.0f, 24.0f, 0.0f}},
    {{26.0f, 0.0f, 28.0f, 0.0f}, {0.0f, 31.0f, 0.0f, 33.0f}},
  }}, layout_r4_dim0major_);
  // clang-format on
  std::unique_ptr<Literal> conv;

  conv = s8->Convert(U32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *u32);

  conv = s8->Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *s32);

  conv = s8->Convert(U64).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *u64);

  conv = s8->Convert(S64).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *s64);

  conv = s8->Convert(PRED).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *pred);

  conv = bf16->Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *s32);

  conv = bf16->Convert(F32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f32);

  conv = pred->Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *int32_pred);

  conv = f32->Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *s32);

  conv = f64->Convert(S32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *s32);

  conv = s32->Convert(F32).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f32);

  conv = f32->Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f16);

  conv = f64->Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f16);

  conv = s32->Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f16);

  conv = u32->Convert(F16).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *f16);

  conv = s32->Convert(C64).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *c64);

  conv = f16->Convert(C64).ConsumeValueOrDie();
  EXPECT_EQ(*conv, *c64);

  EXPECT_EQ(s32->Convert(TUPLE).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(s32->Convert(S16).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(s32->Convert(U16).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(c64->Convert(F32).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(c64->Convert(S32).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(LiteralUtilTest, CopyFromProto_Bool) {
  LiteralProto p;
  p.mutable_shape()->set_element_type(PRED);
  for (int len = 0; len < 25; ++len) {
    p.mutable_shape()->clear_dimensions();
    p.mutable_shape()->add_dimensions(len);
    p.clear_preds();
    for (int i = 0; i < len; ++i) {
      p.add_preds((i % 2) == (len % 2));
    }

    Literal literal(p);
    ASSERT_EQ(len, literal.preds_size());
    int i = 0;
    for (auto it = literal.preds().begin(); it < literal.preds().end(); ++it) {
      EXPECT_EQ((i % 2) == (len % 2), *it);
      ++i;
    }
  }
}

// Note that f16 is currently stored in a byte array in little endian byte order
TEST_F(LiteralUtilTest, ToProto_f16) {
  half h1(1.0f);
  half h2(2.0f);

  auto m = Literal::CreateR2<half>({{h1, h2}, {h2, h1}});
  Literal* l = m.get();
  EXPECT_EQ(4, ShapeUtil::ElementsIn(l->shape()));
  EXPECT_EQ(4, l->f16s().size());
  EXPECT_EQ(4, l->f16s_size());

  LiteralProto p = l->ToProto();
  EXPECT_EQ(4, ShapeUtil::ElementsIn(p.shape()));
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
  p.clear_f16s();
  p.set_f16s(half_vals, 8);

  Literal literal(p);
  ASSERT_EQ(4, literal.f16s_size());
  ASSERT_EQ(h1, literal.f16s(0));
  ASSERT_EQ(h2, literal.f16s(1));
  ASSERT_EQ(h2, literal.f16s(2));
  ASSERT_EQ(h1, literal.f16s(3));

  const std::vector<half>& r = literal.f16s();
  ASSERT_EQ(4, r.size());
  ASSERT_EQ(h1, r[0]);
  ASSERT_EQ(h2, r[1]);
  ASSERT_EQ(h2, r[2]);
  ASSERT_EQ(h1, r[3]);
}

TEST_F(LiteralUtilTest, Subliterals) {
  auto scalar = Literal::CreateR0<float>(1.0);
  auto matrix = Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = Literal::MakeTuple({scalar.get(), matrix.get()});
  auto nested_tuple = Literal::MakeTuple({tuple.get(), scalar.get()});

  EXPECT_EQ(&scalar->GetSubliteral(/*index=*/{}), scalar.get());
  EXPECT_EQ(&matrix->GetSubliteral(/*index=*/{}), matrix.get());
  EXPECT_EQ(&tuple->GetSubliteral(/*index=*/{}), tuple.get());
  EXPECT_EQ(&nested_tuple->GetSubliteral(/*index=*/{}), nested_tuple.get());

  EXPECT_EQ(tuple->GetSubliteral(/*index=*/{0}), *scalar);
  EXPECT_EQ(tuple->GetSubliteral(/*index=*/{1}), *matrix);

  EXPECT_EQ(nested_tuple->GetSubliteral(/*index=*/{0}), *tuple);
  EXPECT_EQ(nested_tuple->GetSubliteral(/*index=*/{0, 0}), *scalar);
  EXPECT_EQ(nested_tuple->GetSubliteral(/*index=*/{0, 1}), *matrix);
  EXPECT_EQ(nested_tuple->GetSubliteral(/*index=*/{1}), *scalar);
}

}  // namespace
}  // namespace xla
