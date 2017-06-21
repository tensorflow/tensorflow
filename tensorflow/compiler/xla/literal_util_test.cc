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
  std::unique_ptr<Literal> literal_r4_2x2x3x3_dim0major_;
  std::unique_ptr<Literal> literal_r4_2x2x3x3_dim0minor_;
};

TEST_F(LiteralUtilTest, LiteralScalarToString) {
  auto true_lit = LiteralUtil::CreateR0<bool>(true);
  ASSERT_EQ("true", LiteralUtil::ToString(*true_lit));

  auto false_lit = LiteralUtil::CreateR0<bool>(false);
  ASSERT_EQ("false", LiteralUtil::ToString(*false_lit));

  auto u32_lit = LiteralUtil::CreateR0<uint32>(42);
  ASSERT_EQ("42", LiteralUtil::ToString(*u32_lit));

  auto s32_lit = LiteralUtil::CreateR0<int32>(-999);
  ASSERT_EQ("-999", LiteralUtil::ToString(*s32_lit));

  auto f32_lit = LiteralUtil::CreateR0<float>(3.14f);
  ASSERT_EQ("3.14", LiteralUtil::ToString(*f32_lit));

  auto f16_lit = LiteralUtil::CreateR0<half>(static_cast<half>(0.5f));
  ASSERT_EQ("0.5", LiteralUtil::ToString(*f16_lit));
}

TEST_F(LiteralUtilTest, LiteralVectorToString) {
  auto pred_vec = LiteralUtil::CreateR1<bool>({true, false, true});
  ASSERT_EQ("{101}", LiteralUtil::ToString(*pred_vec));
}

TEST_F(LiteralUtilTest, R2ToString) {
  const auto literal = LiteralUtil::CreateR2({{1, 2}, {3, 4}, {5, 6}});
  const string expected = R"(s32[3,2] {
  { 1, 2 },
  { 3, 4 },
  { 5, 6 },
})";
  ASSERT_EQ(expected, LiteralUtil::ToString(*literal));
}

TEST_F(LiteralUtilTest, R3ToString) {
  const auto literal =
      LiteralUtil::CreateR3({{{1}, {2}}, {{3}, {4}}, {{5}, {6}}});
  const string expected = R"(s32[3,2,1] {
{ { 1 },
  { 2 } },
{ { 3 },
  { 4 } },
{ { 5 },
  { 6 } }
})";
  ASSERT_EQ(expected, LiteralUtil::ToString(*literal));
}

TEST_F(LiteralUtilTest, TupleToString) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple = LiteralUtil::MakeTuple({scalar.get(), matrix.get()});
  const string expected = R"((f32[], f32[2,2]) (
1,
f32[2,2] {
  { 1, 2 },
  { 3, 4 },
},
))";
  ASSERT_EQ(expected, LiteralUtil::ToString(*tuple));
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
  EXPECT_THAT(literal->shape().dimensions(), ElementsAre(2, 3, 2));
  string result = LiteralUtil::ToString(*literal);
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
  auto literal = LiteralUtil::CreateR4Projected<float>({
    {1, 2},
    {1001, 1002},
    {2001, 2002},
  }, /*projection_p=*/1, /*projection_z=*/2);
  // clang-format on
  EXPECT_THAT(literal->shape().dimensions(), ElementsAre(1, 2, 3, 2));
  string result = LiteralUtil::ToString(*literal);
  const string expected = R"(f32[1,2,3,2] {
  {  // i0=0
    {  // i1=0
      {1, 2},
      {1001, 1002},
      {2001, 2002},
    },
    {  // i1=1
      {1, 2},
      {1001, 1002},
      {2001, 2002},
    },
  },
})";
  ASSERT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, LiteralR4F32Stringifies) {
  EXPECT_THAT(literal_r4_2x2x3x3_dim0major_->shape().dimensions(),
              ElementsAre(2, 2, 3, 3));
  string result = LiteralUtil::ToString(*literal_r4_2x2x3x3_dim0major_);
  const string expected = R"(f32[2,2,3,3] {
  {  // i0=0
    {  // i1=0
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
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
      {201, 202, 203},
      {204, 205, 206},
      {207, 208, 209},
    },
  },
})";
  ASSERT_EQ(expected, result);
}

TEST_F(LiteralUtilTest, EachCellR2F32) {
  // clang-format off
  auto literal = LiteralUtil::CreateR2<float>({
    {3.1f, 4.2f},
    {9.3f, 12.4f},
  });
  // clang-format on
  std::vector<std::tuple<int64, int64, string>> seen;
  LiteralUtil::EachCellAsString(
      *literal,
      [&seen](tensorflow::gtl::ArraySlice<int64> indices, const string& value) {
        seen.emplace_back(indices[0], indices[1], value);
      });

  using Elem = std::tuple<int64, int64, string>;
  std::vector<Elem> expected = {Elem(0, 0, "3.1"), Elem(0, 1, "4.2"),
                                Elem(1, 0, "9.3"), Elem(1, 1, "12.4")};
  EXPECT_EQ(expected, seen);
}

TEST_F(LiteralUtilTest, ScalarEquality) {
  // Test LiteralUtil::Equal with scalars.
  auto f32_42 = LiteralUtil::CreateR0<float>(42.0);
  auto f32_42_clone = LiteralUtil::CreateR0<float>(42.0);

  EXPECT_TRUE(LiteralUtil::Equal(*f32_42, *f32_42));
  EXPECT_TRUE(LiteralUtil::Equal(*f32_42, *f32_42_clone));

  auto f32_123 = LiteralUtil::CreateR0<float>(123.0);
  EXPECT_FALSE(LiteralUtil::Equal(*f32_42, *f32_123));

  auto f64_42 = LiteralUtil::CreateR0<double>(42.0);
  EXPECT_FALSE(LiteralUtil::Equal(*f32_42, *f64_42));
}

TEST_F(LiteralUtilTest, NonScalarEquality) {
  // Test LiteralUtil::Equal with nonscalars.
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_clone = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix_different =
      LiteralUtil::CreateR2<float>({{4.0, 3.0}, {1.0, 2.0}});
  auto vector_literal = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto scalar = LiteralUtil::CreateR0<float>(1.0);

  EXPECT_TRUE(LiteralUtil::Equal(*matrix, *matrix));
  EXPECT_TRUE(LiteralUtil::Equal(*matrix, *matrix_clone));
  EXPECT_FALSE(LiteralUtil::Equal(*matrix, *matrix_different));
  EXPECT_FALSE(LiteralUtil::Equal(*matrix, *vector_literal));
  EXPECT_FALSE(LiteralUtil::Equal(*matrix, *scalar));
}

TEST_F(LiteralUtilTest, DifferentLayoutEquality) {
  // Test LiteralUtil::Equal with literals which have different layouts.
  auto colmajor = MakeUnique<Literal>();
  *colmajor->mutable_shape() = ShapeUtil::MakeShape(F32, {2, 2});
  *colmajor->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  LiteralUtil::Reserve(4, colmajor.get());
  LiteralUtil::Set<float>(colmajor.get(), {0, 0}, 1.0);
  LiteralUtil::Set<float>(colmajor.get(), {0, 1}, 2.0);
  LiteralUtil::Set<float>(colmajor.get(), {1, 0}, 3.0);
  LiteralUtil::Set<float>(colmajor.get(), {1, 1}, 4.0);

  auto rowmajor = MakeUnique<Literal>();
  *rowmajor->mutable_shape() = ShapeUtil::MakeShape(F32, {2, 2});
  *rowmajor->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  LiteralUtil::Reserve(4, rowmajor.get());
  LiteralUtil::Set<float>(rowmajor.get(), {0, 0}, 1.0);
  LiteralUtil::Set<float>(rowmajor.get(), {0, 1}, 2.0);
  LiteralUtil::Set<float>(rowmajor.get(), {1, 0}, 3.0);
  LiteralUtil::Set<float>(rowmajor.get(), {1, 1}, 4.0);

  EXPECT_TRUE(LiteralUtil::Equal(*rowmajor, *colmajor));
}

TEST_F(LiteralUtilTest, TupleEquality) {
  // Test LiteralUtil::Equal with tuples.
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto tuple1 = LiteralUtil::MakeTuple({scalar.get(), matrix.get()});

  // Tuple with the same elements. One element is shared with the original
  // tuple, the other is a clone of the element in the original tuple.
  auto scalar_clone = LiteralUtil::CreateR0<float>(1.0);
  auto tuple2 = LiteralUtil::MakeTuple({scalar_clone.get(), matrix.get()});
  EXPECT_TRUE(LiteralUtil::Equal(*tuple1, *tuple2));

  // Tuple with elements reversed.
  auto reversed_tuple = LiteralUtil::MakeTuple({matrix.get(), scalar.get()});
  EXPECT_FALSE(LiteralUtil::Equal(*tuple1, *reversed_tuple));

  // Tuple with different value.
  auto scalar_42 = LiteralUtil::CreateR0<float>(42.0);
  auto different_tuple =
      LiteralUtil::MakeTuple({scalar_42.get(), matrix.get()});
  EXPECT_FALSE(LiteralUtil::Equal(*tuple1, *different_tuple));
}

TEST_F(LiteralUtilTest, IsAllTuple) {
  auto element1 = LiteralUtil::CreateR0<float>(0.0);
  auto element2 = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});
  auto tuple = LiteralUtil::MakeTuple({element1.get(), element1.get()});

  // Tuples should always return false for IsAll.
  EXPECT_FALSE(LiteralUtil::IsAll(*tuple, 0));
  EXPECT_FALSE(LiteralUtil::IsAll(*tuple, 1));
}

TEST_F(LiteralUtilTest, IsAll) {
  EXPECT_TRUE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(false), 0));
  EXPECT_TRUE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(true), 1));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(false), 1));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(false), 2));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(true), 0));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(true), 2));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<bool>(true), -1));

  // We shouldn't reinterpret int8_min as an unsigned type and then decide that
  // it is equal to 255.
  auto int8_min = std::numeric_limits<int8>::min();
  EXPECT_FALSE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR0<uint8>(255), int8_min));

  EXPECT_TRUE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<float>(42.0), 42));
  EXPECT_FALSE(LiteralUtil::IsAll(*LiteralUtil::CreateR0<float>(42.0001), 42));

  EXPECT_TRUE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR1<int>({100, 100, 100}), 100));
  EXPECT_FALSE(LiteralUtil::IsAll(
      *LiteralUtil::CreateR1<double>({100, 100, 100.001}), 100));

  EXPECT_TRUE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<uint64>({{8, 8}, {8, 8}}), 8));
  EXPECT_FALSE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<uint64>({{8, 8}, {8, 9}}), 8));
  EXPECT_FALSE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<uint64>({{9, 8}, {8, 8}}), 8));

  half h8(8.0f);
  half h9(9.0f);
  EXPECT_TRUE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<half>({{h8}, {h8}}), 8));
  EXPECT_FALSE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<half>({{h8}, {h9}}), 8));
  EXPECT_FALSE(
      LiteralUtil::IsAll(*LiteralUtil::CreateR2<half>({{h9}, {h8}}), 8));

  auto uint64_max = std::numeric_limits<uint64>::max();
  EXPECT_FALSE(LiteralUtil::IsAll(
      *LiteralUtil::CreateR2<uint64>(
          {{uint64_max, uint64_max}, {uint64_max, uint64_max}}),
      -1));
}

TEST_F(LiteralUtilTest, IsAllFloat) {
  // IsAllFloat always returns false when the literal is not floating-point.
  EXPECT_FALSE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<bool>(false), 0));
  EXPECT_FALSE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<int8>(0), 0));
  EXPECT_FALSE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<uint8>(0), 0));
  EXPECT_FALSE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<int>(0), 0));

  EXPECT_TRUE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<float>(0), 0));
  EXPECT_TRUE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<float>(.5), .5));
  EXPECT_TRUE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<float>(-.5), -.5));
  EXPECT_FALSE(
      LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<float>(-.5), -.49));
  EXPECT_FALSE(LiteralUtil::IsAllFloat(
      *LiteralUtil::CreateR2<float>({{0, 0, 0}, {0, .1, 0}}), 0));
  EXPECT_TRUE(LiteralUtil::IsAllFloat(
      *LiteralUtil::CreateR2<float>({{.5, .5, .5}, {.5, .5, .5}}), .5));

  EXPECT_TRUE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<double>(0), 0));
  EXPECT_TRUE(LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<double>(.5), .5));
  EXPECT_TRUE(
      LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<double>(-.5), -.5));
  EXPECT_FALSE(
      LiteralUtil::IsAllFloat(*LiteralUtil::CreateR0<double>(-.5), -.49));
  EXPECT_FALSE(LiteralUtil::IsAllFloat(
      *LiteralUtil::CreateR2<double>({{0, 0, 0}, {0, .1, 0}}), 0));
}

TEST_F(LiteralUtilTest, IsZero) {
  auto scalar_zero = LiteralUtil::CreateR0<float>(0.0f);
  auto scalar_one = LiteralUtil::CreateR0<float>(1.0f);
  EXPECT_TRUE(LiteralUtil::IsZero(*scalar_zero, {}));
  EXPECT_FALSE(LiteralUtil::IsZero(*scalar_one, {}));

  auto array = LiteralUtil::CreateR2<uint32>({{1, 2, 0, 3}, {1, 0, 1, 2}});
  EXPECT_FALSE(LiteralUtil::IsZero(*array, {0, 1}));
  EXPECT_TRUE(LiteralUtil::IsZero(*array, {0, 2}));
  EXPECT_TRUE(LiteralUtil::IsZero(*array, {1, 1}));
  EXPECT_FALSE(LiteralUtil::IsZero(*array, {1, 2}));
}

template <typename T>
class LiteralUtilTestTemplated : public ::testing::Test {};

using TestedTypes = ::testing::Types<float, int32, uint32>;
TYPED_TEST_CASE(LiteralUtilTestTemplated, TestedTypes);

TYPED_TEST(LiteralUtilTestTemplated, Relayout2x2) {
  // Make a non-integer for floating point types.
  TypeParam half = TypeParam(1) / TypeParam(2);
  auto data = LiteralUtil::CreateR2<TypeParam>({{half, 2}, {3, 4}});
  const Layout layout01 = LayoutUtil::MakeLayout({0, 1});
  const Layout layout10 = LayoutUtil::MakeLayout({1, 0});

  auto data01 = LiteralUtil::Relayout(*data, layout01);
  EXPECT_TRUE(LayoutUtil::Equal(data01->shape().layout(), layout01));
  EXPECT_TRUE(LiteralUtil::Equal(*data, *data01));

  auto data10 = LiteralUtil::Relayout(*data, layout10);
  EXPECT_TRUE(LayoutUtil::Equal(data10->shape().layout(), layout10));
  EXPECT_TRUE(LiteralUtil::Equal(*data, *data10));
}

TEST_F(LiteralUtilTest, ReshapeR0) {
  auto original = LiteralUtil::CreateR0<float>(1.7f);
  auto reshape =
      LiteralUtil::Reshape(*original, /*shape=*/{}).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*original, *reshape));
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
  auto reshape = LiteralUtil::Reshape(*original, {3, 4, 2}).ConsumeValueOrDie();

  EXPECT_TRUE(LiteralUtil::Equal(*expected, *reshape));
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
  auto reshape = LiteralUtil::Reshape(*original, {3, 4, 2}).ConsumeValueOrDie();

  EXPECT_TRUE(LiteralUtil::Equal(*expected, *reshape));
}

TEST_F(LiteralUtilTest, TransposeR0) {
  auto original = LiteralUtil::CreateR0<float>(1.7f);
  auto reshape = LiteralUtil::Transpose(*original, /*permutation=*/{});
  EXPECT_TRUE(LiteralUtil::Equal(*original, *reshape));
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
  auto reshape =
      LiteralUtil::Transpose(*original, /*permutation=*/{2, 3, 0, 1});

  LiteralUtil::EachCell<float>(
      *reshape, [&](tensorflow::gtl::ArraySlice<int64> indices, float value) {
        EXPECT_EQ(value,
                  LiteralUtil::Get<float>(*original, {indices[2], indices[3],
                                                      indices[0], indices[1]}));
      });
}

TEST_F(LiteralUtilTest, TestR4RelayoutEquivalence) {
  // Tests that using Relayout on an array is equivalent to creating it in the
  // target layout in the first place.
  auto dim0minor_relaid_to_dim0major = LiteralUtil::Relayout(
      *literal_r4_2x2x3x3_dim0minor_, layout_r4_dim0major_);
  EXPECT_TRUE(LiteralUtil::Equal(*literal_r4_2x2x3x3_dim0major_,
                                 *dim0minor_relaid_to_dim0major));

  auto dim0major_relaid_to_dim0minor = LiteralUtil::Relayout(
      *literal_r4_2x2x3x3_dim0major_, layout_r4_dim0minor_);
  EXPECT_TRUE(LiteralUtil::Equal(*literal_r4_2x2x3x3_dim0minor_,
                                 *dim0major_relaid_to_dim0minor));
}

TEST_F(LiteralUtilTest, TestR2LinearLayout) {
  // Test expected memory layout of R2 dim0-minor (column-major) literal.
  auto mat_dim0minor = LiteralUtil::CreateR2WithLayout<int>(
      {{1, 2, 3}, {4, 5, 6}}, layout_r2_dim0minor_);
  EXPECT_EQ(mat_dim0minor->s32s_size(), 6);
  EXPECT_THAT(mat_dim0minor->s32s(), ElementsAre(1, 4, 2, 5, 3, 6));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_mat_to_dim0major =
      LiteralUtil::Relayout(*mat_dim0minor, layout_r2_dim0major_);
  EXPECT_THAT(relaid_mat_to_dim0major->s32s(), ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout of R2 created with dim0-major (row-major).
  auto mat_dim0major = LiteralUtil::CreateR2WithLayout<int>(
      {{1, 2, 3}, {4, 5, 6}}, layout_r2_dim0major_);
  EXPECT_EQ(mat_dim0major->s32s_size(), 6);
  EXPECT_THAT(mat_dim0major->s32s(), ElementsAre(1, 2, 3, 4, 5, 6));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_mat_to_dim0minor =
      LiteralUtil::Relayout(*mat_dim0major, layout_r2_dim0minor_);
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
  auto lit_dim0minor = LiteralUtil::CreateR3FromArray3DWithLayout<int>(
      arr3d, layout_r3_dim0minor_);

  EXPECT_EQ(lit_dim0minor->s32s_size(), 12);
  std::vector<int> expected_dim0minor{1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12};
  EXPECT_THAT(lit_dim0minor->s32s(),
              testing::ElementsAreArray(expected_dim0minor));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_lit_to_dim0major =
      LiteralUtil::Relayout(*lit_dim0minor, layout_r3_dim0major_);
  std::vector<int> expected_dim0major{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_THAT(relaid_lit_to_dim0major->s32s(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout of R3 created with dim0-major (row-major).
  auto lit_dim0major = LiteralUtil::CreateR3FromArray3DWithLayout<int>(
      arr3d, layout_r3_dim0major_);
  EXPECT_EQ(lit_dim0major->s32s_size(), 12);
  EXPECT_THAT(lit_dim0major->s32s(),
              testing::ElementsAreArray(expected_dim0major));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_lit_to_dim0minor =
      LiteralUtil::Relayout(*lit_dim0major, layout_r3_dim0minor_);
  EXPECT_THAT(relaid_lit_to_dim0minor->s32s(),
              testing::ElementsAreArray(expected_dim0minor));
}

TEST_F(LiteralUtilTest, SliceR0S32) {
  auto input = LiteralUtil::CreateR0<int32>(1);
  auto result = LiteralUtil::Slice(*input, {}, {});
  EXPECT_TRUE(LiteralUtil::Equal(*input, *result));
}

TEST_F(LiteralUtilTest, SliceR1F32) {
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0, 5.0});
  auto result = LiteralUtil::Slice(*input, {3}, {4});
  auto expected = LiteralUtil::CreateR1<float>({4.0});
  EXPECT_TRUE(LiteralUtil::Equal(*expected, *result));
}

TEST_F(LiteralUtilTest, SliceR2U32) {
  auto input_3x4 = LiteralUtil::CreateR2<uint32>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto result = LiteralUtil::Slice(*input_3x4, {0, 2}, {2, 4});
  auto expected = LiteralUtil::CreateR2<uint32>({{3, 4}, {7, 8}});
  EXPECT_TRUE(LiteralUtil::Equal(*expected, *result));
}

TEST_F(LiteralUtilTest, SliceR3U32Full) {
  auto input_2x3x2 = LiteralUtil::CreateR3<uint32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  auto result = LiteralUtil::Slice(*input_2x3x2, {0, 0, 0}, {2, 3, 2});
  EXPECT_TRUE(LiteralUtil::Equal(*input_2x3x2, *result));
}

TEST_F(LiteralUtilTest, PopulateR1S64) {
  Literal output;
  LiteralUtil::PopulateR1<int64>({77}, &output);
  auto expected = LiteralUtil::CreateR1<int64>({77});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateR2U64) {
  Literal output;
  LiteralUtil::PopulateR1<uint64>({{77, 88}}, &output);
  auto expected = LiteralUtil::CreateR1<uint64>({{77, 88}});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F32) {
  Literal output;
  LiteralUtil::PopulateWithValue<float>(2.5f, {}, &output);
  auto expected = LiteralUtil::CreateR0<float>(2.5f);
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR1S64) {
  Literal output;
  LiteralUtil::PopulateWithValue<int64>(-7, {3}, &output);
  auto expected = LiteralUtil::CreateR1<int64>({-7, -7, -7});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR2U64) {
  Literal output;
  LiteralUtil::PopulateWithValue<uint64>(42, {2, 2}, &output);
  auto expected = LiteralUtil::CreateR2<uint64>({{42, 42}, {42, 42}});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR0F16) {
  Literal output;
  half h(0.25f);
  LiteralUtil::PopulateWithValue<half>(h, {}, &output);
  auto expected = LiteralUtil::CreateR0<half>(h);
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR1F16) {
  Literal output;
  half h(0.5f);
  LiteralUtil::PopulateWithValue<half>(h, {3}, &output);
  auto expected = LiteralUtil::CreateR1<half>({h, h, h});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, PopulateWithValueR2F16) {
  Literal output;
  half h(2.0f);
  LiteralUtil::PopulateWithValue<half>(h, {2, 2}, &output);
  auto expected = LiteralUtil::CreateR2<half>({{h, h}, {h, h}});
  EXPECT_TRUE(LiteralUtil::Equal(output, *expected));
}

TEST_F(LiteralUtilTest, ReplicateR2U32) {
  auto input = LiteralUtil::CreateR2<uint32>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  auto output = LiteralUtil::Replicate<uint32>(*input, 3);
  auto expected = LiteralUtil::CreateR3<uint32>(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}});
  EXPECT_TRUE(LiteralUtil::Equal(*output, *expected));
}

TEST_F(LiteralUtilTest, Copy) {
  const int64 dimensions[] = {17, 15, 34, 21};
  const int64 layouts[][4] = {
      {3, 2, 1, 0}, {0, 2, 1, 3}, {0, 1, 2, 3}, {2, 0, 3, 1}, {1, 3, 0, 2}};
  for (const auto& layout : layouts) {
    Shape shape = ShapeUtil::MakeShapeWithLayout(
        primitive_util::NativeToPrimitiveType<uint32>(), dimensions, layout);
    auto blank = LiteralUtil::CreateFromShape(shape);
    auto source = LiteralUtil::CreateFromShape(shape);
    const int64 zero_base[] = {0, 0, 0, 0};
    const int64 step[] = {1, 1, 1, 1};
    uint32 seqnr = 0;
    auto init_proc = [&](const std::vector<int64>& indexes) {
      LiteralUtil::Set(source.get(), indexes, ++seqnr);
      return true;
    };

    ShapeUtil::ForEachIndex(source->shape(), zero_base, dimensions, step,
                            init_proc);

    const int64 src_base[] = {3, 1, 5, 7};
    const int64 dest_base[] = {6, 4, 12, 2};
    const int64 copy_size[] = {7, 8, 11, 9};

    TF_EXPECT_OK(LiteralUtil::Copy(*source, src_base, blank.get(), dest_base,
                                   copy_size));
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
      auto bval = LiteralUtil::Get<uint32>(*blank, blank_indexes);
      matched = (bval != 0 &&
                 bval == LiteralUtil::Get<uint32>(*source, source_indexes));
      return matched;
    };
    ShapeUtil::ForEachIndex(source->shape(), zero_base, copy_size, step,
                            check_proc);
    EXPECT_TRUE(matched);
  }
}

TEST_F(LiteralUtilTest, CopyScalars) {
  auto zero = LiteralUtil::CreateR0<uint32>(0);
  auto nine = LiteralUtil::CreateR0<uint32>(9);
  TF_EXPECT_OK(LiteralUtil::Copy(*nine, {}, zero.get(), {}, {}));
  EXPECT_TRUE(LiteralUtil::Equal(*zero, *nine));

  auto vect = LiteralUtil::CreateR1<uint32>({3, 4, 9, 12, 5, 17, 21});
  TF_EXPECT_OK(LiteralUtil::Copy(*vect, {5}, zero.get(), {}, {}));
  EXPECT_EQ(LiteralUtil::Get<uint32>(*zero, {}), 17);
  TF_EXPECT_OK(LiteralUtil::Copy(*zero, {}, vect.get(), {4}, {}));
  EXPECT_EQ(LiteralUtil::Get<uint32>(*vect, {4}), 17);
}

TEST_F(LiteralUtilTest, F16) {
  // Verify that the internal data views are consistent and that they
  // are in little endian format
  // TODO - modify if we make the data format machine endianess dependent
  auto m1 = LiteralUtil::CreateFromShape(ShapeUtil::MakeShape(F16, {2, 2}));
  Literal* l1 = m1.get();
  const char* d1 = static_cast<const char*>(LiteralUtil::InternalData(*l1));
  EXPECT_EQ(d1[0], 0);
  EXPECT_EQ(d1[1], 0);
  EXPECT_EQ(d1[2], 0);
  EXPECT_EQ(d1[3], 0);
  EXPECT_EQ(d1[4], 0);
  EXPECT_EQ(d1[5], 0);
  EXPECT_EQ(d1[6], 0);
  EXPECT_EQ(d1[7], 0);
  EXPECT_EQ(LiteralUtil::InternalData(*l1),
            LiteralUtil::MutableInternalData(l1));

  half h1(1.0f);
  half h2(2.0f);
  auto m2 = LiteralUtil::CreateR2<half>({{h1, h2}, {h2, h1}});
  Literal* l2 = m2.get();
  const char* d2 = static_cast<const char*>(LiteralUtil::InternalData(*l2));
  EXPECT_EQ(d2[0], 0);
  EXPECT_EQ(d2[1], 0x3C);
  EXPECT_EQ(d2[2], 0);
  EXPECT_EQ(d2[3], 0x40);
  EXPECT_EQ(d2[4], 0);
  EXPECT_EQ(d2[5], 0x40);
  EXPECT_EQ(d2[6], 0);
  EXPECT_EQ(d2[7], 0x3C);
  EXPECT_EQ(LiteralUtil::InternalData(*l2),
            LiteralUtil::MutableInternalData(l2));
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
    auto literal = LiteralUtil::CreateFromShape(shape);
    auto generator = [&](tensorflow::gtl::ArraySlice<int64> indexes) -> uint32 {
      // Offsets from linear index just to avoid R0 literals to be initialized
      // with zero.
      return LiteralUtil::LinearIndex(*literal, indexes) + 17;
    };
    TF_EXPECT_OK(LiteralUtil::Populate<uint32>(literal.get(), generator));

    std::vector<int64> zero_base(data.dimensions.size(), 0);
    std::vector<int64> step(data.dimensions.size(), 1);
    bool matched = true;
    auto check_function = [&](const std::vector<int64>& indexes) {
      auto value = LiteralUtil::Get<uint32>(*literal, indexes);
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
  auto converted = LiteralUtil::Convert<int8, uint32>(*original);

  EXPECT_TRUE(LiteralUtil::Equal(*expected, *converted));
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
  // clang-format on
  std::unique_ptr<Literal> conv;

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s8, U32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *u32));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s8, S32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *s32));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s8, U64).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *u64));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s8, S64).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *s64));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s8, PRED).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *pred));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*pred, S32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *int32_pred));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*f32, S32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *s32));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*f64, S32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *s32));

  conv = LiteralUtil::ConvertIfSrcTypeMatches(*s32, F32).ConsumeValueOrDie();
  EXPECT_TRUE(LiteralUtil::Equal(*conv, *f32));

  EXPECT_EQ(LiteralUtil::ConvertIfSrcTypeMatches(*s32, TUPLE).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(LiteralUtil::ConvertIfSrcTypeMatches(*s32, F16).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(LiteralUtil::ConvertIfSrcTypeMatches(*s32, S16).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(LiteralUtil::ConvertIfSrcTypeMatches(*s32, U16).status().code(),
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

  const char half_vals[8] = {
    0x00, 0x3C, 0x00, 0x40, 0x00, 0x40, 0x00, 0x3C
  };
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


}  // namespace
}  // namespace xla
