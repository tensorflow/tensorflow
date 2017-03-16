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
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

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
  EXPECT_MATCH(testing::PBToVec<tensorflow::protobuf_int64>(
                   literal->shape().dimensions()),
               testing::VectorMatcher<tensorflow::protobuf_int64>({2, 3, 2}));
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
  EXPECT_MATCH(
      testing::PBToVec(literal->shape().dimensions()),
      testing::VectorMatcher<tensorflow::protobuf_int64>({1, 2, 3, 2}));
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
  EXPECT_MATCH(
      testing::PBToVec<tensorflow::protobuf_int64>(
          literal_r4_2x2x3x3_dim0major_->shape().dimensions()),
      testing::VectorMatcher<tensorflow::protobuf_int64>({2, 2, 3, 3}));
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
  EXPECT_MATCH(testing::PBToVec<int32>(mat_dim0minor->s32s()),
               testing::VectorMatcher<int32>({1, 4, 2, 5, 3, 6}));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_mat_to_dim0major =
      LiteralUtil::Relayout(*mat_dim0minor, layout_r2_dim0major_);
  EXPECT_MATCH(testing::PBToVec<int32>(relaid_mat_to_dim0major->s32s()),
               testing::VectorMatcher<int32>({1, 2, 3, 4, 5, 6}));

  // Test expected memory layout of R2 created with dim0-major (row-major).
  auto mat_dim0major = LiteralUtil::CreateR2WithLayout<int>(
      {{1, 2, 3}, {4, 5, 6}}, layout_r2_dim0major_);
  EXPECT_EQ(mat_dim0major->s32s_size(), 6);
  EXPECT_MATCH(testing::PBToVec<int32>(mat_dim0major->s32s()),
               testing::VectorMatcher<int32>({1, 2, 3, 4, 5, 6}));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_mat_to_dim0minor =
      LiteralUtil::Relayout(*mat_dim0major, layout_r2_dim0minor_);
  EXPECT_MATCH(testing::PBToVec<int32>(relaid_mat_to_dim0minor->s32s()),
               testing::VectorMatcher<int32>({1, 4, 2, 5, 3, 6}));
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
  EXPECT_MATCH(testing::PBToVec<int32>(lit_dim0minor->s32s()),
               testing::VectorMatcher<int32>(expected_dim0minor));

  // Test expected memory layout when using Relayout to row major.
  auto relaid_lit_to_dim0major =
      LiteralUtil::Relayout(*lit_dim0minor, layout_r3_dim0major_);
  std::vector<int> expected_dim0major{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_MATCH(testing::PBToVec<int32>(relaid_lit_to_dim0major->s32s()),
               testing::VectorMatcher<int32>(expected_dim0major));

  // Test expected memory layout of R3 created with dim0-major (row-major).
  auto lit_dim0major = LiteralUtil::CreateR3FromArray3DWithLayout<int>(
      arr3d, layout_r3_dim0major_);
  EXPECT_EQ(lit_dim0major->s32s_size(), 12);
  EXPECT_MATCH(testing::PBToVec<int32>(lit_dim0major->s32s()),
               testing::VectorMatcher<int32>(expected_dim0major));

  // Test expected memory layout when using Relayout to column major.
  auto relaid_lit_to_dim0minor =
      LiteralUtil::Relayout(*lit_dim0major, layout_r3_dim0minor_);
  EXPECT_MATCH(testing::PBToVec<int32>(relaid_lit_to_dim0minor->s32s()),
               testing::VectorMatcher<int32>(expected_dim0minor));
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

}  // namespace
}  // namespace xla
