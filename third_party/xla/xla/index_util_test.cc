/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/index_util.h"

#include <initializer_list>
#include <vector>

#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

void SetMinorToMajorLayout(Shape* shape, std::vector<int64_t> dimensions) {
  shape->mutable_layout()->clear_minor_to_major();
  for (auto dimension : dimensions) {
    shape->mutable_layout()->add_minor_to_major(dimension);
  }
}

TEST(IndexUtilTest, VectorIndexing) {
  // Vectors are trivially laid out and the linear index should always be the
  // same as the "multidimensional" index.
  Shape vector_shape = ShapeUtil::MakeShape(F32, {100});
  EXPECT_EQ(42,
            IndexUtil::MultidimensionalIndexToLinearIndex(vector_shape, {42}));
  auto multi_index =
      IndexUtil::LinearIndexToMultidimensionalIndex(vector_shape, 42);
  EXPECT_EQ(1, multi_index.size());
  EXPECT_EQ(42, multi_index[0]);
}

TEST(IndexUtilTest, MatrixIndexingRowMajor) {
  // Set layout to [0, 1]. That is, row major.
  Shape matrix_shape_01 = ShapeUtil::MakeShape(F32, {10, 20});
  SetMinorToMajorLayout(&matrix_shape_01, {0, 1});

  // If index is {a, b} then linear index should be: a + b * 10
  EXPECT_EQ(0, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                             {0, 0}));
  EXPECT_EQ(199, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                               {9, 19}));
  EXPECT_EQ(53, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                              {3, 5}));
  EXPECT_THAT(
      IndexUtil::LinearIndexToMultidimensionalIndex(matrix_shape_01, 53),
      testing::ElementsAre(3, 5));
}

TEST(IndexUtilTest, MatrixIndexingColumnMajor) {
  // Set layout to [1, 0]. That is, column major.
  Shape matrix_shape_10 = ShapeUtil::MakeShape(F32, {10, 20});
  SetMinorToMajorLayout(&matrix_shape_10, {1, 0});

  // If index is {a, b} then linear index should be: a * 20 + b
  EXPECT_EQ(0, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                             {0, 0}));
  EXPECT_EQ(199, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                               {9, 19}));
  EXPECT_EQ(65, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                              {3, 5}));
  EXPECT_THAT(
      IndexUtil::LinearIndexToMultidimensionalIndex(matrix_shape_10, 65),
      testing::ElementsAre(3, 5));
}

TEST(IndexUtilTest, ThreeDArrayIndexing210) {
  // Set layout to [2, 1, 0]. That is, column major.
  Shape shape_210 = ShapeUtil::MakeShape(F32, {10, 20, 30});
  SetMinorToMajorLayout(&shape_210, {2, 1, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b * 30 + c
  EXPECT_EQ(1957, IndexUtil::MultidimensionalIndexToLinearIndex(shape_210,
                                                                {3, 5, 7}));
  EXPECT_EQ(5277, IndexUtil::MultidimensionalIndexToLinearIndex(shape_210,
                                                                {8, 15, 27}));
}

TEST(IndexUtilTest, ThreeDArrayIndexing120) {
  // Set layout to [1, 2, 0]
  Shape shape_120 = ShapeUtil::MakeShape(F32, {10, 20, 30});
  SetMinorToMajorLayout(&shape_120, {1, 2, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b + c * 20
  EXPECT_EQ(1945, IndexUtil::MultidimensionalIndexToLinearIndex(shape_120,
                                                                {3, 5, 7}));
  EXPECT_EQ(5355, IndexUtil::MultidimensionalIndexToLinearIndex(shape_120,
                                                                {8, 15, 27}));
}

TEST(IndexUtilTest, FourDArrayIndexing3210) {
  // Set layout to [3, 2, 1,0]. That is, column major.
  Shape shape_3210 = ShapeUtil::MakeShape(F32, {10, 20, 30, 40});
  SetMinorToMajorLayout(&shape_3210, {3, 2, 1, 0});

  // If index is {a, b, c, d} then linear index should be:
  // a * 20 * 30 * 40 + b * 30 * 40 + c * 40 + d
  EXPECT_EQ(78289, IndexUtil::MultidimensionalIndexToLinearIndex(shape_3210,
                                                                 {3, 5, 7, 9}));
  EXPECT_EQ(211113, IndexUtil::MultidimensionalIndexToLinearIndex(
                        shape_3210, {8, 15, 27, 33}));
}

TEST(IndexUtilTest, LinearToMultiToLinear) {
  // Verify that converting a linear index to a multidimensional index and back
  // always returns the same value for different crazy shapes.  Shape has
  // 1440000000 elements. Inputs are randomly-ish selected.
  std::vector<int64_t> linear_indexes = {0,        1439999999, 1145567336,
                                         43883404, 617295214,  1117613654};

  std::vector<std::vector<int64_t>> minor_to_major_orders;
  minor_to_major_orders.push_back({6, 5, 4, 3, 2, 1, 0});
  minor_to_major_orders.push_back({0, 1, 2, 3, 4, 5, 6});
  minor_to_major_orders.push_back({4, 5, 1, 2, 6, 0, 3});

  for (auto minor_to_major_order : minor_to_major_orders) {
    Shape shape = ShapeUtil::MakeShape(F32, {10, 20, 30, 40, 30, 20, 10});
    SetMinorToMajorLayout(&shape, minor_to_major_order);
    for (auto linear_index : linear_indexes) {
      auto multi_index =
          IndexUtil::LinearIndexToMultidimensionalIndex(shape, linear_index);
      EXPECT_EQ(linear_index, IndexUtil::MultidimensionalIndexToLinearIndex(
                                  shape, multi_index));
    }
  }
}

TEST(IndexUtilTest, BumpIndices2x2) {
  auto shape = ShapeUtil::MakeShape(S32, {2, 2});
  std::vector<int64_t> indices = {0, 0};
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(0, 1));
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 0));
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 1));
  EXPECT_FALSE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
}

}  // namespace
}  // namespace xla
