/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/reduce_utils.h"

#include <gmock/gmock.h>

namespace tflite {
namespace reduce_utils {
namespace {

using ::testing::ElementsAreArray;

void TestFunction(const std::vector<int>& axis_in,
                  const std::vector<int>& shape_in,
                  const std::vector<int>& expected_axis_out,
                  const std::vector<int>& expected_shape_out) {
  int num_dims = shape_in.size();
  int expected_out_num_dims = expected_shape_out.size();
  int actual_out_num_dims;
  int expected_out_num_axis = expected_axis_out.size();
  int actual_out_num_axis;
  std::vector<int> actual_shape_out(num_dims);
  std::vector<int> actual_axis_out(num_dims);
  ResolveAxis(shape_in.size(), axis_in.data(), axis_in.size(),
              actual_axis_out.data(), actual_out_num_axis, shape_in.data(),
              actual_shape_out.data(), actual_out_num_dims);
  EXPECT_EQ(expected_out_num_dims, actual_out_num_dims);
  EXPECT_EQ(expected_out_num_axis, actual_out_num_axis);
  EXPECT_THAT(expected_shape_out,
              ElementsAreArray(actual_shape_out.data(), expected_out_num_dims));
  EXPECT_THAT(expected_axis_out,
              ElementsAreArray(actual_axis_out.data(), expected_out_num_axis));
}

TEST(ResolveAxisTest, Flatten_0_1_2) {
  const std::vector<int> axis_in = {0, 1, 2};
  const std::vector<int> shape_in = {2, 3, 4, 5};
  const std::vector<int> expected_shape_out{24, 5};
  const std::vector<int> expected_axis_out{0};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, Flatten_0_1_2_3) {
  const std::vector<int> axis_in = {3, 2};
  const std::vector<int> shape_in = {2, 3, 4, 5};
  const std::vector<int> expected_shape_out{6, 20};
  const std::vector<int> expected_axis_out{1};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, ZeroDims) {
  const std::vector<int> axis_in = {};
  const std::vector<int> shape_in = {};
  const std::vector<int> expected_shape_out{};
  const std::vector<int> expected_axis_out{};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, DoNothing) {
  const std::vector<int> axis_in = {0};
  const std::vector<int> shape_in = {4, 5};
  const std::vector<int> expected_shape_out{4, 5};
  const std::vector<int> expected_axis_out{0};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, NegativeAxis) {
  const std::vector<int> axis_in = {-2};
  const std::vector<int> shape_in = {4, 3};
  const std::vector<int> expected_shape_out{4, 3};
  const std::vector<int> expected_axis_out{0};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, NegativeAxisFold) {
  const std::vector<int> axis_in = {-1};
  const std::vector<int> shape_in = {4, 3, 5};
  const std::vector<int> expected_shape_out{12, 5};
  const std::vector<int> expected_axis_out{1};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, DuplicateAxis) {
  const std::vector<int> axis_in = {2, 1, 2, 1, 2, 1};
  const std::vector<int> shape_in = {4, 3, 2};
  const std::vector<int> expected_shape_out{4, 6};
  const std::vector<int> expected_axis_out{1};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, DuplicateNegativeAxis) {
  const std::vector<int> axis_in = {2, -1, -2, -1, 2, 1};
  const std::vector<int> shape_in = {4, 3, 2};
  const std::vector<int> expected_shape_out{4, 6};
  const std::vector<int> expected_axis_out{1};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, RemoveSize1Dim) {
  const std::vector<int> axis_in = {0};
  const std::vector<int> shape_in = {1, 4, 3, 1};
  const std::vector<int> expected_shape_out{4, 3};
  const std::vector<int> expected_axis_out{};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, OneSize1DimToScalar) {
  const std::vector<int> axis_in = {0};
  const std::vector<int> shape_in = {1};
  const std::vector<int> expected_shape_out{};
  const std::vector<int> expected_axis_out{};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

TEST(ResolveAxisTest, InterleavedSize1Dim) {
  const std::vector<int> axis_in = {1, 3};
  const std::vector<int> shape_in = {1, 2, 1, 4, 1, 7};
  const std::vector<int> expected_shape_out{8, 7};
  const std::vector<int> expected_axis_out{0};
  TestFunction(axis_in, shape_in, expected_axis_out, expected_shape_out);
}

}  // namespace
}  // namespace reduce_utils
}  // namespace tflite
