// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/utils/examples_iterable.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

class ExamplesIterableTest : public ::testing::Test {};

TEST_F(ExamplesIterableTest, Iterate) {
  // Create a batch of 8 examples having one dense float, two sparse float and
  // two sparse int features. Second sparse float feature is multivalent.
  // The data looks like the following:
  // Instance | DenseF1 | SparseF1 | SparseF2 | SparseI1 | SparseI2 |
  // 0        |   7     |   -3     |    |  1  |   1, 8   |          |
  // 1        |  -2     |          |  4 |     |    0     |    7     |
  // 2        |   8     |    0     |    |  3  |          |    13    |
  // 3        |   1     |    5     |  7 |     |   2, 0   |    4     |
  // 4        |   0     |    0     |    | 4.3 |          |    0     |
  // 5        |  -4     |          |  9 | 0.8 |          |          |
  // 6        |   7     |          |    |     |          |          |
  // 7        |  -2     |          | -4 |     |     5    |          |
  auto dense_float_tensor = test::AsTensor<float>(
      {7.0f, -2.0f, 8.0f, 1.0f, 0.0f, -4.0f, 7.0f, -2.0f}, {8, 1});
  auto sparse_float_indices1 =
      test::AsTensor<int64>({0, 0, 2, 0, 3, 0, 4, 0}, {4, 2});
  auto sparse_float_values1 = test::AsTensor<float>({-3.0f, 0.0f, 5.0f, 0.0f});
  auto sparse_float_shape1 = TensorShape({8, 1});
  sparse::SparseTensor sparse_float_tensor1(
      sparse_float_indices1, sparse_float_values1, sparse_float_shape1);
  auto sparse_float_indices2 = test::AsTensor<int64>(
      {0, 1, 1, 0, 2, 1, 3, 0, 4, 1, 5, 0, 5, 1, 7, 0}, {8, 2});
  auto sparse_float_values2 =
      test::AsTensor<float>({1.f, 4.0f, 3.f, 7.0f, 4.3f, 9.0f, 0.8f, -4.0f});
  auto sparse_float_shape2 = TensorShape({8, 2});
  sparse::SparseTensor sparse_float_tensor2(
      sparse_float_indices2, sparse_float_values2, sparse_float_shape2);
  auto sparse_int_indices1 =
      test::AsTensor<int64>({0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 7, 0}, {6, 2});
  auto sparse_int_values1 = test::AsTensor<int64>({1, 8, 0, 2, 0, 5});
  auto sparse_int_shape1 = TensorShape({8, 2});
  sparse::SparseTensor sparse_int_tensor1(
      sparse_int_indices1, sparse_int_values1, sparse_int_shape1);
  auto sparse_int_indices2 =
      test::AsTensor<int64>({1, 0, 2, 0, 3, 0, 4, 0}, {4, 2});
  auto sparse_int_values2 = test::AsTensor<int64>({7, 13, 4, 0});
  auto sparse_int_shape2 = TensorShape({8, 1});
  sparse::SparseTensor sparse_int_tensor2(
      sparse_int_indices2, sparse_int_values2, sparse_int_shape2);

  auto validate_example_features = [](int64 example_idx,
                                      const Example& example) {
    EXPECT_EQ(1, example.dense_float_features.size());

    switch (example_idx) {
      case 0: {
        EXPECT_EQ(0, example.example_idx);
        EXPECT_EQ(7.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_TRUE(example.sparse_float_features[0][0].has_value());
        EXPECT_EQ(-3.0f, example.sparse_float_features[0][0].get_value());
        // SparseF2 - multivalent.
        EXPECT_FALSE(example.sparse_float_features[1][0].has_value());
        EXPECT_TRUE(example.sparse_float_features[1][1].has_value());
        EXPECT_EQ(1.0f, example.sparse_float_features[1][1].get_value());

        EXPECT_EQ(2, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[0].count(1));
        EXPECT_EQ(1, example.sparse_int_features[0].count(8));
        EXPECT_EQ(0, example.sparse_int_features[1].size());
      } break;
      case 1: {
        EXPECT_EQ(1, example.example_idx);
        EXPECT_EQ(-2.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_FALSE(example.sparse_float_features[0][0].has_value());
        // SparseF2.
        EXPECT_TRUE(example.sparse_float_features[1][0].has_value());
        EXPECT_EQ(4.0f, example.sparse_float_features[1][0].get_value());
        EXPECT_FALSE(example.sparse_float_features[1][1].has_value());

        EXPECT_EQ(1, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[0].count(0));
        EXPECT_EQ(1, example.sparse_int_features[1].size());
        EXPECT_EQ(1, example.sparse_int_features[1].count(7));
      } break;
      case 2: {
        EXPECT_EQ(2, example.example_idx);
        EXPECT_EQ(8.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_TRUE(example.sparse_float_features[0][0].has_value());
        EXPECT_EQ(0.0f, example.sparse_float_features[0][0].get_value());
        // SparseF2.
        EXPECT_FALSE(example.sparse_float_features[1][0].has_value());
        EXPECT_TRUE(example.sparse_float_features[1][1].has_value());
        EXPECT_EQ(3.f, example.sparse_float_features[1][1].get_value());

        EXPECT_EQ(0, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[1].size());
        EXPECT_EQ(1, example.sparse_int_features[1].count(13));
      } break;
      case 3: {
        EXPECT_EQ(3, example.example_idx);
        EXPECT_EQ(1.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_TRUE(example.sparse_float_features[0][0].has_value());
        EXPECT_EQ(5.0f, example.sparse_float_features[0][0].get_value());
        // SparseF2.
        EXPECT_TRUE(example.sparse_float_features[1][0].has_value());
        EXPECT_EQ(7.0f, example.sparse_float_features[1][0].get_value());
        EXPECT_FALSE(example.sparse_float_features[1][1].has_value());

        EXPECT_EQ(2, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[0].count(2));
        EXPECT_EQ(1, example.sparse_int_features[0].count(0));
        EXPECT_EQ(1, example.sparse_int_features[1].size());
        EXPECT_EQ(1, example.sparse_int_features[1].count(4));
      } break;
      case 4: {
        EXPECT_EQ(4, example.example_idx);
        EXPECT_EQ(0.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_TRUE(example.sparse_float_features[0][0].has_value());
        EXPECT_EQ(0.0f, example.sparse_float_features[0][0].get_value());
        // SparseF2.
        EXPECT_FALSE(example.sparse_float_features[1][0].has_value());
        EXPECT_TRUE(example.sparse_float_features[1][1].has_value());
        EXPECT_EQ(4.3f, example.sparse_float_features[1][1].get_value());

        EXPECT_EQ(0, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[1].size());
        EXPECT_EQ(1, example.sparse_int_features[1].count(0));
      } break;
      case 5: {
        EXPECT_EQ(5, example.example_idx);
        EXPECT_EQ(-4.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_FALSE(example.sparse_float_features[0][0].has_value());
        // SparseF2.
        EXPECT_TRUE(example.sparse_float_features[1][0].has_value());
        EXPECT_EQ(9.0f, example.sparse_float_features[1][0].get_value());
        EXPECT_TRUE(example.sparse_float_features[1][1].has_value());
        EXPECT_EQ(0.8f, example.sparse_float_features[1][1].get_value());

        EXPECT_EQ(0, example.sparse_int_features[0].size());
      } break;
      case 6: {
        EXPECT_EQ(6, example.example_idx);
        EXPECT_EQ(7.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_FALSE(example.sparse_float_features[0][0].has_value());
        // SparseF2.
        EXPECT_FALSE(example.sparse_float_features[1][0].has_value());
        EXPECT_FALSE(example.sparse_float_features[1][1].has_value());

        EXPECT_EQ(0, example.sparse_int_features[0].size());
      } break;
      case 7: {
        EXPECT_EQ(7, example.example_idx);
        EXPECT_EQ(-2.0f, example.dense_float_features[0]);
        // SparseF1.
        EXPECT_FALSE(example.sparse_float_features[0][0].has_value());
        // SparseF2.
        EXPECT_TRUE(example.sparse_float_features[1][0].has_value());
        EXPECT_EQ(-4.0f, example.sparse_float_features[1][0].get_value());
        EXPECT_FALSE(example.sparse_float_features[1][1].has_value());

        EXPECT_EQ(1, example.sparse_int_features[0].size());
        EXPECT_EQ(1, example.sparse_int_features[0].count(5));
      } break;
      default: { LOG(QFATAL) << "Invalid example index."; } break;
    }
  };

  // Iterate through all examples sequentially.
  ExamplesIterable full_iterable(
      {dense_float_tensor}, {sparse_float_tensor1, sparse_float_tensor2},
      {sparse_int_tensor1, sparse_int_tensor2}, 0, 8);
  int64 example_idx = 0;

  for (const auto& example : full_iterable) {
    validate_example_features(example_idx, example);
    ++example_idx;
  }
  EXPECT_EQ(8, example_idx);

  // Iterate through slice (2, 6) of examples.
  ExamplesIterable slice_iterable(
      {dense_float_tensor}, {sparse_float_tensor1, sparse_float_tensor2},
      {sparse_int_tensor1, sparse_int_tensor2}, 2, 6);
  example_idx = 2;
  for (const auto& example : slice_iterable) {
    validate_example_features(example_idx, example);
    ++example_idx;
  }
  EXPECT_EQ(6, example_idx);
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
