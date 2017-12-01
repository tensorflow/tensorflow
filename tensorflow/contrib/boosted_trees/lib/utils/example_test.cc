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
#include "tensorflow/contrib/boosted_trees/lib/utils/example.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

class ExampleTest : public ::testing::Test {};

TEST_F(ExampleTest, TestSparseMatrix) {
  // Create the following matrix:
  // row id |   | 0.4 |  0.3
  // 0      | 1 |     |   2
  // 1      | 3 |  1  |   5
  // 2      |   |     |  -4
  // 3      |   |     |
  SparseMatrix<float> matrix;
  matrix.addElement(0, 1, 0.4f);
  matrix.addElement(0, 2, 0.3f);
  matrix.addElement(1, 0, 1.f);
  matrix.addElement(1, 2, 2.f);
  matrix.addElement(2, 0, 3.f);
  matrix.addElement(2, 1, 1.f);
  matrix.addElement(2, 2, 5.f);
  matrix.addElement(3, 2, -4.f);

  // Row 0.
  EXPECT_FALSE(matrix[0][0].has_value());
  EXPECT_TRUE(matrix[0][1].has_value());
  EXPECT_EQ(0.4f, matrix[0][1].get_value());
  EXPECT_TRUE(matrix[0][2].has_value());
  EXPECT_EQ(0.3f, matrix[0][2].get_value());

  // Row 1.
  EXPECT_TRUE(matrix[1][0].has_value());
  EXPECT_EQ(1.f, matrix[1][0].get_value());
  EXPECT_FALSE(matrix[1][1].has_value());
  EXPECT_TRUE(matrix[1][2].has_value());
  EXPECT_EQ(2.f, matrix[1][2].get_value());

  // Row 2.
  EXPECT_TRUE(matrix[2][0].has_value());
  EXPECT_EQ(3.f, matrix[2][0].get_value());
  EXPECT_TRUE(matrix[2][1].has_value());
  EXPECT_EQ(1.f, matrix[2][1].get_value());
  EXPECT_TRUE(matrix[2][2].has_value());
  EXPECT_EQ(5.f, matrix[2][2].get_value());

  // Row 3.
  EXPECT_FALSE(matrix[3][0].has_value());
  EXPECT_FALSE(matrix[3][1].has_value());
  EXPECT_TRUE(matrix[3][2].has_value());
  EXPECT_EQ(-4.f, matrix[3][2].get_value());

  // Row 4.
  EXPECT_FALSE(matrix[4][0].has_value());
  EXPECT_FALSE(matrix[4][1].has_value());
  EXPECT_FALSE(matrix[4][2].has_value());
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
