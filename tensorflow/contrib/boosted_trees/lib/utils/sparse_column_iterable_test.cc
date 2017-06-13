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

#include "tensorflow/contrib/boosted_trees/lib/utils/sparse_column_iterable.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

using test::AsTensor;
using ExampleRowRange = SparseColumnIterable::ExampleRowRange;

class SparseColumnIterableTest : public ::testing::Test {};

TEST_F(SparseColumnIterableTest, Empty) {
  const auto indices = Tensor(DT_INT64, {0, 2});
  SparseColumnIterable iterable(indices.template matrix<int64>(), 0, 0);
  EXPECT_EQ(iterable.begin(), iterable.end());
}

TEST_F(SparseColumnIterableTest, Iterate) {
  // 8 examples having 7 sparse features with the third multi-valent.
  // This can be visualized like the following:
  // Instance | Sparse |
  // 0        |   x    |
  // 1        |        |
  // 2        |        |
  // 3        |  xxx   |
  // 4        |   x    |
  // 5        |        |
  // 6        |        |
  // 7        |   xx   |
  const auto indices =
      AsTensor<int64>({0, 0, 3, 0, 3, 1, 3, 2, 4, 0, 7, 0, 7, 1}, {7, 2});

  auto validate_example_range = [](const ExampleRowRange& range) {
    switch (range.example_idx) {
      case 0: {
        EXPECT_EQ(0, range.start);
        EXPECT_EQ(1, range.end);
      } break;
      case 3: {
        EXPECT_EQ(1, range.start);
        EXPECT_EQ(4, range.end);
      } break;
      case 4: {
        EXPECT_EQ(4, range.start);
        EXPECT_EQ(5, range.end);
      } break;
      case 7: {
        EXPECT_EQ(5, range.start);
        EXPECT_EQ(7, range.end);
      } break;
      default: {
        // Empty examples.
        EXPECT_GE(range.start, range.end);
      } break;
    }
  };

  // Iterate through all examples sequentially.
  SparseColumnIterable full_iterable(indices.template matrix<int64>(), 0, 8);
  int64 expected_example_idx = 0;
  for (const ExampleRowRange& range : full_iterable) {
    EXPECT_EQ(expected_example_idx, range.example_idx);
    validate_example_range(range);
    ++expected_example_idx;
  }
  EXPECT_EQ(8, expected_example_idx);

  // Iterate through slice (2, 6) of examples.
  SparseColumnIterable slice_iterable(indices.template matrix<int64>(), 2, 6);
  expected_example_idx = 2;
  for (const ExampleRowRange& range : slice_iterable) {
    EXPECT_EQ(expected_example_idx, range.example_idx);
    validate_example_range(range);
    ++expected_example_idx;
  }
  EXPECT_EQ(6, expected_example_idx);
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
