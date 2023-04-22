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

#include "tensorflow/compiler/xla/client/padding.h"

#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

// Tests MakePadding utility function for various cases.
class PaddingTest : public ::testing::Test {
 protected:
  // A convenience function to test padding for a single dimension.
  std::pair<int64, int64> ComputePadding(int64 input_dimension,
                                         int64 window_dimension,
                                         int64 window_stride, Padding padding) {
    return MakePadding({input_dimension}, {window_dimension}, {window_stride},
                       padding)[0];
  }
};

TEST_F(PaddingTest, ValidPaddingWithStrideOne) {
  const auto padding = ComputePadding(10, 5, 1, Padding::kValid);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);
}

TEST_F(PaddingTest, ValidPaddingWithStrideThree) {
  const auto padding = ComputePadding(10, 5, 3, Padding::kValid);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);
}

TEST_F(PaddingTest, SamePaddingWithOddWindow) {
  const auto padding = ComputePadding(10, 7, 1, Padding::kSame);
  EXPECT_EQ(padding.first, 3);
  EXPECT_EQ(padding.second, 3);
}

TEST_F(PaddingTest, SamePaddingWithEvenWindow) {
  const auto padding = ComputePadding(10, 6, 1, Padding::kSame);
  EXPECT_EQ(padding.first, 2);
  EXPECT_EQ(padding.second, 3);
}

TEST_F(PaddingTest, SamePaddingWithOddWindowWithStride) {
  const auto padding = ComputePadding(10, 7, 3, Padding::kSame);
  EXPECT_EQ(padding.first, 3);
  EXPECT_EQ(padding.second, 3);
}

TEST_F(PaddingTest, SamePaddingWithEvenWindowWithStride) {
  const auto padding = ComputePadding(10, 6, 4, Padding::kSame);
  EXPECT_EQ(padding.first, 2);
  EXPECT_EQ(padding.second, 2);
}

TEST_F(PaddingTest, SamePaddingForWindowSizeOne) {
  const auto padding = ComputePadding(10, 1, 1, Padding::kSame);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);
}

TEST_F(PaddingTest, SamePaddingForWindowLargerThanInput) {
  const auto padding = ComputePadding(10, 20, 1, Padding::kSame);
  EXPECT_EQ(padding.first, 9);
  EXPECT_EQ(padding.second, 10);
}

// This used to trigger a case with negative padding.
TEST_F(PaddingTest, NonNegativePadding) {
  const auto padding = ComputePadding(4, 1, 2, Padding::kSame);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);
}

}  // namespace
}  // namespace xla
