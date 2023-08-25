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

#include "tensorflow/compiler/xla/python/ifrt/shape.h"

#include <limits>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace ifrt {
namespace {

TEST(ShapeTest, LargeDim) {
  Shape shape({std::numeric_limits<int64_t>::max()});
  EXPECT_THAT(shape.dims(),
              testing::ElementsAre(std::numeric_limits<int64_t>::max()));
}

TEST(ShapeTest, ManyDims) {
  const int kNumDims = 65536;  // Arbitrarily large number.
  std::vector<int64_t> dims(kNumDims);
  std::iota(dims.begin(), dims.end(), 0);
  Shape shape(dims);
  EXPECT_THAT(shape.dims(), testing::ElementsAreArray(dims));
}

TEST(ShapeTest, ScalarNumElements) {
  Shape shape({});
  EXPECT_EQ(shape.num_elements(), 1);
}

TEST(ShapeTest, ZeroDimNumElements) {
  {
    Shape shape({0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({1, 0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({0, 1});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({0, 0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
}

TEST(ShapeTest, NonZeroDimsNumElements) {
  {
    Shape shape({2});
    EXPECT_EQ(shape.num_elements(), 2);
  }
  {
    Shape shape({2, 3});
    EXPECT_EQ(shape.num_elements(), 6);
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
