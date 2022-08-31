/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/shape.h"

#include <gtest/gtest.h>

namespace tflite {
namespace shim {
namespace {

TEST(Shape, Eq) {
  EXPECT_TRUE(Shape({1, 2}) == Shape({1, 2}));
  // scalar
  EXPECT_TRUE(Shape(std::vector<int>{}) == Shape(std::vector<int>{}));
  // vector of size 1
  EXPECT_TRUE(Shape({1}) == Shape({1}));
  // dim mismatch
  EXPECT_FALSE(Shape({1, 2, 1}) == Shape({1, 2}));
  // dim mismatch
  EXPECT_FALSE(Shape({1, 3}) == Shape({1, 2}));
  // unknown dim
  EXPECT_FALSE(Shape({3, -1, 2}) == Shape({3, -1, 2}));
  // two unknown ranks
  EXPECT_FALSE(Shape() == Shape());
  // Unknown rank vs known shape
  EXPECT_FALSE(Shape({3, 4}) == Shape());
  // scalar vs. vector of size 1
  EXPECT_FALSE(Shape(std::vector<int>{}) == Shape({1}));
  // unknown dim
  EXPECT_FALSE(Shape({1, -1}) == Shape({1, 2}));
  // same num elems
  EXPECT_FALSE(Shape({1, 2}) == Shape({2, 1}));
}

TEST(Shape, Compatible) {
  EXPECT_TRUE(Shape({1, 2}).Compatible(Shape({1, 2})));
  // scalar
  EXPECT_TRUE(Shape(std::vector<int>{}).Compatible(Shape(std::vector<int>{})));
  // vector of size 1
  EXPECT_TRUE(Shape({1}).Compatible(Shape({1})));
  // unknown dim
  EXPECT_TRUE(Shape({3, -1, 2}).Compatible(Shape({3, -1, 2})));
  // dim mismatch
  EXPECT_FALSE(Shape({1, 2, 1}).Compatible(Shape({1, 2})));
  // dim mismatch
  EXPECT_FALSE(Shape({1, 3}).Compatible(Shape({1, 2})));
  // two unknown ranks
  EXPECT_TRUE(Shape().Compatible(Shape()));
  // Unknown rank vs known shape
  EXPECT_TRUE(Shape({3, 4}).Compatible(Shape()));
  // scalar vs. vector of size 1
  EXPECT_FALSE(Shape(std::vector<int>{}).Compatible(Shape({1})));
  // unknown dim
  EXPECT_TRUE(Shape({1, -1}).Compatible(Shape({1, 2})));
  // same num elems
  EXPECT_FALSE(Shape({1, 2}).Compatible(Shape({2, 1})));
}

TEST(Shape, ToStr) {
  EXPECT_EQ("[ 1 2 ]", Shape({1, 2}).ToString());
  EXPECT_EQ("[]", Shape({}).ToString());
  EXPECT_EQ("?", Shape().ToString());
  EXPECT_EQ("[ 1 2 3 ]", Shape({1, 2, 3}).ToString());
  EXPECT_EQ("[ 2 ? 3 ]", Shape({2, -1, 3}).ToString());
}

TEST(Shape, FullyDefined) {
  EXPECT_TRUE(Shape({}).FullyDefined());
  EXPECT_FALSE(Shape().FullyDefined());
  EXPECT_TRUE(Shape({1}).FullyDefined());
  EXPECT_TRUE(Shape({1, 2}).FullyDefined());
  EXPECT_FALSE(Shape({3, -1, 2}).FullyDefined());
  EXPECT_FALSE(Shape({-1, -1}).FullyDefined());
}

TEST(Shape, Dim) {
  EXPECT_EQ(-1, Shape().Dim(2));
  EXPECT_EQ(3, Shape({2, 3}).Dim(1));
  EXPECT_EQ(3, Shape({-1, 3}).Dim(1));
  EXPECT_EQ(-1, Shape({-1, 3}).Dim(0));
}

TEST(Shape, AddDims) {
  EXPECT_EQ(5, Shape::AddDims(3, 2));
  EXPECT_EQ(-1, Shape::AddDims(-1, 2));
  EXPECT_EQ(-1, Shape::AddDims(1, -2));
  EXPECT_EQ(-1, Shape::AddDims(-1, -1));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
