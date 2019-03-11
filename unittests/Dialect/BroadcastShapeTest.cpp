//===- BroadcastShapeTest.cpp - broadcasting shape unit tests -------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Dialect/Traits.h"
#include "gmock/gmock.h"

using namespace mlir::OpTrait::util;

using ::testing::ElementsAre;

TEST(BroadcastShapeTest, CompatibleScalarAndScalar) {
  auto result = getBroadcastedShape({}, {});
  ASSERT_TRUE(result.hasValue());
  EXPECT_TRUE(result->empty());
}

TEST(BroadcastShapeTest, Compatible0DAnd1DTensor) {
  auto result = getBroadcastedShape({}, {4});
  ASSERT_TRUE(result.hasValue());
  EXPECT_THAT(result.getValue(), ElementsAre(4));
}

TEST(BroadcastShapeTest, Compatible0DAnd3DTensor) {
  auto result = getBroadcastedShape({}, {3, 5, 4});
  ASSERT_TRUE(result.hasValue());
  EXPECT_THAT(result.getValue(), ElementsAre(3, 5, 4));
}

TEST(BroadcastShapeTest, CompatibleTensorAndTensor) {
  auto result = getBroadcastedShape({1, 7, 8, 9}, {8, 9});
  ASSERT_TRUE(result.hasValue());
  EXPECT_THAT(result.getValue(), ElementsAre(1, 7, 8, 9));
}

TEST(BroadcastShapeTest, InterleavingOnes) {
  auto result = getBroadcastedShape({8, 1, 2, 1, 4}, {5, 1, 7, 1});
  ASSERT_TRUE(result.hasValue());
  EXPECT_THAT(result.getValue(), ElementsAre(8, 5, 2, 7, 4));
}

TEST(BroadcastShapeTest, InterleavingUnknowns) {
  auto result = getBroadcastedShape({1, 2, -1, -1, -1}, {-1, -1, -1, 4, 1});
  EXPECT_TRUE(result.hasValue());
  EXPECT_THAT(result.getValue(), ElementsAre(-1, 2, -1, 4, -1));
}

TEST(BroadcastShapeTest, IncompatibleLowDim) {
  auto result = getBroadcastedShape({4, 3, 5, 5}, {3, 5, 4});
  EXPECT_FALSE(result.hasValue());
}

TEST(BroadcastShapeTest, IncompatibleMiddleDim) {
  auto result = getBroadcastedShape({4, 3, 5, 5}, {3, 7, 5});
  EXPECT_FALSE(result.hasValue());
}

TEST(BroadcastShapeTest, IncompatibleHighDim) {
  auto result = getBroadcastedShape({3, 5, 5}, {4, 5, 5});
  EXPECT_FALSE(result.hasValue());
}
