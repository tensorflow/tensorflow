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
#include "llvm/ADT/SmallVector.h"
#include "gmock/gmock.h"

using namespace mlir::OpTrait::util;

using llvm::SmallVector;
using ::testing::ElementsAre;

TEST(BroadcastShapeTest, CompatibleScalarAndScalar) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(getBroadcastedShape({}, {}, result));
  EXPECT_TRUE(result.empty());
}

TEST(BroadcastShapeTest, Compatible0DAnd1DTensor) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(getBroadcastedShape({}, {4}, result));
  EXPECT_THAT(result, ElementsAre(4));
}

TEST(BroadcastShapeTest, Compatible0DAnd3DTensor) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(getBroadcastedShape({}, {3, 5, 4}, result));
  EXPECT_THAT(result, ElementsAre(3, 5, 4));
}

TEST(BroadcastShapeTest, CompatibleTensorAndTensor) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(getBroadcastedShape({1, 7, 8, 9}, {8, 9}, result));
  EXPECT_THAT(result, ElementsAre(1, 7, 8, 9));
}

TEST(BroadcastShapeTest, InterleavingOnes) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(getBroadcastedShape({8, 1, 2, 1, 4}, {5, 1, 7, 1}, result));
  EXPECT_THAT(result, ElementsAre(8, 5, 2, 7, 4));
}

TEST(BroadcastShapeTest, InterleavingUnknowns) {
  SmallVector<int64_t, 4> result;
  ASSERT_TRUE(
      getBroadcastedShape({1, 2, -1, -1, -1}, {-1, -1, -1, 4, 1}, result));
  EXPECT_THAT(result, ElementsAre(-1, 2, -1, 4, -1));
}

TEST(BroadcastShapeTest, IncompatibleLowDim) {
  SmallVector<int64_t, 4> result;
  ASSERT_FALSE(getBroadcastedShape({4, 3, 5, 5}, {3, 5, 4}, result));
  EXPECT_TRUE(result.empty());
}

TEST(BroadcastShapeTest, IncompatibleMiddleDim) {
  SmallVector<int64_t, 4> result;
  ASSERT_FALSE(getBroadcastedShape({4, 3, 5, 5}, {3, 7, 5}, result));
  EXPECT_TRUE(result.empty());
}

TEST(BroadcastShapeTest, IncompatibleHighDim) {
  SmallVector<int64_t, 4> result;
  ASSERT_FALSE(getBroadcastedShape({3, 5, 5}, {4, 5, 5}, result));
  EXPECT_TRUE(result.empty());
}
