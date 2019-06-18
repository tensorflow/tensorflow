//===- AttributeTest.cpp - Attribute unit tests ---------------------------===//
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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

template <typename EltTy>
static void testSplat(Type eltType, const EltTy &splatElt) {
  VectorType shape = VectorType::get({2, 1}, eltType);

  // Check that the generated splat is the same for 1 element and N elements.
  DenseElementsAttr splat = DenseElementsAttr::get(shape, splatElt);
  EXPECT_TRUE(splat.isSplat());

  auto detectedSplat =
      DenseElementsAttr::get(shape, llvm::makeArrayRef({splatElt, splatElt}));
  EXPECT_EQ(detectedSplat, splat);
}

namespace {
TEST(DenseSplatTest, BoolSplat) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  VectorType shape = VectorType::get({2, 2}, boolTy);

  // Check that splat is automatically detected for boolean values.
  /// True.
  DenseElementsAttr trueSplat = DenseElementsAttr::get(shape, true);
  EXPECT_TRUE(trueSplat.isSplat());
  /// False.
  DenseElementsAttr falseSplat = DenseElementsAttr::get(shape, false);
  EXPECT_TRUE(falseSplat.isSplat());
  EXPECT_NE(falseSplat, trueSplat);

  /// Detect and handle splat within 8 elements (bool values are bit-packed).
  /// True.
  auto detectedSplat = DenseElementsAttr::get(shape, {true, true, true, true});
  EXPECT_EQ(detectedSplat, trueSplat);
  /// False.
  detectedSplat = DenseElementsAttr::get(shape, {false, false, false, false});
  EXPECT_EQ(detectedSplat, falseSplat);
}

TEST(DenseSplatTest, LargeBoolSplat) {
  constexpr int64_t boolCount = 56;

  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  VectorType shape = VectorType::get({boolCount}, boolTy);

  // Check that splat is automatically detected for boolean values.
  /// True.
  DenseElementsAttr trueSplat = DenseElementsAttr::get(shape, true);
  DenseElementsAttr falseSplat = DenseElementsAttr::get(shape, false);
  EXPECT_TRUE(trueSplat.isSplat());
  EXPECT_TRUE(falseSplat.isSplat());

  /// Detect that the large boolean arrays are properly splatted.
  /// True.
  SmallVector<bool, 64> trueValues(boolCount, true);
  auto detectedSplat = DenseElementsAttr::get(shape, trueValues);
  EXPECT_EQ(detectedSplat, trueSplat);
  /// False.
  SmallVector<bool, 64> falseValues(boolCount, false);
  detectedSplat = DenseElementsAttr::get(shape, falseValues);
  EXPECT_EQ(detectedSplat, falseSplat);
}

TEST(DenseSplatTest, BoolNonSplat) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  VectorType shape = VectorType::get({6}, boolTy);

  // Check that we properly handle non-splat values.
  DenseElementsAttr nonSplat =
      DenseElementsAttr::get(shape, {false, false, true, false, false, true});
  EXPECT_FALSE(nonSplat.isSplat());
}

TEST(DenseSplatTest, OddIntSplat) {
  // Test detecting a splat with an odd(non 8-bit) integer bitwidth.
  MLIRContext context;
  constexpr size_t intWidth = 19;
  IntegerType intTy = IntegerType::get(intWidth, &context);
  APInt value(intWidth, 10);

  testSplat(intTy, value);
}

TEST(DenseSplatTest, Int32Splat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(32, &context);
  int value = 64;

  testSplat(intTy, value);
}

TEST(DenseSplatTest, IntAttrSplat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(85, &context);
  Attribute value = IntegerAttr::get(intTy, 109);

  testSplat(intTy, value);
}

TEST(DenseSplatTest, F32Splat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getF32(&context);
  float value = 10.0;

  testSplat(floatTy, value);
}

TEST(DenseSplatTest, F64Splat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getF64(&context);
  double value = 10.0;

  testSplat(floatTy, APFloat(value));
}

TEST(DenseSplatTest, FloatAttrSplat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getBF16(&context);
  Attribute value = FloatAttr::get(floatTy, 10.0);

  testSplat(floatTy, value);
}
} // end namespace
