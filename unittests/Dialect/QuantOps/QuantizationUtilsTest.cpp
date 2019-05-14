//===- QuantizationUtilsTest.cpp - unit tests for quantization utils ------===//
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

#include "mlir/Dialect/QuantOps/QuantizeUtils.h"
#include "mlir/Dialect/QuantOps/UniformSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::quant;

namespace {

// Test UniformQuantizedValueConverter converts all APFloat to a magic number 5.
class TestUniformQuantizedValueConverter
    : public UniformQuantizedValueConverter {
public:
  TestUniformQuantizedValueConverter(UniformQuantizedType type)
      : UniformQuantizedValueConverter(type), qtype(type) {}
  APInt quantizeFloatToInt(APFloat expressedValue) const {
    return APInt(qtype.getStorageType().cast<IntegerType>().getWidth(), 5L);
  }

private:
  UniformQuantizedType qtype;
};

Attribute getTestFloatAttr(double value, MLIRContext *ctx) {
  return FloatAttr::get(FloatType::getF32(ctx), value);
}

template <typename ConcreteAttrClass, typename... Arg>
ConcreteAttrClass getTestElementsAttr(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                      Arg... value) {
  auto eleType = FloatType::getF32(ctx);
  VectorOrTensorType tensorType;
  if (shape.size() == 1 && shape[0] == -1) {
    tensorType = UnrankedTensorType::get(eleType);
  } else {
    tensorType = RankedTensorType::get(shape, eleType);
  }
  return ConcreteAttrClass::get(tensorType, value...);
}

ElementsAttr getTestSparseElementsAttr(MLIRContext *ctx,
                                       ArrayRef<int64_t> shape) {
  auto eleType = FloatType::getF32(ctx);
  VectorOrTensorType tensorType;
  if (shape.size() == 1 && shape[0] == -1) {
    tensorType = UnrankedTensorType::get(eleType);
  } else {
    tensorType = RankedTensorType::get(shape, eleType);
  }
  auto indicesType = RankedTensorType::get({1, 2}, IntegerType::get(64, ctx));
  auto indices =
      DenseIntElementsAttr::get(indicesType, {APInt(64, 0), APInt(64, 0)});
  auto valuesType = RankedTensorType::get({1}, eleType);
  auto values = DenseFPElementsAttr::get(valuesType, {APFloat(0.0f)});
  return SparseElementsAttr::get(tensorType, indices, values);
}

UniformQuantizedType getTestQuantizedType(Type storageType, MLIRContext *ctx) {
  return UniformQuantizedType::get(/*flags=*/false, storageType,
                                   FloatType::getF32(ctx), /*scale=*/1.0,
                                   /*zeroPoint=*/0, /*storageTypeMin=*/0,
                                   /*storageTypeMax=*/255);
}

TEST(QuantizationUtilsTest, convertFloatAttrUniform) {
  MLIRContext ctx;
  IntegerType convertedType = IntegerType::get(8, &ctx);
  auto quantizedType = getTestQuantizedType(convertedType, &ctx);
  TestUniformQuantizedValueConverter converter(quantizedType);

  auto realValue = getTestFloatAttr(1.0, &ctx);
  Type typeResult;
  auto valueResult =
      quantizeAttrUniform(realValue, quantizedType, converter, typeResult);

  EXPECT_EQ(valueResult.cast<IntegerAttr>().getInt(), 5);
  EXPECT_EQ(
      valueResult.cast<IntegerAttr>().getType().cast<IntegerType>().getWidth(),
      convertedType.getWidth());
}

TEST(QuantizationUtilsTest, convertRankedDenseAttrUniform) {
  MLIRContext ctx;
  IntegerType convertedType = IntegerType::get(8, &ctx);
  auto quantizedType = getTestQuantizedType(convertedType, &ctx);
  TestUniformQuantizedValueConverter converter(quantizedType);
  auto realValue = getTestElementsAttr<DenseElementsAttr, ArrayRef<Attribute>>(
      &ctx, {1, 2}, {getTestFloatAttr(1.0, &ctx), getTestFloatAttr(2.0, &ctx)});

  Type returnedType;
  auto returnedValue =
      quantizeAttrUniform(realValue, quantizedType, converter, returnedType);

  // Check Elements attribute shape and kind are not changed.
  auto tensorType = returnedType.cast<TensorType>();
  auto expectedTensorType = realValue.getType().cast<TensorType>();
  EXPECT_EQ(tensorType.getShape(), expectedTensorType.getShape());
  EXPECT_EQ(tensorType.getElementType(), convertedType);
  EXPECT_EQ(returnedValue.getKind(), StandardAttributes::DenseIntElements);

  // Check Elements attribute element value is expected.
  auto firstValue = returnedValue.cast<ElementsAttr>().getValue({0, 0});
  EXPECT_EQ(firstValue.cast<IntegerAttr>().getInt(), 5);
}

TEST(QuantizationUtilsTest, convertRankedSplatAttrUniform) {
  MLIRContext ctx;
  IntegerType convertedType = IntegerType::get(8, &ctx);
  auto quantizedType = getTestQuantizedType(convertedType, &ctx);
  TestUniformQuantizedValueConverter converter(quantizedType);
  auto realValue = getTestElementsAttr<SplatElementsAttr, Attribute>(
      &ctx, {1, 2}, getTestFloatAttr(1.0, &ctx));

  Type returnedType;
  auto returnedValue =
      quantizeAttrUniform(realValue, quantizedType, converter, returnedType);

  // Check Elements attribute shape and kind are not changed.
  auto tensorType = returnedType.cast<TensorType>();
  auto expectedTensorType = realValue.getType().cast<TensorType>();
  EXPECT_EQ(tensorType.getShape(), expectedTensorType.getShape());
  EXPECT_EQ(tensorType.getElementType(), convertedType);
  EXPECT_EQ(returnedValue.getKind(), StandardAttributes::SplatElements);

  // Check Elements attribute element value is expected.
  auto firstValue = returnedValue.cast<ElementsAttr>().getValue({0, 0});
  EXPECT_EQ(firstValue.cast<IntegerAttr>().getInt(), 5);
}

TEST(QuantizationUtilsTest, convertRankedSparseAttrUniform) {
  MLIRContext ctx;
  IntegerType convertedType = IntegerType::get(8, &ctx);
  auto quantizedType = getTestQuantizedType(convertedType, &ctx);
  TestUniformQuantizedValueConverter converter(quantizedType);
  auto realValue = getTestSparseElementsAttr(&ctx, {1, 2});

  Type returnedType;
  auto returnedValue =
      quantizeAttrUniform(realValue, quantizedType, converter, returnedType);

  // Check Elements attribute shape and kind are not changed.
  auto tensorType = returnedType.cast<TensorType>();
  auto expectedTensorType = realValue.getType().cast<TensorType>();
  EXPECT_EQ(tensorType.getShape(), expectedTensorType.getShape());
  EXPECT_EQ(tensorType.getElementType(), convertedType);
  EXPECT_EQ(returnedValue.getKind(), StandardAttributes::SparseElements);

  // Check Elements attribute element value is expected.
  auto firstValue = returnedValue.cast<ElementsAttr>().getValue({0, 0});
  EXPECT_EQ(firstValue.cast<IntegerAttr>().getInt(), 5);
}

} // end namespace
