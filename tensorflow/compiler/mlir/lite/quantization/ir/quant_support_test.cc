/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/UniformSupport.h"

namespace mlir {
namespace TFL {
namespace {

using ::llvm::APFloat;
using ::mlir::quant::UniformQuantizedPerAxisType;
using ::mlir::quant::UniformQuantizedType;
using ::testing::ElementsAre;

class QuantSupportTest : public ::testing::Test {
 protected:
  QuantSupportTest() : builder_(&ctx_), loc_(UnknownLoc::get(&ctx_)) {
    ctx_.loadDialect<quant::QuantDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_;
  Location loc_;
};

TEST_F(QuantSupportTest, FakeQuantUnsigned8Bit) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*rmin=*/0.0, /*rmax=*/255.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/false);
  ASSERT_TRUE(type);
  EXPECT_FALSE(type.isSigned());
  EXPECT_TRUE(type.getStorageType().isInteger(8));
  EXPECT_TRUE(type.getExpressedType().isF32());
  EXPECT_DOUBLE_EQ(type.getScale(), 1.0);
  EXPECT_EQ(type.getZeroPoint(), 0);
  EXPECT_EQ(type.getStorageTypeMin(), 0);
  EXPECT_EQ(type.getStorageTypeMax(), 255);
}

TEST_F(QuantSupportTest, FakeQuantSignedNarrowRange) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*rmin=*/-127.0, /*rmax=*/127.0,
      /*narrowRange=*/true, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.isSigned());
  EXPECT_DOUBLE_EQ(type.getScale(), 1.0);
  EXPECT_EQ(type.getZeroPoint(), 0);
  EXPECT_EQ(type.getStorageTypeMin(), -127);
  EXPECT_EQ(type.getStorageTypeMax(), 127);
}

TEST_F(QuantSupportTest, FakeQuant4BitUsesI4Storage) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/4, /*rmin=*/-8.0, /*rmax=*/7.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getStorageType().isInteger(4));
  EXPECT_EQ(type.getStorageTypeMin(), -8);
  EXPECT_EQ(type.getStorageTypeMax(), 7);
  EXPECT_EQ(type.getZeroPoint(), 0);
}

TEST_F(QuantSupportTest, FakeQuant16Bit) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/16, /*rmin=*/-32768.0, /*rmax=*/32767.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getStorageType().isInteger(16));
  EXPECT_EQ(type.getStorageTypeMin(), -32768);
  EXPECT_EQ(type.getStorageTypeMax(), 32767);
  EXPECT_DOUBLE_EQ(type.getScale(), 1.0);
  EXPECT_EQ(type.getZeroPoint(), 0);
}

TEST_F(QuantSupportTest, FakeQuantNudgesZeroPointIntoRange) {
  // 0 < rmin < rmax: the zero point is clamped to qmin and the scale is kept.
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*rmin=*/1.0, /*rmax=*/256.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/false);
  ASSERT_TRUE(type);
  EXPECT_DOUBLE_EQ(type.getScale(), 1.0);
  EXPECT_EQ(type.getZeroPoint(), 0);
}

TEST_F(QuantSupportTest, FakeQuantEmptyRangeUsesUnitScale) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*rmin=*/0.0, /*rmax=*/0.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  EXPECT_DOUBLE_EQ(type.getScale(), 1.0);
  EXPECT_EQ(type.getZeroPoint(), -128);
}

TEST_F(QuantSupportTest, FakeQuantUnsupportedBitWidthFails) {
  const ScopedDiagnosticHandler handler(&ctx_,
                                        [](Diagnostic&) { return success(); });
  EXPECT_FALSE(fakeQuantAttrsToType(loc_, /*numBits=*/33, /*rmin=*/-1.0,
                                    /*rmax=*/1.0, /*narrowRange=*/false,
                                    builder_.getF32Type()));
}

TEST_F(QuantSupportTest, FakeQuantPerAxis) {
  const UniformQuantizedPerAxisType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*quantizedDimension=*/1,
      /*rmins=*/{-128.0, 0.0}, /*rmax=*/{127.0, 0.0},
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  EXPECT_EQ(type.getQuantizedDimension(), 1);
  EXPECT_THAT(type.getScales(), ElementsAre(1.0, 1.0));
  EXPECT_THAT(type.getZeroPoints(), ElementsAre(0, -128));
}

TEST_F(QuantSupportTest, FakeQuantPerAxisMismatchedSizesFails) {
  const ScopedDiagnosticHandler handler(&ctx_,
                                        [](Diagnostic&) { return success(); });
  EXPECT_FALSE(fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*quantizedDimension=*/0, /*rmins=*/{-1.0, -1.0},
      /*rmax=*/{1.0}, /*narrowRange=*/false, builder_.getF32Type()));
}

TEST_F(QuantSupportTest, ExpressedToQuantizedConverterForTensor) {
  const auto tensor_type = RankedTensorType::get({2}, builder_.getF32Type());
  const auto converter =
      ExpressedToQuantizedConverter::forInputType(tensor_type);
  ASSERT_TRUE(converter);
  EXPECT_EQ(converter.expressed_type, builder_.getF32Type());

  const UniformQuantizedType quantized_type =
      fakeQuantAttrsToType(loc_, /*numBits=*/8, /*rmin=*/0.0, /*rmax=*/255.0,
                           /*narrowRange=*/false, builder_.getF32Type());
  const Type converted = converter.convert(quantized_type);
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted, RankedTensorType::get({2}, quantized_type));
}

TEST_F(QuantSupportTest, ExpressedToQuantizedConverterRejectsIntegers) {
  EXPECT_FALSE(
      ExpressedToQuantizedConverter::forInputType(builder_.getI32Type()));
}

TEST_F(QuantSupportTest, UniformConverterI8FastPath) {
  const UniformQuantizedValueConverter converter(
      /*scale=*/0.5, /*zero_point=*/1.0, /*clamp_min=*/-128.0,
      /*clamp_max=*/127.0, /*storage_bit_width=*/8, /*is_signed=*/true);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(1.0f)), 3);
  // -1.25 / 0.5 + 1 = -1.5, which rounds away from zero.
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(-1.25f)), -2);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(1000.0f)), 127);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(-1000.0f)), -128);
}

TEST_F(QuantSupportTest, UniformConverterGenericPath) {
  // A double input or a non-8-bit storage type takes the APFloat path.
  const UniformQuantizedValueConverter converter_i8(
      /*scale=*/0.5, /*zero_point=*/1.0, /*clamp_min=*/-128.0,
      /*clamp_max=*/127.0, /*storage_bit_width=*/8, /*is_signed=*/true);
  EXPECT_EQ(converter_i8.quantizeFloatToInt64(APFloat(1.0)), 3);
  EXPECT_EQ(converter_i8.quantizeFloatToInt64(APFloat(-1000.0)), -128);

  const UniformQuantizedValueConverter converter_i16(
      /*scale=*/0.5, /*zero_point=*/1.0, /*clamp_min=*/-32768.0,
      /*clamp_max=*/32767.0, /*storage_bit_width=*/16, /*is_signed=*/true);
  EXPECT_EQ(converter_i16.quantizeFloatToInt64(APFloat(1.0f)), 3);
  EXPECT_EQ(converter_i16.quantizeFloatToInt64(APFloat(100000.0f)), 32767);
}

TEST_F(QuantSupportTest, UniformConverterUnsigned) {
  const UniformQuantizedType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*rmin=*/-128.0, /*rmax=*/127.0,
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/false);
  ASSERT_TRUE(type);
  ASSERT_EQ(type.getZeroPoint(), 128);
  const UniformQuantizedValueConverter converter(type);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(0.0f)), 128);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(-1000.0f)), 0);
  EXPECT_EQ(converter.quantizeFloatToInt64(APFloat(1000.0f)), 255);
}

TEST_F(QuantSupportTest, PerAxisConverter) {
  const UniformQuantizedPerAxisType type = fakeQuantAttrsToType(
      loc_, /*numBits=*/8, /*quantizedDimension=*/0,
      /*rmins=*/{-128.0, -64.0}, /*rmax=*/{127.0, 63.5},
      /*narrowRange=*/false, builder_.getF32Type(), /*isSigned=*/true);
  ASSERT_TRUE(type);
  ASSERT_THAT(type.getScales(), ElementsAre(1.0, 0.5));
  ASSERT_THAT(type.getZeroPoints(), ElementsAre(0, 0));

  const auto attr_type = RankedTensorType::get({2, 1}, builder_.getF32Type());
  const auto real_value =
      DenseFPElementsAttr::get(attr_type, ArrayRef<float>{2.0f, 3.0f});
  UniformQuantizedPerAxisValueConverter converter(type);
  const auto quantized = mlir::dyn_cast_if_present<DenseIntElementsAttr>(
      converter.convert(Attribute(real_value)));
  ASSERT_TRUE(quantized);
  const llvm::SmallVector<int8_t> values(quantized.getValues<int8_t>());
  EXPECT_THAT(values, ElementsAre(2, 6));
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
