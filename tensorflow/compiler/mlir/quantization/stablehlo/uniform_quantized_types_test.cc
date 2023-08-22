/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/uniform_quantized_types.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {
namespace {

using ::testing::ElementsAreArray;

class CreateI8F32UniformQuantizedTypeTest : public ::testing::Test {
 protected:
  CreateI8F32UniformQuantizedTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI8F32UniformQuantizedTypeTest, HasI8StorageType) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(8));
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, HasF32ExpressedType) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, IsSigned) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.isSigned());
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, SotrageTypeMinMaxEqualToI8MinMax) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -128);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/8.0, /*zero_point=*/99);

  EXPECT_EQ(quantized_type.getScale(), 8.0);
  EXPECT_EQ(quantized_type.getZeroPoint(), 99);
}

class CreateI8F32UniformQuantizedPerAxisTypeTest : public ::testing::Test {
 protected:
  CreateI8F32UniformQuantizedPerAxisTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest, HasI8StorageType) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int8_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(8));
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest, HasF32ExpressedType) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int8_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest, IsSigned) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int8_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.isSigned());
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       StorageTypeMinMaxEqualToI8MinMax) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int8_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -128);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       HasQuantizationDimensionProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int8_t, 2>{0, 0},
          /*quantization_dimension=*/3);

  EXPECT_EQ(quantized_type.getQuantizedDimension(), 3);
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<float, 2>{8.0, 9.0},
          /*zero_points=*/SmallVector<int8_t, 2>{98, 99},
          /*quantization_dimension=*/0);

  EXPECT_THAT(quantized_type.getScales(), ElementsAreArray({8.0, 9.0}));
  EXPECT_THAT(quantized_type.getZeroPoints(), ElementsAreArray({98, 99}));
}

}  // namespace
}  // namespace quant
}  // namespace mlir
