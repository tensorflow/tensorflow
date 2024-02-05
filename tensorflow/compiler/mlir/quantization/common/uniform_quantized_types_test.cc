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
#include "tensorflow/compiler/mlir/quantization/common/uniform_quantized_types.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {
namespace {

using ::testing::ElementsAreArray;
using ::testing::NotNull;
using ::testing::Test;

class CreateI8F32UniformQuantizedTypeTest : public Test {
 protected:
  CreateI8F32UniformQuantizedTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI8F32UniformQuantizedTypeTest, I8StorageTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);
  // Storage type of `i8` is currently verifiable as `unsigned` in `Types.cpp`.
  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(8));
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, F32ExpressedTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, SignedQuantizedTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.isSigned());
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, StorageTypeMinMaxEqualToI8MinMax) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -128);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, StorageTypeMinMaxNarrowRange) {
  const UniformQuantizedType quantized_type = CreateI8F32UniformQuantizedType(
      UnknownLoc::get(&ctx_), ctx_,
      /*scale=*/1.0, /*zero_point=*/0, /*narrow_range=*/true);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -127);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedTypeTest, HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedType quantized_type =
      CreateI8F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                      /*scale=*/8.0, /*zero_point=*/99);

  EXPECT_EQ(quantized_type.getScale(), 8.0);
  EXPECT_EQ(quantized_type.getZeroPoint(), 99);
}

class CreateI32F32UniformQuantizedTypeTest : public Test {
 protected:
  CreateI32F32UniformQuantizedTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI32F32UniformQuantizedTypeTest, I32StorageTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI32F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                       /*scale=*/1.0, /*zero_point=*/0);

  // Storage type of `i32` is currently verifiable as `unsigned` in `Types.cpp`.
  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(32));
}

TEST_F(CreateI32F32UniformQuantizedTypeTest, F32ExpressedTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI32F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                       /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI32F32UniformQuantizedTypeTest, SignedQuantizedTypeSucceeds) {
  const UniformQuantizedType quantized_type =
      CreateI32F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                       /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_TRUE(quantized_type.isSigned());
}

TEST_F(CreateI32F32UniformQuantizedTypeTest,
       StorageTypeMinMaxEqualToI32MinMax) {
  const UniformQuantizedType quantized_type =
      CreateI32F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                       /*scale=*/1.0, /*zero_point=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(),
            std::numeric_limits<int32_t>::min());
  EXPECT_EQ(quantized_type.getStorageTypeMax(),
            std::numeric_limits<int32_t>::max());
}

TEST_F(CreateI32F32UniformQuantizedTypeTest, HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedType quantized_type =
      CreateI32F32UniformQuantizedType(UnknownLoc::get(&ctx_), ctx_,
                                       /*scale=*/8.0, /*zero_point=*/1111);

  EXPECT_EQ(quantized_type.getScale(), 8.0);
  EXPECT_EQ(quantized_type.getZeroPoint(), 1111);
}

class CreateI8F32UniformQuantizedPerAxisTypeTest : public Test {
 protected:
  CreateI8F32UniformQuantizedPerAxisTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest, I8StorageTypeSucceeds) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  // Storage type of `i8` is currently verifiable as `unsigned` in `Types.cpp`.
  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(8));
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest, F32ExpressedTypeSucceeds) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       SignedQuantizedTypeSucceeds) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.isSigned());
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       StorageTypeMinMaxEqualToI8MinMax) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -128);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       StorageTypeMinMaxNarrowRange) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0, /*narrow_range=*/true);

  EXPECT_EQ(quantized_type.getStorageTypeMin(), -127);
  EXPECT_EQ(quantized_type.getStorageTypeMax(), 127);
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       HasQuantizationDimensionProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/3);

  EXPECT_EQ(quantized_type.getQuantizedDimension(), 3);
}

TEST_F(CreateI8F32UniformQuantizedPerAxisTypeTest,
       HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI8F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{8.0, 9.0},
          /*zero_points=*/SmallVector<int64_t, 2>{98, 99},
          /*quantization_dimension=*/0);

  EXPECT_THAT(quantized_type.getScales(), ElementsAreArray({8.0, 9.0}));
  EXPECT_THAT(quantized_type.getZeroPoints(), ElementsAreArray({98, 99}));
}

class CreateI32F32UniformQuantizedPerAxisTypeTest : public Test {
 protected:
  CreateI32F32UniformQuantizedPerAxisTypeTest() : ctx_() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
};

TEST_F(CreateI32F32UniformQuantizedPerAxisTypeTest, I32StorageTypeSucceeds) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI32F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  // Storage type of `i32` is currently verifiable as `unsigned` in `Types.cpp`.
  EXPECT_TRUE(quantized_type.getStorageType().isSignlessInteger(32));
}

TEST_F(CreateI32F32UniformQuantizedPerAxisTypeTest, F32ExpressedTypeSucceeds) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI32F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_TRUE(quantized_type.getExpressedType().isF32());
}

TEST_F(CreateI32F32UniformQuantizedPerAxisTypeTest,
       StorageTypeMinMaxEqualToI32MinMax) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI32F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/0);

  EXPECT_EQ(quantized_type.getStorageTypeMin(),
            std::numeric_limits<int32_t>::min());
  EXPECT_EQ(quantized_type.getStorageTypeMax(),
            std::numeric_limits<int32_t>::max());
}

TEST_F(CreateI32F32UniformQuantizedPerAxisTypeTest,
       HasQuantizationDimensionProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI32F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{1.0, 1.0},
          /*zero_points=*/SmallVector<int64_t, 2>{0, 0},
          /*quantization_dimension=*/3);

  EXPECT_EQ(quantized_type.getQuantizedDimension(), 3);
}

TEST_F(CreateI32F32UniformQuantizedPerAxisTypeTest,
       HasScaleAndZeroPointProperlySet) {
  const UniformQuantizedPerAxisType quantized_type =
      CreateI32F32UniformQuantizedPerAxisType(
          UnknownLoc::get(&ctx_), ctx_,
          /*scales=*/SmallVector<double, 2>{8.0, 9.0},
          /*zero_points=*/SmallVector<int64_t, 2>{98, 99},
          /*quantization_dimension=*/0);

  EXPECT_THAT(quantized_type.getScales(), ElementsAreArray({8.0, 9.0}));
  EXPECT_THAT(quantized_type.getZeroPoints(), ElementsAreArray({98, 99}));
}

class IsI8F32UniformQuantizedTypeTest : public Test {
 protected:
  IsI8F32UniformQuantizedTypeTest() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_{&ctx_};
};

TEST_F(IsI8F32UniformQuantizedTypeTest, I8F32UniformQuantizedTypeSucceeds) {
  const UniformQuantizedType qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(), /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsI8F32UniformQuantizedType(qi8_type));
}

TEST_F(IsI8F32UniformQuantizedTypeTest, UniformQuantizedTypeSucceeds) {
  const UniformQuantizedType qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(), /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_THAT(qi8_type.dyn_cast_or_null<UniformQuantizedType>(), NotNull());
}

TEST_F(IsI8F32UniformQuantizedTypeTest, StorageTypeI8Succeeds) {
  const UniformQuantizedType qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(), /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsStorageTypeI8(qi8_type));
}

TEST_F(IsI8F32UniformQuantizedTypeTest, ExpressedTypeF32Succeeds) {
  const UniformQuantizedType qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(), /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsExpressedTypeF32(qi8_type));
}

class IsI8F32UniformQuantizedPerAxisTypeTest : public Test {
 protected:
  IsI8F32UniformQuantizedPerAxisTypeTest() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_{&ctx_};
};

TEST_F(IsI8F32UniformQuantizedPerAxisTypeTest,
       I8F32UniformQuantizedPerAxisTypeSucceeds) {
  const UniformQuantizedPerAxisType qi8_per_axis_type =
      quant::UniformQuantizedPerAxisType::get(
          /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
          builder_.getF32Type(),
          /*scales=*/{1.0},
          /*zeroPoints=*/{0}, /*quantizedDimension=*/0, /*storageTypeMin=*/-128,
          /*storageTypeMax=*/127);
  EXPECT_TRUE(IsI8F32UniformQuantizedPerAxisType(qi8_per_axis_type));
  EXPECT_FALSE(IsI8F32UniformQuantizedType(qi8_per_axis_type));
}

TEST_F(IsI8F32UniformQuantizedTypeTest, UniformQuantizedPerAxisTypeSucceeds) {
  const UniformQuantizedPerAxisType qi8_per_axis_type =
      quant::UniformQuantizedPerAxisType::get(
          /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
          builder_.getF32Type(),
          /*scales=*/{1.0},
          /*zeroPoints=*/{0}, /*quantizedDimension=*/0, /*storageTypeMin=*/-128,
          /*storageTypeMax=*/127);
  EXPECT_THAT(qi8_per_axis_type.dyn_cast_or_null<UniformQuantizedPerAxisType>(),
              NotNull());
}

TEST_F(IsI8F32UniformQuantizedPerAxisTypeTest, StorageTypeI8Succeeds) {
  const UniformQuantizedPerAxisType qi8_per_axis_type =
      quant::UniformQuantizedPerAxisType::get(
          /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
          builder_.getF32Type(),
          /*scales=*/{1.0},
          /*zeroPoints=*/{0}, /*quantizedDimension=*/0, /*storageTypeMin=*/-128,
          /*storageTypeMax=*/127);
  EXPECT_TRUE(IsStorageTypeI8(qi8_per_axis_type));
}

TEST_F(IsI8F32UniformQuantizedPerAxisTypeTest, ExpressedTypeF32Succeeds) {
  const UniformQuantizedPerAxisType qi8_per_axis_type =
      quant::UniformQuantizedPerAxisType::get(
          /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
          builder_.getF32Type(),
          /*scales=*/{1.0},
          /*zeroPoints=*/{0}, /*quantizedDimension=*/0, /*storageTypeMin=*/-128,
          /*storageTypeMax=*/127);
  EXPECT_TRUE(IsExpressedTypeF32(qi8_per_axis_type));
}

class IsI32F32UniformQuantizedTypeTest : public Test {
 protected:
  IsI32F32UniformQuantizedTypeTest() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_{&ctx_};
};

TEST_F(IsI32F32UniformQuantizedTypeTest, I32F32UniformQuantizedTypeSucceeds) {
  const UniformQuantizedType qi32_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI32Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-2147483647,
      /*storageTypeMax=*/2147483646);
  EXPECT_TRUE(IsI32F32UniformQuantizedType(qi32_type));
}

TEST_F(IsI32F32UniformQuantizedTypeTest, UniformQuantizedTypeSucceeds) {
  const UniformQuantizedType qi32_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI32Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-2147483647,
      /*storageTypeMax=*/2147483646);
  EXPECT_TRUE(IsI32F32UniformQuantizedType(qi32_type));
  EXPECT_THAT(qi32_type.dyn_cast_or_null<UniformQuantizedType>(), NotNull());
}

TEST_F(IsI32F32UniformQuantizedTypeTest, StorageTypeI32Succeeds) {
  const UniformQuantizedType qi32_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI32Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-2147483647,
      /*storageTypeMax=*/2147483646);
  EXPECT_TRUE(IsI32F32UniformQuantizedType(qi32_type));
  EXPECT_TRUE(IsStorageTypeI32(qi32_type));
}

TEST_F(IsI32F32UniformQuantizedTypeTest, ExpressedTypeF32Succeeds) {
  const UniformQuantizedType qi32_per_axis_type =
      quant::UniformQuantizedType::get(
          /*flags=*/QuantizationFlags::Signed, builder_.getI32Type(),
          builder_.getF32Type(),
          /*scale=*/1.0,
          /*zeroPoint=*/0, /*storageTypeMin=*/-2147483647,
          /*storageTypeMax=*/2147483646);
  EXPECT_TRUE(IsExpressedTypeF32(qi32_per_axis_type));
}

class IsSupportedByTfliteQuantizeOrDequantizeOpsTest : public Test {
 protected:
  IsSupportedByTfliteQuantizeOrDequantizeOpsTest() {
    ctx_.loadDialect<quant::QuantizationDialect>();
  }

  MLIRContext ctx_;
  OpBuilder builder_{&ctx_};
};

TEST_F(IsSupportedByTfliteQuantizeOrDequantizeOpsTest, StorageTypeI8Succeeds) {
  auto qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsSupportedByTfliteQuantizeOrDequantizeOps(
      dyn_cast_or_null<IntegerType>(qi8_type.getStorageType())));
}

TEST_F(IsSupportedByTfliteQuantizeOrDequantizeOpsTest, StorageTypeI16Succeeds) {
  auto qi16_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsSupportedByTfliteQuantizeOrDequantizeOps(
      dyn_cast_or_null<IntegerType>(qi16_type.getStorageType())));
}

TEST_F(IsSupportedByTfliteQuantizeOrDequantizeOpsTest, StorageTypeUI8Succeeds) {
  auto qi8_type = quant::UniformQuantizedType::get(
      /*flags=*/QuantizationFlags::Signed, builder_.getI8Type(),
      builder_.getF32Type(),
      /*scale=*/1.0,
      /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  EXPECT_TRUE(IsSupportedByTfliteQuantizeOrDequantizeOps(
      dyn_cast_or_null<IntegerType>(qi8_type.getStorageType())));
}

}  // namespace
}  // namespace quant
}  // namespace mlir
