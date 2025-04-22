/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

// Test fixture for AreBroadcastAndReductionAxesIndependent function.
class BroadcastAndReductionAxesIndependentTest : public ::testing::Test {
 protected:
  BroadcastAndReductionAxesIndependentTest() : builder_(&context_) {
    context_.loadDialect<arith::ArithDialect>();
  }

  // Builds an mlir::Value representing a tensor with the given shape.
  Value BuildTensor(ArrayRef<int64_t> shape) {
    return builder_.create<arith::ConstantOp>(
        builder_.getUnknownLoc(),
        RankedTensorType::get(shape, builder_.getF32Type()),
        builder_.getZeroAttr(
            RankedTensorType::get(shape, builder_.getF32Type())));
  }

  // Builds a DenseElementsAttr representing an integer array.
  DenseElementsAttr BuildIntArrayAttr(ArrayRef<int32_t> values) {
    return DenseElementsAttr::get(
        RankedTensorType::get({static_cast<int32_t>(values.size())},
                              builder_.getI32Type()),
        values);
  }

  MLIRContext context_;
  OpBuilder builder_;
};

TEST_F(BroadcastAndReductionAxesIndependentTest, IndependentAxes) {
  Value input_tensor = BuildTensor({2, 1, 4, 1});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_TRUE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, OverlappingAxes) {
  Value input_tensor = BuildTensor({1, 3, 4, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, EmptyReductionAxes) {
  Value input_tensor = BuildTensor({1, 3, 1, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_TRUE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, UnrankedInput) {
  Value input_tensor = builder_.create<arith::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getF32Type(),
      builder_.getZeroAttr(builder_.getF32Type()));
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = BuildIntArrayAttr({2, 3, 4, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, InvalidReductionAxesType) {
  Value input_tensor = BuildTensor({2, 3, 4, 5});
  DenseElementsAttr reduction_axes = DenseElementsAttr::get(
      RankedTensorType::get({2}, builder_.getF32Type()), {1.0f, 2.0f});
  DenseElementsAttr target_shape = BuildIntArrayAttr({1, 3, 1, 5});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

TEST_F(BroadcastAndReductionAxesIndependentTest, InvalidTargetShapeType) {
  Value input_tensor = BuildTensor({2, 3, 4, 5});
  DenseElementsAttr reduction_axes = BuildIntArrayAttr({0, 2});
  DenseElementsAttr target_shape = DenseElementsAttr::get(
      RankedTensorType::get({2}, builder_.getF32Type()), {1.0f, 2.0f});

  EXPECT_FALSE(AreBroadcastAndReductionAxesIndependent(
      input_tensor, reduction_axes, target_shape));
  input_tensor.getDefiningOp()->destroy();
}

}  // namespace
}  // namespace TFL

}  // namespace mlir
