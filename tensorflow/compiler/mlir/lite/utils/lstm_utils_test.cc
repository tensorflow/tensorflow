/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {

FuncOp createFusedFunc(mlir::Builder* builder) {
  SmallVector<int64_t, 2> input_shape{1, 2};
  SmallVector<int64_t, 2> weight_shape{3, 12};
  SmallVector<int64_t, 1> bias_shape{2};
  SmallVector<int64_t, 2> projection_shape{1, 2};
  SmallVector<int64_t, 1> layer_norm_scale{4};
  SmallVector<int64_t, 2> output_shape{1, 2};
  auto input_type = builder->getTensorType(input_shape, builder->getF32Type());
  auto weight_type =
      builder->getTensorType(weight_shape, builder->getF32Type());
  auto bias_type = builder->getTensorType(bias_shape, builder->getF32Type());
  auto projection_type =
      builder->getTensorType(projection_shape, builder->getF32Type());
  auto layer_norm_scale_type =
      builder->getTensorType(layer_norm_scale, builder->getF32Type());
  auto output_type =
      builder->getTensorType(output_shape, builder->getF32Type());
  SmallVector<mlir::Type, 4> input_types{input_type, weight_type, bias_type,
                                         projection_type,
                                         layer_norm_scale_type};
  auto func_type = builder->getFunctionType(input_types, output_type);

  auto func =
      FuncOp::create(mlir::NameLoc::get(builder->getIdentifier("fused_func"),
                                        builder->getContext()),
                     "fused_func", func_type, {});
  func.addEntryBlock();
  return func;
}

// TODO(ashwinm): Revisit if this test should be moved to a test pass
// with FileCheck test after the pass that consumes the lstm_utils to stack
// the layers.
class LstmUtilsTest : public ::testing::Test {
 protected:
  LstmUtilsTest() {}

  void SetUp() override {
    builder_ = std::unique_ptr<mlir::Builder>(new Builder(&context_));
    fused_lstm_func_ = createFusedFunc(builder_.get());
  }

  void TearDown() override {
    fused_lstm_func_.erase();
    builder_.reset();
  }
  FuncOp fused_lstm_func_;
  mlir::MLIRContext context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(LstmUtilsTest, ConvertLSTMCellSimple) {
  auto convert =
      mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM(fused_lstm_func_, false);

  auto result = convert.Initialize();
  EXPECT_FALSE(failed(result));

  convert.RewriteFunc();
  fused_lstm_func_.dump();

  // verify transpose
  EXPECT_EQ(
      fused_lstm_func_.getAttrOfType<StringAttr>("tf._implements").getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_lstm_func_.getType().getNumResults(), 1);

  auto transpose_op = fused_lstm_func_.getBody().front().begin();
  transpose_op++;
  EXPECT_EQ(transpose_op->getOperand(0)
                ->getType()
                .cast<RankedTensorType>()
                .getDimSize(0),
            3);
  EXPECT_EQ(transpose_op->getOperand(0)
                ->getType()
                .cast<RankedTensorType>()
                .getDimSize(1),
            12);
  EXPECT_EQ(
      transpose_op->getResult(0)->getType().cast<RankedTensorType>().getDimSize(
          0),
      12);
  EXPECT_EQ(
      transpose_op->getResult(0)->getType().cast<RankedTensorType>().getDimSize(
          1),
      3);

  auto return_op = fused_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(return_op->getName().getStringRef(),
            mlir::ReturnOp::getOperationName());
  return_op++;
  EXPECT_EQ(return_op->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(return_op->getNumOperands(), 24);
  EXPECT_EQ(return_op->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(return_op->getOperand(1)->getType().isa<NoneType>());
  // input layer norm is None
  EXPECT_TRUE(return_op->getOperand(20)->getType().isa<NoneType>());

  EXPECT_EQ(fused_lstm_func_.getType().getNumResults(), 1);
  auto output_types = fused_lstm_func_.getType().getResults();
  SmallVector<int64_t, 2> output_shape{1, 2};
  EXPECT_EQ(output_types[0].cast<RankedTensorType>().getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(output_types[0].cast<RankedTensorType>().getDimSize(i),
              output_shape[i]);
  }
}

TEST_F(LstmUtilsTest, ConvertLSTMCellSimpleToFusedLSTMCoupleInputForget) {
  auto convert =
      mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM(fused_lstm_func_, true);

  auto result = convert.Initialize();
  EXPECT_FALSE(failed(result));

  convert.RewriteFunc();
  fused_lstm_func_.dump();

  auto it = fused_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(), mlir::ReturnOp::getOperationName());
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = true, so input2input is None.
  EXPECT_TRUE(it->getOperand(1)->getType().isa<NoneType>());
}

TEST_F(LstmUtilsTest, ConvertLayerNormLSTMCellSimpleToFusedLSTM) {
  auto convert = mlir::TFL::ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM(
      fused_lstm_func_, false);

  auto result = convert.Initialize();
  EXPECT_FALSE(failed(result));

  convert.RewriteFunc();
  fused_lstm_func_.dump();

  EXPECT_EQ(
      fused_lstm_func_.getAttrOfType<StringAttr>("tf._implements").getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_lstm_func_.getType().getNumResults(), 1);

  auto it = fused_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(), mlir::ReturnOp::getOperationName());
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(it->getOperand(1)->getType().isa<NoneType>());

  // input layer norm
  EXPECT_FALSE(it->getOperand(20)->getType().isa<NoneType>());
  EXPECT_EQ(
      it->getOperand(20)->getType().cast<RankedTensorType>().getShape().size(),
      1);
  EXPECT_EQ(
      it->getOperand(20)->getType().cast<RankedTensorType>().getDimSize(0), 3);

  EXPECT_EQ(fused_lstm_func_.getType().getNumResults(), 1);
  auto output_types = fused_lstm_func_.getType().getResults();
  SmallVector<int64_t, 2> output_shape{1, 2};
  EXPECT_EQ(output_types[0].cast<RankedTensorType>().getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(output_types[0].cast<RankedTensorType>().getDimSize(i),
              output_shape[i]);
  }
}

}  // namespace TFL
}  // namespace mlir
