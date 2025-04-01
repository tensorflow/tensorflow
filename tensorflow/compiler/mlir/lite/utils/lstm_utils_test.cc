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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {

func::FuncOp createLstmCompositeFunc(mlir::Builder* builder, bool ln,
                                     bool cifg) {
  SmallVector<int64_t, 2> input_shape{1, 2};
  SmallVector<int64_t, 2> weight_shape{3, 12};
  SmallVector<int64_t, 1> bias_shape{2};
  SmallVector<int64_t, 2> projection_shape{1, 2};
  SmallVector<int64_t, 1> layer_norm_scale{4};
  SmallVector<int64_t, 2> output_shape{1, 2};
  auto input_type = RankedTensorType::get(input_shape, builder->getF32Type());
  auto weight_type = RankedTensorType::get(weight_shape, builder->getF32Type());
  auto bias_type = RankedTensorType::get(bias_shape, builder->getF32Type());
  auto projection_type =
      RankedTensorType::get(projection_shape, builder->getF32Type());
  auto layer_norm_scale_type =
      RankedTensorType::get(layer_norm_scale, builder->getF32Type());
  auto output_type = RankedTensorType::get(output_shape, builder->getF32Type());
  SmallVector<mlir::Type, 4> input_types{input_type, weight_type, bias_type,
                                         projection_type,
                                         layer_norm_scale_type};
  auto func_type = builder->getFunctionType(input_types, output_type);

  auto func = func::FuncOp::create(
      mlir::NameLoc::get(builder->getStringAttr("fused_func")), "fused_func",
      func_type, {});
  func.addEntryBlock();

  std::vector<std::string> attributes;
  if (ln) {
    attributes.push_back(kLayerNormalizedLstmCellSimple);
  } else {
    attributes.push_back(kLstmCellSimple);
  }

  if (cifg) {
    attributes.push_back(kCoupleInputForgetGates);
  }

  mlir::StringAttr attr_values =
      builder->getStringAttr(llvm::join(attributes, ","));

  func->setAttr(kTFImplements, attr_values);
  return func;
}

class LstmUtilsTest : public ::testing::Test {
 protected:
  LstmUtilsTest() {}

  void SetUp() override {
    context_ = std::make_unique<mlir::MLIRContext>();
    context_->loadDialect<arith::ArithDialect, mlir::func::FuncDialect,
                          tensor::TensorDialect, mlir::TF::TensorFlowDialect,
                          TensorFlowLiteDialect>();
    builder_ = std::make_unique<mlir::Builder>(context_.get());
    fused_lstm_func_ = createLstmCompositeFunc(builder_.get(), false, false);
    fused_lstm_func_cifg_ =
        createLstmCompositeFunc(builder_.get(), false, true);
    fused_ln_lstm_func_ = createLstmCompositeFunc(builder_.get(), true, false);
  }

  void TearDown() override {
    fused_lstm_func_.erase();
    fused_lstm_func_cifg_.erase();
    fused_ln_lstm_func_.erase();
    builder_.reset();
  }

  func::FuncOp fused_lstm_func_;
  func::FuncOp fused_lstm_func_cifg_;
  func::FuncOp fused_ln_lstm_func_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(LstmUtilsTest, ConvertLSTMCellSimple) {
  mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM convert(fused_lstm_func_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_lstm_func_.dump();

  // verify transpose
  EXPECT_EQ(
      fused_lstm_func_->getAttrOfType<StringAttr>(kTFImplements).getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_lstm_func_.getFunctionType().getNumResults(), 1);

  auto transpose_op = fused_lstm_func_.getBody().front().begin();
  transpose_op++;
  EXPECT_EQ(mlir::cast<RankedTensorType>(transpose_op->getOperand(0).getType())
                .getDimSize(0),
            3);
  EXPECT_EQ(mlir::cast<RankedTensorType>(transpose_op->getOperand(0).getType())
                .getDimSize(1),
            12);
  EXPECT_EQ(mlir::cast<RankedTensorType>(transpose_op->getResult(0).getType())
                .getDimSize(0),
            12);
  EXPECT_EQ(mlir::cast<RankedTensorType>(transpose_op->getResult(0).getType())
                .getDimSize(1),
            3);

  auto it = fused_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;  // tensor_cast
  it++;  // lstm
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(mlir::isa<NoneType>(it->getOperand(1).getType()));
  // input layer norm is None
  EXPECT_TRUE(mlir::isa<NoneType>(it->getOperand(20).getType()));
  // proj_bias is F32
  EXPECT_TRUE(mlir::cast<RankedTensorType>(it->getOperand(17).getType())
                  .getElementType()
                  .isF32());

  // output gate bias is 0 since it is out of bounds of the bias tensor, so
  // we set its value as a const tensor of specified size and value 0.
  EXPECT_TRUE(
      mlir::cast<ElementsAttr>(mlir::cast<mlir::arith::ConstantOp>(
                                   it->getOpOperand(15).get().getDefiningOp())
                                   .getValue())
          .getValues<FloatAttr>()[0]
          .getValue()
          .isExactlyValue(0.0f));

  EXPECT_EQ(fused_lstm_func_.getFunctionType().getNumResults(), 1);
  auto output_types = fused_lstm_func_.getFunctionType().getResults();
  SmallVector<int64_t, 2> output_shape{1, mlir::ShapedType::kDynamic};
  EXPECT_EQ(mlir::cast<RankedTensorType>(output_types[0]).getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(mlir::cast<RankedTensorType>(output_types[0]).getDimSize(i),
              output_shape[i]);
  }
}

TEST_F(LstmUtilsTest, ConvertLSTMCellSimpleToFusedLSTMCoupleInputForget) {
  mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM convert(fused_lstm_func_cifg_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_lstm_func_cifg_.dump();

  llvm::SmallVector<std::string, 2> attributes{kLstmCellSimple,
                                               kCoupleInputForgetGates};
  EXPECT_EQ(fused_lstm_func_cifg_->getAttrOfType<StringAttr>(kTFImplements)
                .getValue(),
            llvm::join(attributes, ","));

  auto it = fused_lstm_func_cifg_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = true, so input2input is None.
  EXPECT_TRUE(mlir::isa<NoneType>(it->getOperand(1).getType()));
}

TEST_F(LstmUtilsTest, ConvertLayerNormLSTMCellSimpleToFusedLSTM) {
  mlir::TFL::ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM convert(
      fused_ln_lstm_func_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_ln_lstm_func_.dump();

  EXPECT_EQ(
      fused_ln_lstm_func_->getAttrOfType<StringAttr>(kTFImplements).getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_ln_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_ln_lstm_func_.getFunctionType().getNumResults(), 1);

  auto it = fused_ln_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(mlir::isa<NoneType>(it->getOperand(1).getType()));

  // input layer norm
  EXPECT_FALSE(mlir::isa<NoneType>(it->getOperand(20).getType()));
  EXPECT_EQ(mlir::cast<RankedTensorType>(it->getOperand(20).getType())
                .getShape()
                .size(),
            1);
  EXPECT_EQ(
      mlir::cast<RankedTensorType>(it->getOperand(20).getType()).getDimSize(0),
      3);

  EXPECT_EQ(fused_ln_lstm_func_.getFunctionType().getNumResults(), 1);
  auto output_types = fused_ln_lstm_func_.getFunctionType().getResults();
  SmallVector<int64_t, 2> output_shape{1, mlir::ShapedType::kDynamic};
  EXPECT_EQ(mlir::cast<RankedTensorType>(output_types[0]).getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(mlir::cast<RankedTensorType>(output_types[0]).getDimSize(i),
              output_shape[i]);
  }
}

}  // namespace TFL
}  // namespace mlir
