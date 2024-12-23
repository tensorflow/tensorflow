/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/unfold_large_splat_constants_pass.h"

#include <cstddef>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"

namespace mlir {
namespace TFL {
namespace {
// The threshold of constant bits to be unfolded (1Mb). If there is a splat
// constant with size equal or greater to this threshold, then it will be
// unfolded back to a regular `tfl.fill` operation.
constexpr size_t kConstantSizeThresholdInBits = 1e+6;
void MaybeUnfoldLargeSplatConstant(mlir::OpBuilder* op_builder,
                                   mlir::arith::ConstantOp const_op) {
  auto splat_elements_attr =
      mlir::dyn_cast<SplatElementsAttr>(const_op.getValue());
  if (!splat_elements_attr) {
    return;
  }
  auto element_type = splat_elements_attr.getType().getElementType();
  if (!(element_type.isF32() || element_type.isF16() ||
        element_type.isInteger(1) || element_type.isInteger(32) ||
        element_type.isInteger(64))) {
    return;
  }
  if (splat_elements_attr.getNumElements() *
          splat_elements_attr.getType().getElementTypeBitWidth() <
      kConstantSizeThresholdInBits) {
    return;
  }

  op_builder->setInsertionPoint(const_op);
  mlir::arith::ConstantOp fill_shape =
      op_builder->create<mlir::arith::ConstantOp>(
          const_op->getLoc(), DenseIntElementsAttr::get(
                                  tensorflow::GetTypeFromTFTensorShape(
                                      {splat_elements_attr.getType().getRank()},
                                      op_builder->getI64Type()),
                                  splat_elements_attr.getType().getShape()));
  mlir::arith::ConstantOp fill_value =
      op_builder->create<mlir::arith::ConstantOp>(
          const_op->getLoc(),
          DenseElementsAttr::get(
              tensorflow::GetTypeFromTFTensorShape(
                  {}, splat_elements_attr.getType().getElementType()),
              splat_elements_attr.getSplatValue<Attribute>()));
  TFL::FillOp fill = op_builder->create<TFL::FillOp>(
      const_op->getLoc(), splat_elements_attr.getType(), fill_shape,
      fill_value);
  const_op->replaceAllUsesWith(fill);
  const_op->erase();
}
}  // namespace

void UnfoldLargeSplatConstantPass::runOnOperation() {
  auto module = getOperation();

  mlir::OpBuilder op_builder(&module.getBodyRegion());
  module.walk([&](mlir::arith::ConstantOp const_op) {
    MaybeUnfoldLargeSplatConstant(&op_builder, const_op);
  });
}

}  // namespace TFL
}  // namespace mlir
