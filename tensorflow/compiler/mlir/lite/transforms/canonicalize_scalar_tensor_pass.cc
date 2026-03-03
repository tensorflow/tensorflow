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

#include "tensorflow/compiler/mlir/lite/transforms/canonicalize_scalar_tensor_pass.h"

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Converts a scalar tensor type (shape []) to a 1D tensor type (shape [1]).
mlir::Type ConvertScalarTo1DType(mlir::Type type) {
  if (auto ranked_ty = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (ranked_ty.getRank() == 0) {
      return mlir::RankedTensorType::get({1}, ranked_ty.getElementType());
    }
  }
  return type;
}

}  // namespace

void CanonicalizeScalarTensorPass::runOnOperation() {
  mlir::MLIRContext* ctx = &getContext();
  func::FuncOp func_op = getOperation();

  // Update Function signatures and entry block arguments.
  mlir::FunctionType old_func_type = func_op.getFunctionType();

  mlir::SmallVector<mlir::Type> new_input_types;
  for (mlir::Type input_type : old_func_type.getInputs()) {
    new_input_types.push_back(ConvertScalarTo1DType(input_type));
  }

  mlir::SmallVector<mlir::Type> new_result_types;
  for (mlir::Type result_type : old_func_type.getResults()) {
    new_result_types.push_back(ConvertScalarTo1DType(result_type));
  }

  func_op.setType(
      mlir::FunctionType::get(ctx, new_input_types, new_result_types));

  if (!func_op.empty()) {
    for (auto [arg, new_type] :
         llvm::zip(func_op.getArguments(), new_input_types)) {
      arg.setType(new_type);
    }
  }

  // Walk all operations.
  func_op->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::func::FuncOp>(op)) {
      return;
    }

    // Handle results for all other operations.
    for (mlir::OpResult result : op->getResults()) {
      mlir::Type old_type = result.getType();
      mlir::Type new_type = ConvertScalarTo1DType(old_type);

      if (new_type != old_type) {
        if (op->hasTrait<mlir::OpTrait::ConstantLike>() ||
            mlir::isa<mlir::TFL::ReshapeOp>(op)) {
          // Insert explicit Reshape for constants to maintain valid IR.
          mlir::OpBuilder builder(op->getContext());
          builder.setInsertionPointAfter(op);

          auto shape_type =
              mlir::RankedTensorType::get({1}, builder.getI32Type());
          auto shape_attr = mlir::DenseIntElementsAttr::get(shape_type, {1});
          auto shape_const = mlir::arith::ConstantOp::create(
              builder, op->getLoc(), shape_type, shape_attr);

          auto reshape_op = mlir::TFL::ReshapeOp::create(
              builder, op->getLoc(), new_type, result, shape_const);

          result.replaceAllUsesExcept(reshape_op.getResult(), reshape_op);
        } else {
          result.setType(new_type);
        }
      }
    }
  });
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateCanonicalizeScalarTensorPass() {
  return std::make_unique<CanonicalizeScalarTensorPass>();
}

}  // namespace TFL
}  // namespace mlir
