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

#include "tensorflow/compiler/mlir/lite/transforms/downcast_x64_pass.h"

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Recursively converts f64 -> f32 and i64 -> i32 for scalars and tensors.
mlir::Type ConvertX64TypeToX32(mlir::Type type) {
  if (auto tensor_type = mlir::dyn_cast<mlir::TensorType>(type)) {
    mlir::Type element_type = tensor_type.getElementType();
    mlir::Type new_element_type = ConvertX64TypeToX32(element_type);
    if (element_type == new_element_type) {
      return type;
    }

    if (auto ranked_ty = mlir::dyn_cast<mlir::RankedTensorType>(tensor_type)) {
      return mlir::RankedTensorType::get(ranked_ty.getShape(),
                                         new_element_type);
    }
    return mlir::UnrankedTensorType::get(new_element_type);
  }

  if (type.isF64()) {
    return mlir::Float32Type::get(type.getContext());
  }
  if (type.isInteger(64)) {
    return mlir::IntegerType::get(type.getContext(), 32);
  }
  return type;
}

}  // namespace

void DowncastX64Pass::runOnOperation() {
  mlir::MLIRContext* ctx = &getContext();
  func::FuncOp func_op = getOperation();

  // Update Function signatures and entry block arguments.
  func_op->walk([&](mlir::func::FuncOp func_op) {
    mlir::FunctionType old_func_type = func_op.getFunctionType();

    mlir::SmallVector<mlir::Type> new_input_types;
    for (mlir::Type input_type : old_func_type.getInputs()) {
      new_input_types.push_back(ConvertX64TypeToX32(input_type));
    }

    mlir::SmallVector<mlir::Type> new_result_types;
    for (mlir::Type result_type : old_func_type.getResults()) {
      new_result_types.push_back(ConvertX64TypeToX32(result_type));
    }

    func_op.setType(
        mlir::FunctionType::get(ctx, new_input_types, new_result_types));

    // Update block arguments within the function.
    if (!func_op.empty()) {
      for (auto [arg, new_type] :
           llvm::zip(func_op.getArguments(), new_input_types)) {
        arg.setType(new_type);
      }
    }
  });

  // Update all operation results and handle Constant Attributes.
  func_op->walk([&](mlir::Operation* op) {
    // Skip the function itself as we handled it above.
    if (mlir::isa<mlir::func::FuncOp>(op)) {
      return;
    }

    // Handle constant-like operations by inserting CastOps.
    if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      for (mlir::OpResult result : op->getResults()) {
        mlir::Type old_type = result.getType();
        mlir::Type new_type = ConvertX64TypeToX32(old_type);

        if (new_type != old_type) {
          // Insert CastOp to let the constant folder handle the narrowing.
          mlir::OpBuilder builder(op->getContext());
          builder.setInsertionPointAfter(op);

          auto cast_op = mlir::TFL::CastOp::create(builder, op->getLoc(),
                                                   new_type, result);
          result.replaceAllUsesExcept(cast_op.getResult(), cast_op);
        }
      }
      return;
    }

    // Update result types for all operations.
    for (mlir::Value result : op->getResults()) {
      result.setType(ConvertX64TypeToX32(result.getType()));
    }
  });
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>> CreateDowncastX64Pass() {
  return std::make_unique<DowncastX64Pass>();
}

}  // end namespace TFL
}  // end namespace mlir
