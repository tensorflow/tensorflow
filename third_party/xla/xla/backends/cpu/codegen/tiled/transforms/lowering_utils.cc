/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace xla::cpu {

mlir::VectorType GetVectorType(mlir::RankedTensorType tensor_type) {
  return mlir::VectorType::get(tensor_type.getShape(),
                               tensor_type.getElementType());
}

mlir::TypedValue<mlir::VectorType> CastToVector(mlir::OpBuilder& builder,
                                                mlir::Value input) {
  if (input.getType().isIntOrFloat()) {
    return builder.create<mlir::vector::FromElementsOp>(
        input.getLoc(), mlir::VectorType::get({}, input.getType()), input);
  }

  auto input_tensor =
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(input);
  auto vector_type = GetVectorType(input_tensor.getType());
  auto cast_op = builder.create<mlir::UnrealizedConversionCastOp>(
      input.getLoc(), vector_type, input_tensor);
  return mlir::cast<mlir::TypedValue<mlir::VectorType>>(cast_op.getResult(0));
}

mlir::RankedTensorType GetTensorType(mlir::VectorType vector_type) {
  return mlir::RankedTensorType::get(vector_type.getShape(),
                                     vector_type.getElementType());
}

mlir::TypedValue<mlir::RankedTensorType> CastToTensor(mlir::OpBuilder& builder,
                                                      mlir::Value input) {
  if (input.getType().isIntOrFloat()) {
    return builder.create<mlir::tensor::FromElementsOp>(
        input.getLoc(), mlir::RankedTensorType::get({}, input.getType()),
        input);
  }

  auto input_vector = mlir::cast<mlir::TypedValue<mlir::VectorType>>(input);
  auto tensor_type = GetTensorType(input_vector.getType());
  auto cast_op = builder.create<mlir::UnrealizedConversionCastOp>(
      input.getLoc(), tensor_type, input_vector);
  return mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
      cast_op.getResult(0));
}

}  // namespace xla::cpu
