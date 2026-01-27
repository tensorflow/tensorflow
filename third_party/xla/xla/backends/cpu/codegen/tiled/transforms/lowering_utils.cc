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

#include <cstdint>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace xla::cpu {

static llvm::SmallVector<mlir::Value> MakeZeroIndices(mlir::OpBuilder& builder,
                                                      mlir::Location loc,
                                                      int64_t rank) {
  return llvm::SmallVector<mlir::Value>(
      rank, mlir::arith::ConstantIndexOp::create(builder, loc, 0));
}

mlir::VectorType GetVectorType(mlir::ShapedType type) {
  return mlir::VectorType::get(type.getShape(), type.getElementType());
}

mlir::TypedValue<mlir::VectorType> ReadTensorToVector(mlir::OpBuilder& builder,
                                                      mlir::Value input) {
  if (input.getType().isIntOrFloat()) {
    return mlir::vector::FromElementsOp::create(
        builder, input.getLoc(), mlir::VectorType::get({}, input.getType()),
        input);
  }

  auto input_tensor =
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(input);
  auto vector_type = GetVectorType(input_tensor.getType());

  return mlir::vector::TransferReadOp::create(
      builder, input.getLoc(), vector_type, input_tensor,
      MakeZeroIndices(builder, input.getLoc(), vector_type.getRank()),
      std::nullopt);
}

mlir::RankedTensorType GetTensorType(mlir::ShapedType type) {
  return mlir::RankedTensorType::get(type.getShape(), type.getElementType());
}

mlir::TypedValue<mlir::RankedTensorType> WriteVectorToTensor(
    mlir::OpBuilder& builder, mlir::Value input) {
  if (input.getType().isIntOrFloat()) {
    return mlir::tensor::FromElementsOp::create(
        builder, input.getLoc(),
        mlir::RankedTensorType::get({}, input.getType()), input);
  }

  auto input_vector = mlir::cast<mlir::TypedValue<mlir::VectorType>>(input);
  mlir::VectorType vector_type = input_vector.getType();
  auto empty_tensor = mlir::tensor::EmptyOp::create(
      builder, input.getLoc(), vector_type.getShape(),
      vector_type.getElementType());
  return mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
      mlir::vector::TransferWriteOp::create(
          builder, input.getLoc(), input, empty_tensor,
          MakeZeroIndices(builder, input.getLoc(), vector_type.getRank()))
          .getResult());
}

mlir::TypedValue<mlir::MemRefType> CreateBufferOfShape(mlir::OpBuilder& builder,
                                                       mlir::Location loc,
                                                       mlir::ShapedType shape) {
  mlir::MemRefType memrefType =
      mlir::MemRefType::get(shape.getShape(), shape.getElementType());
  return mlir::memref::AllocaOp::create(builder, loc, memrefType);
}

}  // namespace xla::cpu
