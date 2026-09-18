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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace xla::cpu {

mlir::AffineMapAttr GetOperandIndexingMap(
    mlir::OpBuilder& builder, int64_t iterator_count, int64_t rank,
    llvm::ArrayRef<int64_t> batch_dims,
    llvm::ArrayRef<int64_t> contracting_dims, int64_t free_dim_offset) {
  llvm::SmallVector<unsigned> targets(rank, -1);
  unsigned idx = 0;
  for (int64_t dim : batch_dims) {
    targets[dim] = idx++;
  }
  for (int64_t dim : contracting_dims) {
    targets[dim] = idx++;
  }
  for (unsigned& target : targets) {
    if (target == -1) {
      target = free_dim_offset + idx++;
    }
  }
  auto affine_map = mlir::AffineMap::getMultiDimMapWithTargets(
      iterator_count, targets, builder.getContext());

  return mlir::AffineMapAttr::get(affine_map);
}

mlir::AffineMapAttr GetOutputIndexingMap(mlir::OpBuilder& builder,
                                         int64_t iterator_count,
                                         int64_t batch_dim_count,
                                         int64_t contracting_dim_count) {
  llvm::SmallVector<unsigned> targets(iterator_count - contracting_dim_count);
  unsigned idx = 0;
  for (int64_t dim = 0; dim != batch_dim_count; ++dim) {
    targets[dim] = idx++;
  }
  idx += contracting_dim_count;
  int64_t total_free_dims =
      iterator_count - batch_dim_count - contracting_dim_count;
  for (int64_t dim = 0; dim != total_free_dims; ++dim) {
    targets[batch_dim_count + dim] = idx++;
  }
  auto affine_map = mlir::AffineMap::getMultiDimMapWithTargets(
      iterator_count, targets, builder.getContext());

  return mlir::AffineMapAttr::get(affine_map);
}

mlir::ArrayAttr GetIteratorTypes(mlir::OpBuilder& builder,
                                 int64_t iterator_count,
                                 int64_t batch_dim_count,
                                 int64_t contracting_dim_count) {
  llvm::SmallVector<mlir::Attribute> iterator_types;
  iterator_types.reserve(iterator_count);
  for (int64_t dim = 0; dim != batch_dim_count; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::parallel));
  }
  for (int64_t dim = 0; dim != contracting_dim_count; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::reduction));
  }
  int64_t free_dims = iterator_count - batch_dim_count - contracting_dim_count;
  for (int64_t dim = 0; dim != free_dims; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::parallel));
  }

  return mlir::ArrayAttr::get(builder.getContext(), iterator_types);
}

mlir::LogicalResult GetFusedAddUnit(mlir::Operation* op,
                                    mlir::PatternRewriter& rewriter,
                                    mlir::Operation*& add_op,
                                    mlir::Value& accumulator) {
  const mlir::Location op_loc = op->getLoc();
  if (!op->hasOneUse()) {
    return rewriter.notifyMatchFailure(
        op_loc,
        "Dot op must have exactly one user in order to be lowered to vector "
        "contract.");
  }

  add_op = *op->getUsers().begin();
  if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(add_op)) {
    return rewriter.notifyMatchFailure(
        op_loc,
        "Dot op must be consumed by an AddOp to be convertible to vector "
        "contract.");
  }

  accumulator = add_op->getOperand(0) == op->getResult(0)
                    ? add_op->getOperand(1)
                    : add_op->getOperand(0);
  return mlir::success();
}

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
