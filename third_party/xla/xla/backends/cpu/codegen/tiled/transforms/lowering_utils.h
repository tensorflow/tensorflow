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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_LOWERING_UTILS_H_
#define XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_LOWERING_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace xla::cpu {

mlir::AffineMapAttr GetOperandIndexingMap(
    mlir::OpBuilder& builder, int64_t iterator_count, int64_t rank,
    llvm::ArrayRef<int64_t> batch_dims,
    llvm::ArrayRef<int64_t> contracting_dims, int64_t free_dim_offset);

mlir::AffineMapAttr GetOutputIndexingMap(mlir::OpBuilder& builder,
                                         int64_t iterator_count,
                                         int64_t batch_dim_count,
                                         int64_t contracting_dim_count);

mlir::ArrayAttr GetIteratorTypes(mlir::OpBuilder& builder,
                                 int64_t iterator_count,
                                 int64_t batch_dim_count,
                                 int64_t contracting_dim_count);

mlir::LogicalResult GetFusedAddUnit(mlir::Operation* op,
                                    mlir::PatternRewriter& rewriter,
                                    mlir::Operation*& add_op,
                                    mlir::Value& accumulator);

// Get the vector type that has the same shape and element type as the tensor
// type.
mlir::VectorType GetVectorType(mlir::ShapedType tensor_type);

// Cast the input to a vector value.
// If the input is a scalar it will be simply constructed as a
// vector.from_elements to create a 0D vector.
// If it is a vector it will be cast to a vector using an unrealized cast op.
// Any other type will crash.
mlir::TypedValue<mlir::VectorType> ReadTensorToVector(mlir::OpBuilder& builder,
                                                      mlir::Value input);

// Get the tensor type that has the same shape and element type as the vector
// type.
mlir::RankedTensorType GetTensorType(mlir::ShapedType vector_type);

// Cast the input to a tensor value.
// If the input is a scalar it will be simply constructed as a
// tensor.from_elements to create a 0D tensor.
// If it is a vector it will be cast to a tensor using an unrealized cast op.
// Any other type will crash.
mlir::TypedValue<mlir::RankedTensorType> WriteVectorToTensor(
    mlir::OpBuilder& builder, mlir::Value input);

mlir::TypedValue<mlir::MemRefType> CreateBufferOfShape(mlir::OpBuilder& builder,
                                                       mlir::Location loc,
                                                       mlir::ShapedType shape);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_LOWERING_UTILS_H_
