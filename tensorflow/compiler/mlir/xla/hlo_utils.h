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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_UTILS_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, mlir::Builder builder);

Status CopyDenseElementsDataToXlaFormat(mlir::DenseElementsAttr data,
                                        std::vector<uint8>* output);

StatusOr<int> GetElementTypeBytes(mlir::Type type);

// Creates an DenseIntElementsAttr using the elements of the vector and the
// optional shape.
mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape = {});

StatusOr<mlir::Type> ConvertPrimitiveTypeToMLIRType(PrimitiveType element_type,
                                                    mlir::Builder builder);

mlir::mhlo::GatherDimensionNumbersAttr CreateGatherDimensionNumbers(
    const GatherDimensionNumbers& input, mlir::Builder builder);

// Converts the given XLA shape for tensors to the template MLIR type. Note that
// any dynamic bounds in the input shape is lost and those dimensions are fully
// dynamic in the MLIR type.
template <typename TypeT>
static StatusOr<TypeT> ConvertTensorShapeToType(const Shape& shape,
                                                mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMLIRType(shape.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  for (auto element : llvm::enumerate(shape.dynamic_dimensions())) {
    bool is_dynamic = element.value();
    if (is_dynamic) {
      array[element.index()] = mlir::ShapedType::kDynamicSize;
    }
  }
  return TypeT::get(array, element_type_or.ValueOrDie());
}

StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder);

template <>
inline StatusOr<mlir::MemRefType> ConvertTensorShapeToType(
    const Shape& shape, mlir::Builder builder) {
  if (shape.is_dynamic()) {
    return tensorflow::errors::FailedPrecondition(
        "MemRefType don't support dynamic shapes");
  }
  return ConvertTensorShapeToMemRefType(shape, builder);
}

// Converts the given XLA shape to the template MLIR type. Note that  any
// dynamic bounds in the input shape is lost and those dimensions are fully
// dynamic in the MLIR type.
template <typename TypeT>
static StatusOr<mlir::Type> ConvertShapeToType(const Shape& shape,
                                               mlir::Builder builder) {
  if (shape.IsTuple()) {
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype,
                          ConvertShapeToType<TypeT>(subtype, builder));
      contents.push_back(mlir_subtype);
    }
    return builder.getTupleType(contents);
  }
  if (shape.IsToken()) {
    return mlir::mhlo::TokenType::get(builder.getContext());
  }
  return ConvertTensorShapeToType<TypeT>(shape, builder);
}

::xla::StatusOr<::xla::HloOpcode> MhloToHloOpcode(mlir::Operation* op);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_UTILS_H_
