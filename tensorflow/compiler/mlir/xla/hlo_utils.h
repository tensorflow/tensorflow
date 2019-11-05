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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_HLO_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_HLO_UTILS_H_

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const Literal& literal, mlir::Builder builder);

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64> vector, mlir::Builder builder);

template <typename TypeT>
StatusOr<TypeT> ConvertTensorShapeToType(const Shape& shape,
                                         mlir::Builder builder) {
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());

  switch (shape.element_type()) {
    case PrimitiveType::PRED:
      return TypeT::get(array, builder.getI1Type());
    case PrimitiveType::F16:
      return TypeT::get(array, builder.getF16Type());
    case PrimitiveType::F32:
      return TypeT::get(array, builder.getF32Type());
    case PrimitiveType::F64:
      return TypeT::get(array, builder.getF64Type());
    case PrimitiveType::S8:
      return TypeT::get(array, builder.getIntegerType(8));
    case PrimitiveType::S16:
      return TypeT::get(array, builder.getIntegerType(16));
    case PrimitiveType::S32:
      return TypeT::get(array, builder.getIntegerType(32));
    case PrimitiveType::S64:
      return TypeT::get(array, builder.getIntegerType(64));
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", PrimitiveType_Name(shape.element_type())));
  }
}

template <typename TypeT>
StatusOr<mlir::Type> ConvertShapeToType(const Shape& shape,
                                        mlir::Builder builder) {
  if (shape.IsTuple()) {
    mlir::Type mlir_type;
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype,
                          ConvertShapeToType<TypeT>(subtype, builder));
      contents.push_back(mlir_subtype);
    }
    return builder.getTupleType(contents);
  }
  return ConvertTensorShapeToType<TypeT>(shape, builder);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_HLO_UTILS_H_
