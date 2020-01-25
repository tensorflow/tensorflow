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

#include "tensorflow/compiler/mlir/xla/hlo_utils.h"

#include "mlir/IR/AffineMap.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "tensorflow/compiler/xla/literal.h"

namespace xla {
namespace {

using mlir::AffineMap;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::ShapedType;
using xla::Literal;
using xla::StatusOr;

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(const ShapedType& type,
                                                     const Literal& literal) {
  auto data_span = literal.data<CppType>();
  return ::mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data_span.data(), data_span.size()));
}

llvm::SmallVector<AffineMap, 2> GetPermutationIfAvailable(
    const Shape& shape, mlir::Builder builder) {
  if (!shape.has_layout() || shape.layout().minor_to_major().empty()) {
    return {};
  }
  llvm::SmallVector<unsigned, 2> permutation;
  for (auto dim : llvm::reverse(shape.layout().minor_to_major())) {
    permutation.push_back(dim);
  }
  return {AffineMap::getPermutationMap(permutation, builder.getContext())};
}

}  // namespace

StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  using mlir::MemRefType;
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());

  switch (shape.element_type()) {
    case PrimitiveType::PRED: {
      return MemRefType::get(array, builder.getI1Type(),
                             GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::F16:
        return MemRefType::get(array, builder.getF16Type(),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::F32:
        return MemRefType::get(array, builder.getF32Type(),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::F64:
        return MemRefType::get(array, builder.getF64Type(),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::S8:
        return MemRefType::get(array, builder.getIntegerType(8),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::S16:
        return MemRefType::get(array, builder.getIntegerType(16),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::S32:
        return MemRefType::get(array, builder.getIntegerType(32),
                               GetPermutationIfAvailable(shape, builder));
      case PrimitiveType::S64:
        return MemRefType::get(array, builder.getIntegerType(64),
                               GetPermutationIfAvailable(shape, builder));
      default:
        return tensorflow::errors::Internal(absl::StrCat(
            "Unsupported type: ", PrimitiveType_Name(shape.element_type())));
    }
  }
}

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const Literal& literal, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  auto element_type = literal.shape().element_type();
  switch (element_type) {
    case PrimitiveType::PRED:
      return CreateDenseAttrFromLiteral<bool>(type, literal);
    case PrimitiveType::F16:
      return CreateDenseAttrFromLiteral<float>(type, literal);
    case PrimitiveType::F32:
      return CreateDenseAttrFromLiteral<float>(type, literal);
    case PrimitiveType::F64:
      return CreateDenseAttrFromLiteral<double>(type, literal);
    case PrimitiveType::S8:
      return CreateDenseAttrFromLiteral<int8>(type, literal);
    case PrimitiveType::S16:
      return CreateDenseAttrFromLiteral<int16>(type, literal);
    case PrimitiveType::S32:
      return CreateDenseAttrFromLiteral<int32>(type, literal);
    case PrimitiveType::S64:
      return CreateDenseAttrFromLiteral<int64>(type, literal);
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64> vector, mlir::Builder builder) {
  return mlir::DenseIntElementsAttr::get(
             mlir::RankedTensorType::get(vector.size(),
                                         builder.getIntegerType(64)),
             vector)
      .cast<mlir::DenseIntElementsAttr>();
}

}  // namespace xla
