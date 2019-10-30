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

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/literal.h"

namespace xla {
namespace {

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

}  // namespace

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

mlir::ElementsAttr ConvertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type new_type) {
  auto old_type = getElementTypeOrSelf(elements);
  size_t bit_width = new_type.isBF16() ? 64 : new_type.getIntOrFloatBitWidth();

  if (old_type.isa<mlir::FloatType>()) {
    // mapValues always takes a function returning APInt, even when the output
    // is actually float.
    using func_type = mlir::APInt(const llvm::APFloat&);
    if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
      // Float -> Float
      return elements.mapValues(
          new_type, llvm::function_ref<func_type>(
                        [&newFloatType](const llvm::APFloat& floatVal) {
                          llvm::APFloat newDouble(
                              mlir::FloatAttr::getValueAsDouble(floatVal));
                          bool loses_info = false;
                          newDouble.convert(newFloatType.getFloatSemantics(),
                                            llvm::APFloat::rmNearestTiesToEven,
                                            &loses_info);
                          return newDouble.bitcastToAPInt();
                        }));
    }
    // Float -> Int
    return elements.mapValues(
        new_type, llvm::function_ref<func_type>(
                      [&bit_width](const llvm::APFloat& floatVal) {
                        return llvm::APInt(
                            bit_width,
                            mlir::FloatAttr::getValueAsDouble(floatVal));
                      }));
  }

  // old_type is Integer
  // mapValues always takes a function returning APInt, even when the output
  // is actually float.
  using func_type = llvm::APInt(const llvm::APInt&);
  if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
    // Int -> Float
    return elements.mapValues(
        new_type, llvm::function_ref<func_type>([&newFloatType](
                                                    const llvm::APInt& intVal) {
          llvm::APFloat newDouble(static_cast<double>(intVal.getSExtValue()));
          bool loses_info = false;
          newDouble.convert(newFloatType.getFloatSemantics(),
                            llvm::APFloat::rmNearestTiesToEven, &loses_info);
          return newDouble.bitcastToAPInt();
        }));
  }
  // new_type is Integer
  // Int -> Int
  return elements.mapValues(
      new_type,
      llvm::function_ref<func_type>([&bit_width](const llvm::APInt& intVal) {
        return llvm::APInt(bit_width, intVal.getSExtValue());
      }));
}

}  // namespace xla
