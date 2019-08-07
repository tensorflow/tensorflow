//===- QuantizeUtils.h - Support utilities for quantization -----*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_DIALECT_QUANTOPS_QUANTIZEUTILS_H_
#define MLIR_DIALECT_QUANTOPS_QUANTIZEUTILS_H_

namespace mlir {
class Attribute;
class Type;

namespace quant {
class QuantizedType;
class UniformQuantizedType;
class UniformQuantizedValueConverter;

/// Converts an attribute from a type based on
/// quantizedElementType.getExpressedType() to one based on
/// quantizedElementType.getStorageType(), where quantizedElementType is as from
/// QuantizedType::getQuantizedElementType().
/// Returns nullptr if the conversion is not supported. On success, stores the
/// converted type in outConvertedType.
///
/// Examples:
/// 1. realValue is a primitive value attribute:
/// (realValue: FloatAttr, quantizedElementType: UniformQuantizedType[i8:f32])
///   -> (IntegerAttr, outConvertedType: i8)
/// 2. realValue is an elements attribute:
/// (realValue: DenseElementsAttr[tensor<2x2xf32>],
///  quantizedElementType: UniformQuantizedType[i8:f32])
///   -> (DenseElementsAttr[tensor<2x2xi8>], outConvertedType: tensor<2x2xi8>)
Attribute quantizeAttr(Attribute realValue, QuantizedType quantizedElementType,
                       Type &outConvertedType);

/// Converts an attribute from a type based on
/// quantizedElementType.getExpressedType() to one based on
/// quantizedElementType.getStorageType(), where quantizedElementType is as from
/// QuantizedType::getQuantizedElementType() and casted to an
/// UniformQuantizedType. Returns nullptr if the conversion is not supported. On
/// success, stores the converted type in outConvertedType.
///
/// Examples:
/// 1. realValue is a primitive value attribute:
/// (realValue: FloatAttr, quantizedElementType: UniformQuantizedType[i8:f32])
///   -> (IntegerAttr, outConvertedType: i8)
/// 2. realValue is an elements attribute:
/// (realValue: DenseElementsAttr[tensor<2x2xf32>],
///  quantizedElementType: UniformQuantizedType[i8:f32])
///   -> (DenseElementsAttr[tensor<2x2xi8>], outConvertedType: tensor<2x2xi8>)
Attribute quantizeAttrUniform(Attribute realValue,
                              UniformQuantizedType quantizedElementType,
                              const UniformQuantizedValueConverter &converter,
                              Type &outConvertedType);
} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_QUANTIZEUTILS_H_
