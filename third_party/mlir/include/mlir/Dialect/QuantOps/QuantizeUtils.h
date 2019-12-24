//===- QuantizeUtils.h - Support utilities for quantization -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
