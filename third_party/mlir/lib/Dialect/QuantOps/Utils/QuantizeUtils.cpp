//===- QuantizeUtils.cpp - Support utilities for quantization -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantizeUtils.h"
#include "mlir/Dialect/QuantOps/UniformSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::quant;

/// Converts a possible primitive, real expressed value attribute to a
/// corresponding storage attribute (typically FloatAttr -> IntegerAttr).
/// quantizedElementType is the QuantizedType that describes the expressed
/// origValue.
/// Returns a converter Attribute or nullptr if conversion is not possible.
static Attribute convertPrimitiveValueAttr(
    Attribute origRealValue, QuantizedType quantizedElementType,
    const UniformQuantizedValueConverter &converter, Type &outConvertedType) {
  if (origRealValue.isa<FloatAttr>()) {
    FloatAttr floatAttr = origRealValue.cast<FloatAttr>();
    outConvertedType = quantizedElementType.getStorageType();
    return IntegerAttr::get(quantizedElementType.getStorageType(),
                            converter.quantizeFloatToInt(floatAttr.getValue()));
  }

  return nullptr;
}

/// Converts a real expressed DenseFPElementsAttr to a corresponding
/// DenseElementsAttr (typically DenseIntElementsAttr) containing quantized
/// storage values assuming the given quantizedElementType and converter.
static DenseElementsAttr
convertDenseFPElementsAttr(DenseFPElementsAttr realFPElementsAttr,
                           QuantizedType quantizedElementType,
                           const UniformQuantizedValueConverter &converter) {
  // Convert to corresponding quantized value attributes.
  SmallVector<APInt, 8> quantValues;
  if (realFPElementsAttr.isSplat()) {
    quantValues.push_back(
        converter.quantizeFloatToInt(*realFPElementsAttr.begin()));
  } else {
    quantValues.reserve(realFPElementsAttr.getNumElements());
    for (APFloat realVal : realFPElementsAttr) {
      quantValues.push_back(converter.quantizeFloatToInt(realVal));
    }
  }

  // Cast from an expressed-type-based type to storage-type-based type,
  // preserving the dense shape (i.e. tensor<4xf32> -> tensor<4xi8>).
  ShapedType newDenseType =
      quantizedElementType
          .castExpressedToStorageType(realFPElementsAttr.getType())
          .dyn_cast_or_null<ShapedType>();
  if (!newDenseType) {
    return nullptr;
  }
  return DenseIntElementsAttr::get(newDenseType, quantValues);
}

/// Converts a real expressed SplatElementsAttr to a corresponding
/// SplatElementsAttr containing quantized storage values assuming the given
/// quantizedElementType and converter.
static SparseElementsAttr
convertSparseElementsAttr(SparseElementsAttr realSparseAttr,
                          QuantizedType quantizedElementType,
                          const UniformQuantizedValueConverter &converter) {
  DenseElementsAttr realDenseAttr = realSparseAttr.getValues();
  if (!realDenseAttr.isa<DenseFPElementsAttr>()) {
    return nullptr;
  }
  DenseElementsAttr quantDenseAttr =
      convertDenseFPElementsAttr(realDenseAttr.cast<DenseFPElementsAttr>(),
                                 quantizedElementType, converter);
  if (!quantDenseAttr) {
    return nullptr;
  }

  // Cast from an expressed-type-based type to storage-type-based type,
  // preserving the sparse shape (i.e. tensor<4xf32> -> tensor<4xi8>).
  ShapedType newSparseType =
      quantizedElementType.castExpressedToStorageType(realSparseAttr.getType())
          .dyn_cast_or_null<ShapedType>();
  if (!newSparseType) {
    return nullptr;
  }
  return SparseElementsAttr::get(newSparseType, realSparseAttr.getIndices(),
                                 quantDenseAttr);
}

/// Converts a real expressed Attribute to a corresponding Attribute containing
/// quantized storage values assuming the given uniform quantizedElementType and
/// converter.
Attribute mlir::quant::quantizeAttrUniform(
    Attribute realValue, UniformQuantizedType quantizedElementType,
    const UniformQuantizedValueConverter &converter, Type &outConvertedType) {
  // Fork to handle different variants of constants supported.
  if (realValue.isa<DenseFPElementsAttr>()) {
    // Dense tensor or vector constant.
    auto converted = convertDenseFPElementsAttr(
        realValue.cast<DenseFPElementsAttr>(), quantizedElementType, converter);
    outConvertedType = converted.getType();
    return converted;
  } else if (realValue.isa<SparseElementsAttr>()) {
    // Sparse tensor or vector constant.
    auto converted = convertSparseElementsAttr(
        realValue.cast<SparseElementsAttr>(), quantizedElementType, converter);
    outConvertedType = converted.getType();
    return converted;
  } else {
    // Nothing else matched: try to convert a primitive.
    return convertPrimitiveValueAttr(realValue, quantizedElementType, converter,
                                     outConvertedType);
  }
}

/// Convert an attribute from a type based on
/// quantizedElementType.getExpressedType() to one based on
/// quantizedElementType.getStorageType().
/// Returns nullptr if the conversion is not supported.
/// On success, stores the converted type in outConvertedType.
Attribute mlir::quant::quantizeAttr(Attribute realValue,
                                    QuantizedType quantizedElementType,
                                    Type &outConvertedType) {
  if (auto uniformQuantized =
          quantizedElementType.dyn_cast<UniformQuantizedType>()) {
    UniformQuantizedValueConverter converter(uniformQuantized);
    return quantizeAttrUniform(realValue, uniformQuantized, converter,
                               outConvertedType);

  } else if (auto uniformQuantizedPerAxis =
                 quantizedElementType.dyn_cast<UniformQuantizedPerAxisType>()) {
    UniformQuantizedPerAxisValueConverter converter(uniformQuantizedPerAxis);
    auto converted = converter.convert(realValue);
    // TODO(fengliuai): why we need this outConvertedType? remove it?
    if (converted) {
      outConvertedType = converted.getType();
    }
    return converted;
  } else {
    return nullptr;
  }
}
