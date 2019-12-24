//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "TypeDetail.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

unsigned QuantizedType::getFlags() const {
  return static_cast<ImplType *>(impl)->flags;
}

LogicalResult QuantizedType::verifyConstructionInvariants(
    Optional<Location> loc, MLIRContext *context, unsigned flags,
    Type storageType, Type expressedType, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  // Verify that the storage type is integral.
  // This restriction may be lifted at some point in favor of using bf16
  // or f16 as exact representations on hardware where that is advantageous.
  auto intStorageType = storageType.dyn_cast<IntegerType>();
  if (!intStorageType)
    return emitOptionalError(loc, "storage type must be integral");
  unsigned integralWidth = intStorageType.getWidth();

  // Verify storage width.
  if (integralWidth == 0 || integralWidth > MaxStorageBits)
    return emitOptionalError(loc, "illegal storage type size: ", integralWidth);

  // Verify storageTypeMin and storageTypeMax.
  bool isSigned =
      (flags & QuantizationFlags::Signed) == QuantizationFlags::Signed;
  int64_t defaultIntegerMin =
      getDefaultMinimumForInteger(isSigned, integralWidth);
  int64_t defaultIntegerMax =
      getDefaultMaximumForInteger(isSigned, integralWidth);
  if (storageTypeMax - storageTypeMin <= 0 ||
      storageTypeMin < defaultIntegerMin ||
      storageTypeMax > defaultIntegerMax) {
    return emitOptionalError(loc, "illegal storage min and storage max: (",
                             storageTypeMin, ":", storageTypeMax, ")");
  }
  return success();
}

Type QuantizedType::getStorageType() const {
  return static_cast<ImplType *>(impl)->storageType;
}

int64_t QuantizedType::getStorageTypeMin() const {
  return static_cast<ImplType *>(impl)->storageTypeMin;
}

int64_t QuantizedType::getStorageTypeMax() const {
  return static_cast<ImplType *>(impl)->storageTypeMax;
}

unsigned QuantizedType::getStorageTypeIntegralWidth() const {
  // NOTE: If ever supporting non-integral storage types, some other scheme
  // for determining the width will be needed.
  return static_cast<ImplType *>(impl)->storageType.getIntOrFloatBitWidth();
}

Type QuantizedType::getExpressedType() const {
  return static_cast<ImplType *>(impl)->expressedType;
}

bool QuantizedType::isCompatibleExpressedType(Type candidateExpressedType) {
  if (candidateExpressedType.isa<ShapedType>()) {
    return candidateExpressedType.cast<ShapedType>().getElementType() ==
           getExpressedType();
  }
  return candidateExpressedType == getExpressedType();
}

QuantizedType
QuantizedType::getQuantizedElementType(Type primitiveOrContainerType) {
  if (primitiveOrContainerType.isa<ShapedType>()) {
    Type elementType =
        primitiveOrContainerType.cast<ShapedType>().getElementType();
    return elementType.dyn_cast<QuantizedType>();
  }
  return primitiveOrContainerType.dyn_cast<QuantizedType>();
}

Type QuantizedType::castFromStorageType(Type candidateType) {
  if (candidateType == getStorageType()) {
    // i.e. i32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  } else if (candidateType.isa<RankedTensorType>()) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return RankedTensorType::get(
        candidateType.cast<RankedTensorType>().getShape(), getStorageType());
  } else if (candidateType.isa<UnrankedTensorType>()) {
    // i.e. tensor<i8> -> tensor<!quant<"uniform[i8:f32]{1.0}">>
    return UnrankedTensorType::get(getStorageType());
  } else if (candidateType.isa<VectorType>()) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return VectorType::get(candidateType.cast<VectorType>().getShape(),
                           getStorageType());
  }

  return nullptr;
}

Type QuantizedType::castToStorageType(Type quantizedType) {
  if (quantizedType.isa<QuantizedType>()) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> i8
    return quantizedType.cast<QuantizedType>().getStorageType();
  } else if (quantizedType.isa<ShapedType>()) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = quantizedType.cast<ShapedType>();
    if (!sType.getElementType().isa<QuantizedType>()) {
      return nullptr;
    }
    Type storageType =
        sType.getElementType().cast<QuantizedType>().getStorageType();
    if (quantizedType.isa<RankedTensorType>()) {
      return RankedTensorType::get(sType.getShape(), storageType);
    } else if (quantizedType.isa<UnrankedTensorType>()) {
      return UnrankedTensorType::get(storageType);
    } else if (quantizedType.isa<VectorType>()) {
      return VectorType::get(sType.getShape(), storageType);
    }
  }

  return nullptr;
}

Type QuantizedType::castFromExpressedType(Type candidateType) {
  if (candidateType == getExpressedType()) {
    // i.e. f32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  } else if (candidateType.isa<ShapedType>()) {
    ShapedType candidateShapedType = candidateType.cast<ShapedType>();
    if (candidateShapedType.getElementType() != getExpressedType()) {
      return nullptr;
    }

    if (candidateType.isa<RankedTensorType>()) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return RankedTensorType::get(candidateShapedType.getShape(), *this);
    } else if (candidateType.isa<UnrankedTensorType>()) {
      // i.e. tensor<xf32> -> tensor<x!quant<"uniform[i8:f32]{1.0}">>
      return UnrankedTensorType::get(*this);
    } else if (candidateType.isa<VectorType>()) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return VectorType::get(candidateShapedType.getShape(), *this);
    }
  }

  return nullptr;
}

Type QuantizedType::castToExpressedType(Type quantizedType) {
  if (quantizedType.isa<QuantizedType>()) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> f32
    return quantizedType.cast<QuantizedType>().getExpressedType();
  } else if (quantizedType.isa<ShapedType>()) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = quantizedType.cast<ShapedType>();
    if (!sType.getElementType().isa<QuantizedType>()) {
      return nullptr;
    }
    Type expressedType =
        sType.getElementType().cast<QuantizedType>().getExpressedType();
    if (quantizedType.isa<RankedTensorType>()) {
      return RankedTensorType::get(sType.getShape(), expressedType);
    } else if (quantizedType.isa<UnrankedTensorType>()) {
      return UnrankedTensorType::get(expressedType);
    } else if (quantizedType.isa<VectorType>()) {
      return VectorType::get(sType.getShape(), expressedType);
    }
  }

  return nullptr;
}

Type QuantizedType::castExpressedToStorageType(Type candidateType) {
  Type expressedQuantizedType = castFromExpressedType(candidateType);
  if (!expressedQuantizedType) {
    return nullptr;
  }
  return QuantizedType::castToStorageType(expressedQuantizedType);
}

AnyQuantizedType AnyQuantizedType::get(unsigned flags, Type storageType,
                                       Type expressedType,
                                       int64_t storageTypeMin,
                                       int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), QuantizationTypes::Any, flags,
                   storageType, expressedType, storageTypeMin, storageTypeMax);
}

AnyQuantizedType AnyQuantizedType::getChecked(unsigned flags, Type storageType,
                                              Type expressedType,
                                              int64_t storageTypeMin,
                                              int64_t storageTypeMax,
                                              Location location) {
  return Base::getChecked(location, storageType.getContext(),
                          QuantizationTypes::Any, flags, storageType,
                          expressedType, storageTypeMin, storageTypeMax);
}

LogicalResult AnyQuantizedType::verifyConstructionInvariants(
    Optional<Location> loc, MLIRContext *context, unsigned flags,
    Type storageType, Type expressedType, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyConstructionInvariants(
          loc, context, flags, storageType, expressedType, storageTypeMin,
          storageTypeMax))) {
    return failure();
  }

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (expressedType && !expressedType.isa<FloatType>())
    return emitOptionalError(loc, "expressed type must be floating point");

  return success();
}

UniformQuantizedType UniformQuantizedType::get(unsigned flags, Type storageType,
                                               Type expressedType, double scale,
                                               int64_t zeroPoint,
                                               int64_t storageTypeMin,
                                               int64_t storageTypeMax) {
  return Base::get(storageType.getContext(),
                   QuantizationTypes::UniformQuantized, flags, storageType,
                   expressedType, scale, zeroPoint, storageTypeMin,
                   storageTypeMax);
}

UniformQuantizedType
UniformQuantizedType::getChecked(unsigned flags, Type storageType,
                                 Type expressedType, double scale,
                                 int64_t zeroPoint, int64_t storageTypeMin,
                                 int64_t storageTypeMax, Location location) {
  return Base::getChecked(location, storageType.getContext(),
                          QuantizationTypes::UniformQuantized, flags,
                          storageType, expressedType, scale, zeroPoint,
                          storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedType::verifyConstructionInvariants(
    Optional<Location> loc, MLIRContext *context, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyConstructionInvariants(
          loc, context, flags, storageType, expressedType, storageTypeMin,
          storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitOptionalError(loc,
                             "uniform quantization requires expressed type");

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!expressedType.isa<FloatType>())
    return emitOptionalError(loc, "expressed type must be floating point");

  // Verify scale.
  if (scale <= 0.0 || std::isinf(scale) || std::isnan(scale))
    return emitOptionalError(loc, "illegal scale: ", scale);

  return success();
}

double UniformQuantizedType::getScale() const { return getImpl()->scale; }

int64_t UniformQuantizedType::getZeroPoint() const {
  return getImpl()->zeroPoint;
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::get(
    unsigned flags, Type storageType, Type expressedType,
    ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return Base::get(storageType.getContext(),
                   QuantizationTypes::UniformQuantizedPerAxis, flags,
                   storageType, expressedType, scales, zeroPoints,
                   quantizedDimension, storageTypeMin, storageTypeMax);
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::getChecked(
    unsigned flags, Type storageType, Type expressedType,
    ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin, int64_t storageTypeMax,
    Location location) {
  return Base::getChecked(location, storageType.getContext(),
                          QuantizationTypes::UniformQuantizedPerAxis, flags,
                          storageType, expressedType, scales, zeroPoints,
                          quantizedDimension, storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedPerAxisType::verifyConstructionInvariants(
    Optional<Location> loc, MLIRContext *context, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyConstructionInvariants(
          loc, context, flags, storageType, expressedType, storageTypeMin,
          storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitOptionalError(loc,
                             "uniform quantization requires expressed type");

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!expressedType.isa<FloatType>())
    return emitOptionalError(loc, "expressed type must be floating point");

  // Ensure that the number of scales and zeroPoints match.
  if (scales.size() != zeroPoints.size())
    return emitOptionalError(loc, "illegal number of scales and zeroPoints: ",
                             scales.size(), ", ", zeroPoints.size());

  // Verify scale.
  for (double scale : scales) {
    if (scale <= 0.0 || std::isinf(scale) || std::isnan(scale))
      return emitOptionalError(loc, "illegal scale: ", scale);
  }

  return success();
}

ArrayRef<double> UniformQuantizedPerAxisType::getScales() const {
  return getImpl()->getScales();
}

ArrayRef<int64_t> UniformQuantizedPerAxisType::getZeroPoints() const {
  return getImpl()->getZeroPoints();
}

int32_t UniformQuantizedPerAxisType::getQuantizedDimension() const {
  return getImpl()->quantizedDimension;
}
