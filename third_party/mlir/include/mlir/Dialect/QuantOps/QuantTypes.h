//===- QuantTypes.h - Quantization Ops and Types ----------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_QUANTOPS_QUANT_TYPES_H_
#define MLIR_DIALECT_QUANTOPS_QUANT_TYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace quant {

class QuantizedIntegerType;

namespace detail {

struct QuantizedTypeStorage;
struct AnyQuantizedTypeStorage;
struct UniformQuantizedTypeStorage;
struct UniformQuantizedPerAxisTypeStorage;

} // namespace detail

namespace QuantizationTypes {
enum Kind {
  Any = Type::FIRST_QUANTIZATION_TYPE,
  UniformQuantized,
  UniformQuantizedPerAxis,
  LAST_USED_QUANTIZATION_TYPE = UniformQuantizedPerAxis,
};
} // namespace QuantizationTypes

/// Enumeration of bit-mapped flags related to quantized types.
namespace QuantizationFlags {
enum FlagValue {
  // Indicates that the storage type should be interpreted as a signed
  // integer. The default is to interpret it as an unsigned value.
  Signed = 1,
};
} // namespace QuantizationFlags

/// Base class for all quantized types known to this dialect.
/// All quantized types have:
///   - storageType: The (narrower) numeric type that is being used to
///     approximate some expressed type.
///   - expressedType: The type that is being approximated.
///
/// The base class provides generic support for manipulating the types based
/// on these fields.
class QuantizedType : public Type {
public:
  using ImplType = detail::QuantizedTypeStorage;
  using Type::Type;

  /// The maximum number of bits supported for storage types.
  static constexpr unsigned MaxStorageBits = 32;

  static LogicalResult
  verifyConstructionInvariants(Optional<Location> loc, MLIRContext *context,
                               unsigned flags, Type storageType,
                               Type expressedType, int64_t storageTypeMin,
                               int64_t storageTypeMax);

  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return type.getKind() >= Type::FIRST_QUANTIZATION_TYPE &&
           type.getKind() <= QuantizationTypes::LAST_USED_QUANTIZATION_TYPE;
  }

  /// Gets the minimum possible stored by a storageType. storageTypeMin must
  /// be greater than or equal to this value.
  static int64_t getDefaultMinimumForInteger(bool isSigned,
                                             unsigned integralWidth) {
    if (isSigned) {
      return llvm::minIntN(integralWidth);
    }
    return 0;
  }

  /// Gets the maximum possible stored by a storageType. storageTypeMax must
  /// be less than or equal to this value.
  static int64_t getDefaultMaximumForInteger(bool isSigned,
                                             unsigned integralWidth) {
    if (isSigned) {
      return llvm::maxIntN(integralWidth);
    }
    return llvm::maxUIntN(integralWidth);
  }

  /// Gets the original expressed type that this quantized type approximates.
  /// Note that this presumes that the quantized type was always derived from
  /// a floating point type, which in the broadest definition, is not true (i.e.
  /// it could be some form of integral, fixed type or affine type in its own
  /// right); however, at the high level, no examples of such usage are
  /// presently known and the restriction serves some useful purposes (such as
  /// always being able to reverse a transformation or measure error). In most
  /// cases, this will be f32.
  Type getExpressedType() const;

  /// Gets the flags associated with this type. Typically a more specific
  /// accessor is appropriate.
  unsigned getFlags() const;

  // Convenience helpers.
  /// Whether the storage type should be interpreted as a signed quantity
  /// (true) or an unsigned value (false).
  bool isSigned() const {
    return (getFlags() & QuantizationFlags::Signed) ==
           QuantizationFlags::Signed;
  }

  /// Gets the underlying type used for to store values. Note that this may
  /// be signed or unsigned. Use the isSigned() accessor to differentiate.
  Type getStorageType() const;

  /// The minimum value that storageType can take.
  int64_t getStorageTypeMin() const;

  /// The maximum value that storageType can take.
  int64_t getStorageTypeMax() const;

  /// Gets the integral bit width that the underlying storage type can exactly
  /// represent. For integral storage types, this will just be their width.
  unsigned getStorageTypeIntegralWidth() const;

  /// Returns whether the candidateExpressedType is a match for this
  /// QuantizedType. This will be true if the candidate type is either a
  /// primitive type or a container type whose element type equals this
  /// QuantizedType's expressed type.
  /// Examples of compatible candidateExpressedType:
  ///   !quant.uniform<i8:f32, 1.0> =~ f32
  ///   !quant.uniform<i8:f32, 1.0> =~ tensor<4xf32>
  bool isCompatibleExpressedType(Type candidateExpressedType);

  /// Returns the element type as a QuantizedType or nullptr if it is not
  /// a quantized type. If the type is primitive, returns that. If it is a
  /// container (vector/tensor), return the element type.
  /// Examples:
  ///   !quant.uniform<i8:f32, 1.0> -> !quant.uniform<i8:f32, 1.0>
  ///   tensor<4x!quant.uniform<i8:f32, 1.0> -> quant.uniform<i8:f32, 1.0>
  static QuantizedType getQuantizedElementType(Type primitiveOrContainerType);

  /// Casts from a type based on the storageType to a corresponding type based
  /// on this type (returns nullptr if the cast is not valid).
  /// Examples:
  ///   i8 -> !quant.uniform<i8:f32, 1.0>
  ///   tensor<4xi8> -> tensor<4x!quant.uniform<i8:f32, 1.0}>>
  ///   vector<4xi8> -> vector<4x!quant.uniform<i8:f32, 1.0>>
  Type castFromStorageType(Type candidateType);

  /// Casts from a type based on a QuantizedType to a corresponding type based
  /// on the storageType (returns nullptr if the cast is not valid).
  /// This is the inverse of castFromStorageType().
  static Type castToStorageType(Type quantizedType);

  /// Casts from a type based on the expressedType to a corresponding type based
  /// on this type (returns nullptr if the cast is not valid).
  /// Examples:
  ///   f32 -> !quant.uniform<i8:f32, 1.0>
  ///   tensor<4xf32> -> tensor<4x!quant.uniform<i8:f32, 1.0>>
  ///   vector<4xf32> -> vector<4x!quant.uniform<i8:f32, 1.0>>
  Type castFromExpressedType(Type candidateType);

  /// Casts from a type based on QuantizedType to a corresponding type based
  /// on the expressedType (returns nullptr if the cast is not valid).
  /// This is the inverse of castFromExpressedType.
  static Type castToExpressedType(Type quantizedType);

  /// Casts from a type based on the expressedType to the equivalent type
  /// based on storageType by way of this QuantizedType. Equivalent to:
  ///   QuantizedType::castToStorageType(castFromExpressedType(candidateType))
  /// (but with validity checks).
  /// Example (for this = !quant.uniform<i8:f32, 1.0>):
  ///   tensor<4xf32> -> tensor<4xi8>
  Type castExpressedToStorageType(Type candidateType);

private:
  /// Hide the following methods inherited from `Type`. It is almost certainly
  /// a bug to call them from a `QuantizedType` object. Users should call
  /// `getStorageType` or `getExpressedType` to get the underlying types
  /// they want to inspect.
  using Type::isBF16;
  using Type::isF16;
  using Type::isF32;
  using Type::isF64;
  using Type::isIndex;
  using Type::isInteger;
};

/// A quantized type that maps storage to/from expressed types in an
/// unspecified way.
///
/// Typical syntax:
///   quant.any<i8:f32>
///   quant.any<i8>
///   quant.any<i8<-16,15>>
///
/// Note that for the any type, the expressed type is optional.
class AnyQuantizedType
    : public Type::TypeBase<AnyQuantizedType, QuantizedType,
                            detail::AnyQuantizedTypeStorage> {
public:
  using Base::Base;

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == QuantizationTypes::Any; }

  /// Gets an instance of the type with all parameters specified but not
  /// checked.
  static AnyQuantizedType get(unsigned flags, Type storageType,
                              Type expressedType, int64_t storageTypeMin,
                              int64_t storageTypeMax);

  /// Gets an instance of the type with all specified parameters checked.
  /// Returns a nullptr convertible type on failure.
  static AnyQuantizedType getChecked(unsigned flags, Type storageType,
                                     Type expressedType, int64_t storageTypeMin,
                                     int64_t storageTypeMax, Location location);

  /// Verifies construction invariants and issues errors/warnings.
  static LogicalResult
  verifyConstructionInvariants(Optional<Location> loc, MLIRContext *context,
                               unsigned flags, Type storageType,
                               Type expressedType, int64_t storageTypeMin,
                               int64_t storageTypeMax);
};

/// Represents a family of uniform, quantized types.
///
/// Each instance of this type expresses a mapping between real values (most
/// often expressed in floating point f32) and quantized values (either fixed
/// point or affine).
///
/// The relationship is:
///     real_value = scale * (quantized_value - zero_point)
///
/// It is used as part of high level graph transformations that have the goal
/// of re-expressing parts of a computation in terms of this common form for
/// more efficient execution at runtime. In addition, it is designed to be
/// expressive enough to facilitate lowering to precise types and operations
/// in target hardware.
///
/// As a high-level type, focused on intermediate passes, this type holds
/// opinions consistent with high-level usage. If lowering math kernels below
/// the high level arithmetic ops (i.e. to LLVM IR or hardware specific
/// instruction sets), it is expected that the information expressed here
/// will be used to drive low level codegen and target specific type selection,
/// but this type will likely be erased in the process.
///
/// Syntax synopsis:
///   Per-layer, all parameters expressed:
///     !quant<uniform[StorageType:ExpressedType]{Scale:ZeroPoint}>
///   Per-layer, optional parameters omitted:
///     !quant<uniform[StorageType]{Scale}>
///
///   StorageType: 'i'|'u' NumBits
///   ExpressedType: 'f16', 'f32', 'bf16', 'f64'
///   Scale: A legal double value
///   ZeroPoint: An integer value
class UniformQuantizedType
    : public Type::TypeBase<UniformQuantizedType, QuantizedType,
                            detail::UniformQuantizedTypeStorage> {
public:
  using Base::Base;

  /// Gets an instance of the type with all parameters specified but not
  /// checked.
  static UniformQuantizedType get(unsigned flags, Type storageType,
                                  Type expressedType, double scale,
                                  int64_t zeroPoint, int64_t storageTypeMin,
                                  int64_t storageTypeMax);

  /// Gets an instance of the type with all specified parameters checked.
  /// Returns a nullptr convertible type on failure.
  static UniformQuantizedType
  getChecked(unsigned flags, Type storageType, Type expressedType, double scale,
             int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax,
             Location location);

  /// Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(
      Optional<Location> loc, MLIRContext *context, unsigned flags,
      Type storageType, Type expressedType, double scale, int64_t zeroPoint,
      int64_t storageTypeMin, int64_t storageTypeMax);

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == QuantizationTypes::UniformQuantized;
  }

  /// Gets the scale term. The scale designates the difference between the real
  /// values corresponding to consecutive quantized values differing by 1.
  double getScale() const;

  /// Gets the storage value corresponding to the real value 0 in the affine
  /// equation.
  int64_t getZeroPoint() const;

  // Fixed point values are real numbers divided by a scale.
  // Currently, only signed storage types are treated as fixed point.
  // A fixed point value can be obtained from an affine value by subtracting
  // the zeroPoint.
  // In the future, this may be explicit versus implied by type and zeroPoint.
  bool isFixedPoint() const { return isSigned() && getZeroPoint() == 0; }
};

/// Represents per-axis (also known as per-channel quantization).
///
/// Syntax synopsis:
///   Per-axis, all parameters expressed:
///     !quant<uniform[StorageType:ExpressedType:QuantizedDim]{QuantParams}>
///   Per-axis, optional parameters omitted:
///     !quant<uniform[StorageType]{Scale}>
///
///   StorageType: 'i'|'u' NumBits
///   ExpressedType: 'f16', 'f32', 'bf16', 'f64'
///   QuantizedDim: An integer value
///   QuantParams: (Scale ':' ZeroPoint)+
///   Scale: A legal double value
///   ZeroPoint: An integer value
class UniformQuantizedPerAxisType
    : public Type::TypeBase<UniformQuantizedPerAxisType, QuantizedType,
                            detail::UniformQuantizedPerAxisTypeStorage> {
public:
  using Base::Base;

  /// Gets an instance of the type with all parameters specified but not
  /// checked.
  static UniformQuantizedPerAxisType
  get(unsigned flags, Type storageType, Type expressedType,
      ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
      int32_t quantizedDimension, int64_t storageTypeMin,
      int64_t storageTypeMax);

  /// Gets an instance of the type with all specified parameters checked.
  /// Returns a nullptr convertible type on failure.
  static UniformQuantizedPerAxisType
  getChecked(unsigned flags, Type storageType, Type expressedType,
             ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
             int32_t quantizedDimension, int64_t storageTypeMin,
             int64_t storageTypeMax, Location location);

  /// Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(
      Optional<Location> loc, MLIRContext *context, unsigned flags,
      Type storageType, Type expressedType, ArrayRef<double> scales,
      ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
      int64_t storageTypeMin, int64_t storageTypeMax);

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == QuantizationTypes::UniformQuantizedPerAxis;
  }

  /// Gets the quantization scales. The scales designate the difference between
  /// the real values corresponding to consecutive quantized values differing
  /// by 1. The ith scale corresponds to the ith slice in the
  /// quantized_dimension.
  ArrayRef<double> getScales() const;

  /// Gets the storage values corresponding to the real value 0 in the affine
  /// equation. The ith zero point corresponds to the ith slice in the
  /// quantized_dimension.
  ArrayRef<int64_t> getZeroPoints() const;

  /// Specifies the dimension of the Tensor's shape that the scales and
  /// zero_points correspond to. For example, a tensor t, with dims=[4, 3, 2, 1]
  /// with quantization params:
  ///   scales=[1.0, 2.0, 3.0], zeroPoints=[1, 2, 3], quantizedDimension=1
  /// will be quantized across the second dimension of t.
  ///   t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
  ///   t[:, 1, :, :] will have scale[1]=2.0, zero_point[0]=2
  ///   t[:, 2, :, :] will have scale[2]=3.0, zero_point[0]=3
  int32_t getQuantizedDimension() const;

  /// Fixed point values are real numbers divided by a scale.
  /// Currently, only signed storage types are treated as fixed point.
  /// A fixed point value can be obtained from an affine value by subtracting
  /// the zeroPoint.
  /// In the future, this may be explicit versus implied by type and zeroPoint.
  bool isFixedPoint() const {
    if (!isSigned())
      return false;
    return llvm::all_of(getZeroPoints(),
                        [](int64_t zeroPoint) { return zeroPoint != 0; });
  }
};

} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_QUANT_TYPES_H_
