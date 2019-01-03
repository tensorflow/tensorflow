//===- StandardTypes.h - MLIR Standard Type Classes -------------*- C++ -*-===//
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

#ifndef MLIR_IR_STANDARDTYPES_H
#define MLIR_IR_STANDARDTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace llvm {
class fltSemantics;
} // namespace llvm

namespace mlir {
class AffineMap;
class FloatType;
class IndexType;
class IntegerType;
class Location;
class MLIRContext;

namespace detail {

struct IntegerTypeStorage;
struct VectorOrTensorTypeStorage;
struct VectorTypeStorage;
struct TensorTypeStorage;
struct RankedTensorTypeStorage;
struct UnrankedTensorTypeStorage;
struct MemRefTypeStorage;

} // namespace detail

namespace StandardTypes {
enum Kind {
  // Floating point.
  BF16 = Type::Kind::FIRST_STANDARD_TYPE,
  F16,
  F32,
  F64,
  FIRST_FLOATING_POINT_TYPE = BF16,
  LAST_FLOATING_POINT_TYPE = F64,

  // Derived types.
  Integer,
  Vector,
  RankedTensor,
  UnrankedTensor,
  MemRef,
};

} // namespace StandardTypes

inline bool Type::isBF16() const { return getKind() == StandardTypes::BF16; }
inline bool Type::isF16() const { return getKind() == StandardTypes::F16; }
inline bool Type::isF32() const { return getKind() == StandardTypes::F32; }
inline bool Type::isF64() const { return getKind() == StandardTypes::F64; }

/// Integer types can have arbitrary bitwidth up to a large fixed limit.
class IntegerType : public Type {
public:
  using ImplType = detail::IntegerTypeStorage;
  using Type::Type;

  /// Get or create a new IntegerType of the given width within the context.
  /// Assume the width is within the allowed range and assert on failures.
  /// Use getChecked to handle failures gracefully.
  static IntegerType get(unsigned width, MLIRContext *context);

  /// Get or create a new IntegerType of the given width within the context,
  /// defined at the given, potentially unknown, location.  If the width is
  /// outside the allowed range, emit errors and return a null type.
  static IntegerType getChecked(unsigned width, MLIRContext *context,
                                Location location);

  /// Return the bitwidth of this integer type.
  unsigned getWidth() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardTypes::Integer; }

  /// Unique identifier for this type class.
  static char typeID;

  /// Integer representation maximal bitwidth.
  static constexpr unsigned kMaxWidth = 4096;
};

inline IntegerType Type::getInteger(unsigned width, MLIRContext *ctx) {
  return IntegerType::get(width, ctx);
}

/// Return true if this is an integer type with the specified width.
inline bool Type::isInteger(unsigned width) const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.getWidth() == width;
  return false;
}

inline bool Type::isIntOrIndex() const {
  return isa<IndexType>() || isa<IntegerType>();
}

inline bool Type::isIntOrIndexOrFloat() const {
  return isa<IndexType>() || isa<IntegerType>() || isa<FloatType>();
}

inline bool Type::isIntOrFloat() const {
  return isa<IntegerType>() || isa<FloatType>();
}

class FloatType : public Type {
public:
  using Type::Type;

  static FloatType get(StandardTypes::Kind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind >= StandardTypes::FIRST_FLOATING_POINT_TYPE &&
           kind <= StandardTypes::LAST_FLOATING_POINT_TYPE;
  }

  /// Return the bitwidth of this float type.
  unsigned getWidth() const;

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics() const;

  /// Unique identifier for this type class.
  static char typeID;
};

inline FloatType Type::getBF16(MLIRContext *ctx) {
  return FloatType::get(StandardTypes::BF16, ctx);
}
inline FloatType Type::getF16(MLIRContext *ctx) {
  return FloatType::get(StandardTypes::F16, ctx);
}
inline FloatType Type::getF32(MLIRContext *ctx) {
  return FloatType::get(StandardTypes::F32, ctx);
}
inline FloatType Type::getF64(MLIRContext *ctx) {
  return FloatType::get(StandardTypes::F64, ctx);
}

/// This is a common base class between Vector, UnrankedTensor, and RankedTensor
/// types, because many operations work on values of these aggregate types.
class VectorOrTensorType : public Type {
public:
  using ImplType = detail::VectorOrTensorTypeStorage;
  using Type::Type;

  /// Return the element type.
  Type getElementType() const;

  /// If an element type is an integer or a float, return its width.  Abort
  /// otherwise.
  unsigned getElementTypeBitWidth() const;

  /// If this is ranked tensor or vector type, return the number of elements. If
  /// it is an unranked tensor, abort.
  unsigned getNumElements() const;

  /// If this is ranked tensor or vector type, return the rank. If it is an
  /// unranked tensor, return -1.
  int getRank() const;

  /// If this is ranked tensor or vector type, return the shape. If it is an
  /// unranked tensor, abort.
  ArrayRef<int> getShape() const;

  /// If this is unranked tensor or any dimension has unknown size (<0),
  /// it doesn't have static shape. If all dimensions have known size (>= 0),
  /// it has static shape.
  bool hasStaticShape() const;

  /// If this is ranked tensor or vector type, return the size of the specified
  /// dimension. It aborts if the tensor is unranked (this can be checked by
  /// the getRank call method).
  int getDimSize(unsigned i) const;

  /// Get the total amount of bits occupied by a value of this type.  This does
  /// not take into account any memory layout or widening constraints, e.g. a
  /// vector<3xi57> is reported to occupy 3x57=171 bit, even though in practice
  /// it will likely be stored as in a 4xi64 vector register.  Fail an assertion
  /// if the size cannot be computed statically, i.e. if the tensor has a
  /// dynamic shape or if its elemental type does not have a known bit width.
  long getSizeInBits() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardTypes::Vector ||
           kind == StandardTypes::RankedTensor ||
           kind == StandardTypes::UnrankedTensor;
  }
};

/// Vector types represent multi-dimensional SIMD vectors, and have a fixed
/// known constant shape with one or more dimension.
class VectorType : public VectorOrTensorType {
public:
  using ImplType = detail::VectorTypeStorage;
  using VectorOrTensorType::VectorOrTensorType;

  /// Get or create a new VectorType of the provided shape and element type.
  /// Assumes the arguments define a well-formed VectorType.
  static VectorType get(ArrayRef<int> shape, Type elementType);

  /// Get or create a new VectorType of the provided shape and element type
  /// declared at the given, potentially unknown, location.  If the VectorType
  /// defined by the arguments would be ill-formed, emit errors and return
  /// nullptr-wrapping type.
  static VectorType getChecked(ArrayRef<int> shape, Type elementType,
                               Location location);

  /// Returns true of the given type can be used as an element of a vector type.
  /// In particular, vectors can consist of integer or float primitives.
  static bool isValidElementType(Type t) { return t.isIntOrFloat(); }

  ArrayRef<int> getShape() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardTypes::Vector; }

  /// Unique identifier for this type class.
  static char typeID;
};

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
class TensorType : public VectorOrTensorType {
public:
  using ImplType = detail::TensorTypeStorage;
  using VectorOrTensorType::VectorOrTensorType;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type) {
    // Note: Non standard/builtin types are allowed to exist within tensor
    // types. Dialects are expected to verify that tensor types have a valid
    // element type within that dialect.
    return type.isIntOrFloat() || type.isa<VectorType>() ||
           (type.getKind() >= Type::Kind::LAST_STANDARD_TYPE);
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardTypes::RankedTensor ||
           kind == StandardTypes::UnrankedTensor;
  }
};

/// Ranked tensor types represent multi-dimensional arrays that have a shape
/// with a fixed number of dimensions. Each shape element can be a positive
/// integer or unknown (represented -1).
class RankedTensorType : public TensorType {
public:
  using ImplType = detail::RankedTensorTypeStorage;
  using TensorType::TensorType;

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static RankedTensorType get(ArrayRef<int> shape, Type elementType);

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// RankedTensorType defined by the arguments would be ill-formed, emit errors
  /// and return a nullptr-wrapping type.
  static RankedTensorType getChecked(ArrayRef<int> shape, Type elementType,
                                     Location location);

  ArrayRef<int> getShape() const;

  static bool kindof(unsigned kind) {
    return kind == StandardTypes::RankedTensor;
  }

  /// Unique identifier for this type class.
  static char typeID;
};

/// Unranked tensor types represent multi-dimensional arrays that have an
/// unknown shape.
class UnrankedTensorType : public TensorType {
public:
  using ImplType = detail::UnrankedTensorTypeStorage;
  using TensorType::TensorType;

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static UnrankedTensorType get(Type elementType);

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// UnrankedTensorType defined by the arguments would be ill-formed, emit
  /// errors and return a nullptr-wrapping type.
  static UnrankedTensorType getChecked(Type elementType, Location location);

  ArrayRef<int> getShape() const { return ArrayRef<int>(); }

  static bool kindof(unsigned kind) {
    return kind == StandardTypes::UnrankedTensor;
  }

  /// Unique identifier for this type class.
  static char typeID;
};

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a positive integer or
/// unknown (represented by any negative integer). MemRef types also have an
/// affine map composition, represented as an array AffineMap pointers.
class MemRefType : public Type {
public:
  using ImplType = detail::MemRefTypeStorage;
  using Type::Type;

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space.  Assumes the arguments define a
  /// well-formed MemRef type.  Use getChecked to gracefully handle MemRefType
  /// construction failures.
  static MemRefType get(ArrayRef<int> shape, Type elementType,
                        ArrayRef<AffineMap> affineMapComposition,
                        unsigned memorySpace);

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space declared at the given location.
  /// If the location is unknown, the last argument should be an instance of
  /// UnknownLoc.  If the MemRefType defined by the arguments would be
  /// ill-formed, emits errors (to the handler registered with the context or to
  /// the error stream) and returns nullptr.
  static MemRefType getChecked(ArrayRef<int> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace, Location location);

  unsigned getRank() const { return getShape().size(); }

  /// Returns an array of memref shape dimension sizes.
  ArrayRef<int> getShape() const;

  /// Return the size of the specified dimension, or -1 if unspecified.
  int getDimSize(unsigned i) const { return getShape()[i]; }

  /// Returns the elemental type for this memref shape.
  Type getElementType() const;

  /// Returns an array of affine map pointers representing the memref affine
  /// map composition.
  ArrayRef<AffineMap> getAffineMaps() const;

  /// Returns the memory space in which data referred to by this memref resides.
  unsigned getMemorySpace() const;

  /// Returns the number of dimensions with dynamic size.
  unsigned getNumDynamicDims() const;

  static bool kindof(unsigned kind) { return kind == StandardTypes::MemRef; }

  /// Unique identifier for this type class.
  static char typeID;

private:
  static MemRefType getSafe(ArrayRef<int> shape, Type elementType,
                            ArrayRef<AffineMap> affineMapComposition,
                            unsigned memorySpace, Optional<Location> location);
};

} // end namespace mlir

#endif // MLIR_IR_STANDARDTYPES_H
