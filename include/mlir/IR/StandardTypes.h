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

namespace llvm {
struct fltSemantics;
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
struct ShapedTypeStorage;
struct VectorTypeStorage;
struct RankedTensorTypeStorage;
struct UnrankedTensorTypeStorage;
struct MemRefTypeStorage;
struct ComplexTypeStorage;
struct TupleTypeStorage;

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

  // Target pointer sized integer, used (e.g.) in affine mappings.
  Index,

  // Derived types.
  Integer,
  Vector,
  RankedTensor,
  UnrankedTensor,
  MemRef,
  Complex,
  Tuple,
  None,
};

} // namespace StandardTypes

inline bool Type::isBF16() { return getKind() == StandardTypes::BF16; }
inline bool Type::isF16() { return getKind() == StandardTypes::F16; }
inline bool Type::isF32() { return getKind() == StandardTypes::F32; }
inline bool Type::isF64() { return getKind() == StandardTypes::F64; }

inline bool Type::isIndex() { return getKind() == StandardTypes::Index; }

/// Index is a special integer-like type with unknown platform-dependent bit
/// width.
class IndexType : public Type::TypeBase<IndexType, Type> {
public:
  using Base::Base;

  /// Crete an IndexType instance, unique in the given context.
  static IndexType get(MLIRContext *context) {
    return Base::get(context, StandardTypes::Index);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == StandardTypes::Index; }
};

/// Integer types can have arbitrary bitwidth up to a large fixed limit.
class IntegerType
    : public Type::TypeBase<IntegerType, Type, detail::IntegerTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new IntegerType of the given width within the context.
  /// Assume the width is within the allowed range and assert on failures.
  /// Use getChecked to handle failures gracefully.
  static IntegerType get(unsigned width, MLIRContext *context);

  /// Get or create a new IntegerType of the given width within the context,
  /// defined at the given, potentially unknown, location.  If the width is
  /// outside the allowed range, emit errors and return a null type.
  static IntegerType getChecked(unsigned width, MLIRContext *context,
                                Location location);

  /// Verify the construction of an integer type.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, unsigned width);

  /// Return the bitwidth of this integer type.
  unsigned getWidth() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardTypes::Integer; }

  /// Integer representation maximal bitwidth.
  static constexpr unsigned kMaxWidth = 4096;
};

/// Return true if this is an integer type with the specified width.
inline bool Type::isInteger(unsigned width) {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.getWidth() == width;
  return false;
}

inline bool Type::isIntOrIndex() {
  return isa<IndexType>() || isa<IntegerType>();
}

inline bool Type::isIntOrIndexOrFloat() {
  return isa<IndexType>() || isa<IntegerType>() || isa<FloatType>();
}

inline bool Type::isIntOrFloat() {
  return isa<IntegerType>() || isa<FloatType>();
}

class FloatType : public Type::TypeBase<FloatType, Type> {
public:
  using Base::Base;

  static FloatType get(StandardTypes::Kind kind, MLIRContext *context) {
    assert(kindof(kind) && "Not a FP kind.");
    return Base::get(context, kind);
  }

  // Convenience factories.
  static FloatType getBF16(MLIRContext *ctx) {
    return get(StandardTypes::BF16, ctx);
  }
  static FloatType getF16(MLIRContext *ctx) {
    return get(StandardTypes::F16, ctx);
  }
  static FloatType getF32(MLIRContext *ctx) {
    return get(StandardTypes::F32, ctx);
  }
  static FloatType getF64(MLIRContext *ctx) {
    return get(StandardTypes::F64, ctx);
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind >= StandardTypes::FIRST_FLOATING_POINT_TYPE &&
           kind <= StandardTypes::LAST_FLOATING_POINT_TYPE;
  }

  /// Return the bitwidth of this float type.
  unsigned getWidth();

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics();
};

// TODO(b/132735995) Add support for MemRef
/// This is a common base class between Vector, UnrankedTensor, and RankedTensor
/// types because they share behavior and semantics around shape, rank, and
/// fixed element type.
class ShapedType : public Type {
public:
  using ImplType = detail::ShapedTypeStorage;
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
  int64_t getRank() const;

  /// If this is ranked tensor or vector type, return the shape. If it is an
  /// unranked tensor, abort.
  ArrayRef<int64_t> getShape() const;

  /// If this is unranked tensor or any dimension has unknown size (<0),
  /// it doesn't have static shape. If all dimensions have known size (>= 0),
  /// it has static shape.
  bool hasStaticShape() const;

  /// If this is ranked tensor or vector type, return the size of the specified
  /// dimension. It aborts if the tensor is unranked (this can be checked by
  /// the getRank call method).
  int64_t getDimSize(unsigned i) const;

  /// Get the total amount of bits occupied by a value of this type.  This does
  /// not take into account any memory layout or widening constraints, e.g. a
  /// vector<3xi57> is reported to occupy 3x57=171 bit, even though in practice
  /// it will likely be stored as in a 4xi64 vector register.  Fail an assertion
  /// if the size cannot be computed statically, i.e. if the tensor has a
  /// dynamic shape or if its elemental type does not have a known bit width.
  int64_t getSizeInBits() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardTypes::Vector ||
           kind == StandardTypes::RankedTensor ||
           kind == StandardTypes::UnrankedTensor;
  }
};

/// Vector types represent multi-dimensional SIMD vectors, and have a fixed
/// known constant shape with one or more dimension.
class VectorType
    : public Type::TypeBase<VectorType, ShapedType, detail::VectorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new VectorType of the provided shape and element type.
  /// Assumes the arguments define a well-formed VectorType.
  static VectorType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new VectorType of the provided shape and element type
  /// declared at the given, potentially unknown, location.  If the VectorType
  /// defined by the arguments would be ill-formed, emit errors and return
  /// nullptr-wrapping type.
  static VectorType getChecked(ArrayRef<int64_t> shape, Type elementType,
                               Location location);

  /// Verify the construction of a vector type.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, ArrayRef<int64_t> shape,
                               Type elementType);

  /// Returns true of the given type can be used as an element of a vector type.
  /// In particular, vectors can consist of integer or float primitives.
  static bool isValidElementType(Type t) { return t.isIntOrFloat(); }

  ArrayRef<int64_t> getShape() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardTypes::Vector; }
};

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
class TensorType : public ShapedType {
public:
  using ShapedType::ShapedType;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type) {
    // Note: Non standard/builtin types are allowed to exist within tensor
    // types. Dialects are expected to verify that tensor types have a valid
    // element type within that dialect.
    return type.isIntOrFloat() || type.isa<VectorType>() ||
           type.isa<OpaqueType>() ||
           (type.getKind() > Type::Kind::LAST_STANDARD_TYPE);
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
class RankedTensorType
    : public Type::TypeBase<RankedTensorType, TensorType,
                            detail::RankedTensorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new RankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// RankedTensorType defined by the arguments would be ill-formed, emit errors
  /// and return a nullptr-wrapping type.
  static RankedTensorType getChecked(ArrayRef<int64_t> shape, Type elementType,
                                     Location location);

  /// Verify the construction of a ranked tensor type.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, ArrayRef<int64_t> shape,
                               Type elementType);

  ArrayRef<int64_t> getShape() const;

  static bool kindof(unsigned kind) {
    return kind == StandardTypes::RankedTensor;
  }
};

/// Unranked tensor types represent multi-dimensional arrays that have an
/// unknown shape.
class UnrankedTensorType
    : public Type::TypeBase<UnrankedTensorType, TensorType,
                            detail::UnrankedTensorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static UnrankedTensorType get(Type elementType);

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// UnrankedTensorType defined by the arguments would be ill-formed, emit
  /// errors and return a nullptr-wrapping type.
  static UnrankedTensorType getChecked(Type elementType, Location location);

  /// Verify the construction of a unranked tensor type.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, Type elementType);

  ArrayRef<int64_t> getShape() const { return llvm::None; }

  static bool kindof(unsigned kind) {
    return kind == StandardTypes::UnrankedTensor;
  }
};

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a non-negative integer or
/// unknown (represented by any negative integer). MemRef types also have an
/// affine map composition, represented as an array AffineMap pointers.
class MemRefType
    : public Type::TypeBase<MemRefType, Type, detail::MemRefTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space.  Assumes the arguments define a
  /// well-formed MemRef type.  Use getChecked to gracefully handle MemRefType
  /// construction failures.
  static MemRefType get(ArrayRef<int64_t> shape, Type elementType,
                        ArrayRef<AffineMap> affineMapComposition = {},
                        unsigned memorySpace = 0) {
    auto result = getImpl(shape, elementType, affineMapComposition, memorySpace,
                          /*location=*/llvm::None);
    assert(result && "Failed to construct instance of MemRefType.");
    return result;
  }

  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space declared at the given location.
  /// If the location is unknown, the last argument should be an instance of
  /// UnknownLoc.  If the MemRefType defined by the arguments would be
  /// ill-formed, emits errors (to the handler registered with the context or to
  /// the error stream) and returns nullptr.
  static MemRefType getChecked(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace, Location location) {
    return getImpl(shape, elementType, affineMapComposition, memorySpace,
                   location);
  }

  unsigned getRank() const { return getShape().size(); }

  /// Returns an array of memref shape dimension sizes.
  ArrayRef<int64_t> getShape() const;

  /// Return the size of the specified dimension, or -1 if unspecified.
  int64_t getDimSize(unsigned i) const { return getShape()[i]; }

  /// Returns the elemental type for this memref shape.
  Type getElementType() const;

  /// Returns an array of affine map pointers representing the memref affine
  /// map composition.
  ArrayRef<AffineMap> getAffineMaps() const;

  /// Returns the memory space in which data referred to by this memref resides.
  unsigned getMemorySpace() const;

  /// Returns the number of dimensions with dynamic size.
  unsigned getNumDynamicDims() const;

  /// If any dimension of the shape has unknown size (<0), it doesn't have
  /// static shape.
  bool hasStaticShape() const { return getNumDynamicDims() == 0; }

  static bool kindof(unsigned kind) { return kind == StandardTypes::MemRef; }

  /// Integer value indicating that the size in a dimension is dynamic.
  static constexpr int64_t kDynamicDimSize = -1;

private:
  /// Get or create a new MemRefType defined by the arguments.  If the resulting
  /// type would be ill-formed, return nullptr.  If the location is provided,
  /// emit detailed error messages.
  static MemRefType getImpl(ArrayRef<int64_t> shape, Type elementType,
                            ArrayRef<AffineMap> affineMapComposition,
                            unsigned memorySpace, Optional<Location> location);
  using Base::getImpl;
};

/// The 'complex' type represents a complex number with a parameterized element
/// type, which is composed of a real and imaginary value of that element type.
///
/// The element must be a floating point or integer scalar type.
///
class ComplexType
    : public Type::TypeBase<ComplexType, Type, detail::ComplexTypeStorage> {
public:
  using Base::Base;

  /// Get or create a ComplexType with the provided element type.
  static ComplexType get(Type elementType);

  /// Get or create a ComplexType with the provided element type.  This emits
  /// and error at the specified location and returns null if the element type
  /// isn't supported.
  static ComplexType getChecked(Type elementType, Location location);

  /// Verify the construction of an integer type.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, Type elementType);

  Type getElementType();

  static bool kindof(unsigned kind) { return kind == StandardTypes::Complex; }

private:
  static ComplexType getCheckedImpl(Type elementType,
                                    Optional<Location> location);
};

/// Tuple types represent a collection of other types. Note: This type merely
/// provides a common mechanism for representing tuples in MLIR. It is up to
/// dialect authors to provides operations for manipulating them, e.g.
/// extract_tuple_element. When possible, users should prefer multi-result
/// operations in the place of tuples.
class TupleType
    : public Type::TypeBase<TupleType, Type, detail::TupleTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new TupleType with the provided element types. Assumes the
  /// arguments define a well-formed type.
  static TupleType get(ArrayRef<Type> elementTypes, MLIRContext *context);

  /// Get or create an empty tuple type.
  static TupleType get(MLIRContext *context) { return get({}, context); }

  /// Return the elements types for this tuple.
  ArrayRef<Type> getTypes() const;

  /// Return the number of held types.
  unsigned size() const;

  /// Iterate over the held elements.
  using iterator = ArrayRef<Type>::iterator;
  iterator begin() const { return getTypes().begin(); }
  iterator end() const { return getTypes().end(); }

  /// Return the element type at index 'index'.
  Type getType(unsigned index) const {
    assert(index < size() && "invalid index for tuple type");
    return getTypes()[index];
  }

  static bool kindof(unsigned kind) { return kind == StandardTypes::Tuple; }
};

/// NoneType is a unit type, i.e. a type with exactly one possible value, where
/// its value does not have a defined dynamic representation.
class NoneType : public Type::TypeBase<NoneType, Type> {
public:
  using Base::Base;

  /// Get or create a NoneType instance.
  static NoneType get(MLIRContext *context) {
    return Base::get(context, StandardTypes::None);
  }

  static bool kindof(unsigned kind) { return kind == StandardTypes::None; }
};
} // end namespace mlir

#endif // MLIR_IR_STANDARDTYPES_H
