//===- Types.h - MLIR Type Classes ------------------------------*- C++ -*-===//
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

#ifndef MLIR_IR_TYPES_H
#define MLIR_IR_TYPES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {
class AffineMap;
class FloatType;
class IndexType;
class IntegerType;
class Location;
class MLIRContext;
class OtherType;

namespace detail {

class TypeStorage;
class IndexTypeStorage;
class IntegerTypeStorage;
class FloatTypeStorage;
struct OtherTypeStorage;
struct FunctionTypeStorage;
struct VectorOrTensorTypeStorage;
struct VectorTypeStorage;
struct TensorTypeStorage;
struct RankedTensorTypeStorage;
struct UnrankedTensorTypeStorage;
struct MemRefTypeStorage;

} // namespace detail

/// Instances of the Type class are immutable, uniqued, immortal, and owned by
/// MLIRContext.  As such, they are passed around by raw non-const pointer.
///
class Type {
public:
  /// Integer identifier for all the concrete type kinds.
  enum class Kind {
    // Target pointer sized integer, used (e.g.) in affine mappings.
    Index,

    // TensorFlow types.
    TFControl,
    TFResource,
    TFVariant,
    TFComplex64,
    TFComplex128,
    TFF32REF,
    TFString,

    /// These are marker for the first and last 'other' type.
    FIRST_OTHER_TYPE = TFControl,
    LAST_OTHER_TYPE = TFString,

    // Floating point.
    BF16,
    F16,
    F32,
    F64,
    FIRST_FLOATING_POINT_TYPE = BF16,
    LAST_FLOATING_POINT_TYPE = F64,

    // Derived types.
    Integer,
    Function,
    Vector,
    RankedTensor,
    UnrankedTensor,
    MemRef,
  };

  using ImplType = detail::TypeStorage;

  Type() : type(nullptr) {}
  /* implicit */ Type(const ImplType *type)
      : type(const_cast<ImplType *>(type)) {}

  Type(const Type &other) : type(other.type) {}
  Type &operator=(Type other) {
    type = other.type;
    return *this;
  }

  bool operator==(Type other) const { return type == other.type; }
  bool operator!=(Type other) const { return !(*this == other); }
  explicit operator bool() const { return type; }

  bool operator!() const { return type == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  /// Return the classification for this type.
  Kind getKind() const;

  /// Return the LLVMContext in which this type was uniqued.
  MLIRContext *getContext() const;

  // Convenience predicates.  This is only for 'other' and floating point types,
  // derived types should use isa/dyn_cast.
  bool isIndex() const { return getKind() == Kind::Index; }
  bool isTFControl() const { return getKind() == Kind::TFControl; }
  bool isTFResource() const { return getKind() == Kind::TFResource; }
  bool isTFVariant() const { return getKind() == Kind::TFVariant; }
  bool isTFComplex64() const { return getKind() == Kind::TFComplex64; }
  bool isTFComplex128() const { return getKind() == Kind::TFComplex128; }
  bool isTFF32REF() const { return getKind() == Kind::TFF32REF; }
  bool isTFString() const { return getKind() == Kind::TFString; }
  bool isBF16() const { return getKind() == Kind::BF16; }
  bool isF16() const { return getKind() == Kind::F16; }
  bool isF32() const { return getKind() == Kind::F32; }
  bool isF64() const { return getKind() == Kind::F64; }

  /// Return true if this is an integer type with the specified width.
  bool isInteger(unsigned width) const;

  /// Return the bit width of an integer or a float type, assert failure on
  /// other types.
  unsigned getIntOrFloatBitWidth() const;

  /// Return true if this is an integer or index type.
  bool isIntOrIndex() const;
  /// Return true if this is an integer, index, or float type.
  bool isIntOrIndexOrFloat() const;
  /// Return true of this is an integer or a float type.
  bool isIntOrFloat() const;

  // Convenience factories.
  static IndexType getIndex(MLIRContext *ctx);
  static IntegerType getInteger(unsigned width, MLIRContext *ctx);
  static FloatType getBF16(MLIRContext *ctx);
  static FloatType getF16(MLIRContext *ctx);
  static FloatType getF32(MLIRContext *ctx);
  static FloatType getF64(MLIRContext *ctx);
  static OtherType getTFControl(MLIRContext *ctx);
  static OtherType getTFString(MLIRContext *ctx);
  static OtherType getTFResource(MLIRContext *ctx);
  static OtherType getTFVariant(MLIRContext *ctx);
  static OtherType getTFComplex64(MLIRContext *ctx);
  static OtherType getTFComplex128(MLIRContext *ctx);
  static OtherType getTFF32REF(MLIRContext *ctx);

  /// Print the current type.
  void print(raw_ostream &os) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(Type arg);

  unsigned getSubclassData() const;
  void setSubclassData(unsigned val);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(type);
  }
  static Type getFromOpaquePointer(const void *pointer) {
    return Type((ImplType *)(pointer));
  }

protected:
  ImplType *type;
};

inline raw_ostream &operator<<(raw_ostream &os, Type type) {
  type.print(os);
  return os;
}

/// Integer types can have arbitrary bitwidth up to a large fixed limit.
class IntegerType : public Type {
public:
  using ImplType = detail::IntegerTypeStorage;
  IntegerType() = default;
  /* implicit */ IntegerType(Type::ImplType *ptr);

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
  static bool kindof(Kind kind) { return kind == Kind::Integer; }

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
  using ImplType = detail::FloatTypeStorage;
  FloatType() = default;
  /* implicit */ FloatType(Type::ImplType *ptr);

  static FloatType get(Kind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) {
    return kind >= Kind::FIRST_FLOATING_POINT_TYPE &&
           kind <= Kind::LAST_FLOATING_POINT_TYPE;
  }

  /// Return the bitwidth of this float type.
  unsigned getWidth() const;
};

inline FloatType Type::getBF16(MLIRContext *ctx) {
  return FloatType::get(Kind::BF16, ctx);
}
inline FloatType Type::getF16(MLIRContext *ctx) {
  return FloatType::get(Kind::F16, ctx);
}
inline FloatType Type::getF32(MLIRContext *ctx) {
  return FloatType::get(Kind::F32, ctx);
}
inline FloatType Type::getF64(MLIRContext *ctx) {
  return FloatType::get(Kind::F64, ctx);
}

/// Index is special integer-like type with unknown platform-dependent bit width
/// used in subscripts and loop induction variables.
class IndexType : public Type {
public:
  using ImplType = detail::IndexTypeStorage;
  IndexType() = default;
  /* implicit */ IndexType(Type::ImplType *ptr);

  /// Crete an IndexType instance, unique in the given context.
  static IndexType get(MLIRContext *context);

  /// Support method to enable LLVM-style type casting.
  static bool kindof(Kind kind) { return kind == Kind::Index; }
};

/// This is a type for the random collection of special base types.
class OtherType : public Type {
public:
  using ImplType = detail::OtherTypeStorage;
  OtherType() = default;
  /* implicit */ OtherType(Type::ImplType *ptr);

  static OtherType get(Kind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) {
    return kind >= Kind::FIRST_OTHER_TYPE && kind <= Kind::LAST_OTHER_TYPE;
  }
};

inline IndexType Type::getIndex(MLIRContext *ctx) {
  return IndexType::get(ctx);
}
inline OtherType Type::getTFControl(MLIRContext *ctx) {
  return OtherType::get(Kind::TFControl, ctx);
}
inline OtherType Type::getTFResource(MLIRContext *ctx) {
  return OtherType::get(Kind::TFResource, ctx);
}
inline OtherType Type::getTFString(MLIRContext *ctx) {
  return OtherType::get(Kind::TFString, ctx);
}
inline OtherType Type::getTFVariant(MLIRContext *ctx) {
  return OtherType::get(Kind::TFVariant, ctx);
}
inline OtherType Type::getTFComplex64(MLIRContext *ctx) {
  return OtherType::get(Kind::TFComplex64, ctx);
}
inline OtherType Type::getTFComplex128(MLIRContext *ctx) {
  return OtherType::get(Kind::TFComplex128, ctx);
}
inline OtherType Type::getTFF32REF(MLIRContext *ctx) {
  return OtherType::get(Kind::TFF32REF, ctx);
}

/// Function types map from a list of inputs to a list of results.
class FunctionType : public Type {
public:
  using ImplType = detail::FunctionTypeStorage;
  FunctionType() = default;
  /* implicit */ FunctionType(Type::ImplType *ptr);

  static FunctionType get(ArrayRef<Type> inputs, ArrayRef<Type> results,
                          MLIRContext *context);

  // Input types.
  unsigned getNumInputs() const { return getSubclassData(); }

  Type getInput(unsigned i) const { return getInputs()[i]; }

  ArrayRef<Type> getInputs() const;

  // Result types.
  unsigned getNumResults() const;

  Type getResult(unsigned i) const { return getResults()[i]; }

  ArrayRef<Type> getResults() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Function; }
};

/// This is a common base class between Vector, UnrankedTensor, and RankedTensor
/// types, because many operations work on values of these aggregate types.
class VectorOrTensorType : public Type {
public:
  using ImplType = detail::VectorOrTensorTypeStorage;
  VectorOrTensorType() = default;
  /* implicit */ VectorOrTensorType(Type::ImplType *ptr);

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
  static bool kindof(Kind kind) {
    return kind == Kind::Vector || kind == Kind::RankedTensor ||
           kind == Kind::UnrankedTensor;
  }
};

/// Vector types represent multi-dimensional SIMD vectors, and have a fixed
/// known constant shape with one or more dimension.
class VectorType : public VectorOrTensorType {
public:
  using ImplType = detail::VectorTypeStorage;
  VectorType() = default;
  /* implicit */ VectorType(Type::ImplType *ptr);

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
  static bool kindof(Kind kind) { return kind == Kind::Vector; }
};

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
class TensorType : public VectorOrTensorType {
public:
  using ImplType = detail::TensorTypeStorage;
  TensorType() = default;
  /* implicit */ TensorType(Type::ImplType *ptr);

  /// Return true if the specified element type is a TensorFlow type that is ok
  /// in a tensor.
  static bool isValidTFElementType(Type type) {
    return type.isa<FloatType>() || type.isa<IntegerType>() ||
           type.isa<OtherType>();
  }

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type) {
    return isValidTFElementType(type) || type.isa<VectorType>();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) {
    return kind == Kind::RankedTensor || kind == Kind::UnrankedTensor;
  }
};

/// Ranked tensor types represent multi-dimensional arrays that have a shape
/// with a fixed number of dimensions. Each shape element can be a positive
/// integer or unknown (represented -1).
class RankedTensorType : public TensorType {
public:
  using ImplType = detail::RankedTensorTypeStorage;
  RankedTensorType() = default;
  /* implicit */ RankedTensorType(Type::ImplType *ptr);

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

  static bool kindof(Kind kind) { return kind == Kind::RankedTensor; }
};

/// Unranked tensor types represent multi-dimensional arrays that have an
/// unknown shape.
class UnrankedTensorType : public TensorType {
public:
  using ImplType = detail::UnrankedTensorTypeStorage;
  UnrankedTensorType() = default;
  /* implicit */ UnrankedTensorType(Type::ImplType *ptr);

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed type.
  static UnrankedTensorType get(Type elementType);

  /// Get or create a new UnrankedTensorType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// UnrankedTensorType defined by the arguments would be ill-formed, emit
  /// errors and return a nullptr-wrapping type.
  static UnrankedTensorType getChecked(Type elementType, Location location);

  ArrayRef<int> getShape() const { return ArrayRef<int>(); }

  static bool kindof(Kind kind) { return kind == Kind::UnrankedTensor; }
};

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a positive integer or
/// unknown (represented by any negative integer). MemRef types also have an
/// affine map composition, represented as an array AffineMap pointers.
class MemRefType : public Type {
public:
  using ImplType = detail::MemRefTypeStorage;
  MemRefType() = default;
  /* implicit */ MemRefType(Type::ImplType *ptr);

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

  static bool kindof(Kind kind) { return kind == Kind::MemRef; }

private:
  static MemRefType getSafe(ArrayRef<int> shape, Type elementType,
                            ArrayRef<AffineMap> affineMapComposition,
                            unsigned memorySpace, Optional<Location> location);
};

// Make Type hashable.
inline ::llvm::hash_code hash_value(Type arg) {
  return ::llvm::hash_value(arg.type);
}

template <typename U> bool Type::isa() const {
  assert(type && "isa<> used on a null type.");
  return U::kindof(getKind());
}
template <typename U> U Type::dyn_cast() const {
  return isa<U>() ? U(type) : U(nullptr);
}
template <typename U> U Type::dyn_cast_or_null() const {
  return (type && isa<U>()) ? U(type) : U(nullptr);
}
template <typename U> U Type::cast() const {
  assert(isa<U>());
  return U(type);
}

} // end namespace mlir

namespace llvm {

// Type hash just like pointers.
template <> struct DenseMapInfo<mlir::Type> {
  static mlir::Type getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static mlir::Type getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Type val) { return mlir::hash_value(val); }
  static bool isEqual(mlir::Type LHS, mlir::Type RHS) { return LHS == RHS; }
};

/// We align TypeStorage by 8, so allow LLVM to steal the low bits.
template <> struct PointerLikeTypeTraits<mlir::Type> {
public:
  static inline void *getAsVoidPointer(mlir::Type I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Type getFromVoidPointer(void *P) {
    return mlir::Type::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif  // MLIR_IR_TYPES_H
