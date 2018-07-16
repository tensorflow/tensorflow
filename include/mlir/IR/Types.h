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

namespace mlir {
class AffineMap;
class MLIRContext;
class PrimitiveType;
class IntegerType;

/// Instances of the Type class are immutable, uniqued, immortal, and owned by
/// MLIRContext.  As such, they are passed around by raw non-const pointer.
///
class Type {
public:
  /// Integer identifier for all the concrete type kinds.
  enum class Kind {
    // Target pointer sized integer.
    AffineInt,

    // Floating point.
    BF16,
    F16,
    F32,
    F64,

    /// This is a marker for the last primitive type.  The range of primitive
    /// types is expected to be this element and earlier.
    LAST_PRIMITIVE_TYPE = F64,

    // Derived types.
    Integer,
    Function,
    Vector,
    RankedTensor,
    UnrankedTensor,
    MemRef,
  };

  /// Return the classification for this type.
  Kind getKind() const {
    return kind;
  }

  /// Return the LLVMContext in which this type was uniqued.
  MLIRContext *getContext() const { return context; }

  /// Print the current type.
  void print(raw_ostream &os) const;
  void dump() const;

  // Convenience factories.
  static IntegerType *getInteger(unsigned width, MLIRContext *ctx);
  static PrimitiveType *getAffineInt(MLIRContext *ctx);
  static PrimitiveType *getBF16(MLIRContext *ctx);
  static PrimitiveType *getF16(MLIRContext *ctx);
  static PrimitiveType *getF32(MLIRContext *ctx);
  static PrimitiveType *getF64(MLIRContext *ctx);

protected:
  explicit Type(Kind kind, MLIRContext *context)
    : context(context), kind(kind), subclassData(0) {
  }
  explicit Type(Kind kind, MLIRContext *context, unsigned subClassData)
    : Type(kind, context) {
    setSubclassData(subClassData);
  }

  ~Type() = default;

  unsigned getSubclassData() const { return subclassData; }

  void setSubclassData(unsigned val) {
    subclassData = val;
    // Ensure we don't have any accidental truncation.
    assert(getSubclassData() == val && "Subclass data too large for field");
  }

private:
  Type(const Type&) = delete;
  void operator=(const Type&) = delete;
  /// This refers to the MLIRContext in which this type was uniqued.
  MLIRContext *const context;

  /// Classification of the subclass, used for type checking.
  Kind kind : 8;

  // Space for subclasses to store data.
  unsigned subclassData : 24;
};

inline raw_ostream &operator<<(raw_ostream &os, const Type &type) {
  type.print(os);
  return os;
}

/// Primitive types are the atomic base of the type system, including integer
/// and floating point values.
class PrimitiveType : public Type {
public:
  static PrimitiveType *get(Kind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() <= Kind::LAST_PRIMITIVE_TYPE;
  }
private:
  PrimitiveType(Kind kind, MLIRContext *context);
};

inline PrimitiveType *Type::getAffineInt(MLIRContext *ctx) {
  return PrimitiveType::get(Kind::AffineInt, ctx);
}
inline PrimitiveType *Type::getBF16(MLIRContext *ctx) {
  return PrimitiveType::get(Kind::BF16, ctx);
}
inline PrimitiveType *Type::getF16(MLIRContext *ctx) {
  return PrimitiveType::get(Kind::F16, ctx);
}
inline PrimitiveType *Type::getF32(MLIRContext *ctx) {
  return PrimitiveType::get(Kind::F32, ctx);
}
inline PrimitiveType *Type::getF64(MLIRContext *ctx) {
  return PrimitiveType::get(Kind::F64, ctx);
}

/// Integer types can have arbitrary bitwidth up to a large fixed limit of 4096.
class IntegerType : public Type {
public:
  static IntegerType *get(unsigned width, MLIRContext *context);

  /// Return the bitwidth of this integer type.
  unsigned getWidth() const {
    return width;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() == Kind::Integer;
  }
private:
  unsigned width;
  IntegerType(unsigned width, MLIRContext *context);
};

inline IntegerType *Type::getInteger(unsigned width, MLIRContext *ctx) {
  return IntegerType::get(width, ctx);
}

/// Function types map from a list of inputs to a list of results.
class FunctionType : public Type {
public:
  static FunctionType *get(ArrayRef<Type*> inputs, ArrayRef<Type*> results,
                           MLIRContext *context);

  ArrayRef<Type*> getInputs() const {
    return ArrayRef<Type*>(inputsAndResults, getSubclassData());
  }

  ArrayRef<Type*> getResults() const {
    return ArrayRef<Type*>(inputsAndResults+getSubclassData(), numResults);
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() == Kind::Function;
  }

private:
  unsigned numResults;
  Type *const *inputsAndResults;

  FunctionType(Type *const *inputsAndResults, unsigned numInputs,
               unsigned numResults, MLIRContext *context);
};


/// Vector types represent multi-dimensional SIMD vectors, and have a fixed
/// known constant shape with one or more dimension.
class VectorType : public Type {
public:
  static VectorType *get(ArrayRef<unsigned> shape, Type *elementType);

  ArrayRef<unsigned> getShape() const {
    return ArrayRef<unsigned>(shapeElements, getSubclassData());
  }

  PrimitiveType *getElementType() const {
    return elementType;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() == Kind::Vector;
  }

private:
  const unsigned *shapeElements;
  PrimitiveType *elementType;

  VectorType(ArrayRef<unsigned> shape, PrimitiveType *elementType,
             MLIRContext *context);
};

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
class TensorType : public Type {
public:
  Type *getElementType() const { return elementType; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() == Kind::RankedTensor ||
           type->getKind() == Kind::UnrankedTensor;
  }

protected:
  /// The type of each scalar element of the tensor.
  Type *elementType;

  TensorType(Kind kind, Type *elementType, MLIRContext *context);
};

/// Ranked tensor types represent multi-dimensional arrays that have a shape
/// with a fixed number of dimensions. Each shape element can be a positive
/// integer or unknown (represented by any negative integer).
class RankedTensorType : public TensorType {
public:
  static RankedTensorType *get(ArrayRef<int> shape,
                               Type *elementType);

  ArrayRef<int> getShape() const {
    return ArrayRef<int>(shapeElements, getSubclassData());
  }

  unsigned getRank() const { return getShape().size(); }

  static bool classof(const Type *type) {
    return type->getKind() == Kind::RankedTensor;
  }

private:
  const int *shapeElements;

  RankedTensorType(ArrayRef<int> shape, Type *elementType,
                   MLIRContext *context);
};

/// Unranked tensor types represent multi-dimensional arrays that have an
/// unknown shape.
class UnrankedTensorType : public TensorType {
public:
  static UnrankedTensorType *get(Type *elementType);

  static bool classof(const Type *type) {
    return type->getKind() == Kind::UnrankedTensor;
  }

private:
  UnrankedTensorType(Type *elementType, MLIRContext *context);
};

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a positive integer or
/// unknown (represented by any negative integer). MemRef types also have an
/// affine map composition, represented as an array AffineMap pointers.
// TODO: Use -1 for unknown dimensions (rather than arbitrary negative numbers).
class MemRefType : public Type {
public:
  /// Get or create a new MemRefType based on shape, element type, affine
  /// map composition, and memory space.
  static MemRefType *get(ArrayRef<int> shape, Type *elementType,
                         ArrayRef<AffineMap*> affineMapComposition,
                         unsigned memorySpace);

  /// Returns an array of memref shape dimension sizes.
  ArrayRef<int> getShape() const {
    return ArrayRef<int>(shapeElements, getSubclassData());
  }

  /// Returns the elemental type for this memref shape.
  Type *getElementType() const { return elementType; }

  /// Returns an array of affine map pointers representing the memref affine
  /// map composition.
  ArrayRef<AffineMap*> getAffineMaps() const;

  /// Returns the memory space in which data referred to by this memref resides.
  unsigned getMemorySpace() const { return memorySpace; }

  static bool classof(const Type *type) {
    return type->getKind() == Kind::MemRef;
  }

private:
  /// The type of each scalar element of the memref.
  Type *elementType;
  /// An array of integers which stores the shape dimension sizes.
  const int *shapeElements;
  /// The number of affine maps in the 'affineMapList' array.
  unsigned numAffineMaps;
  /// List of affine maps in affine map composition.
  AffineMap *const *const affineMapList;
  /// Memory space in which data referenced by memref resides.
  unsigned memorySpace;

  MemRefType(ArrayRef<int> shape, Type *elementType,
             ArrayRef<AffineMap*> affineMapList, unsigned memorySpace,
             MLIRContext *context);
};

} // end namespace mlir

#endif  // MLIR_IR_TYPES_H
