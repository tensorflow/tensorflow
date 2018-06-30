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
  class MLIRContext;
  class PrimitiveType;
  class IntegerType;

/// Integer identifier for all the concrete type kinds.
enum class TypeKind {
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

  // TODO: MemRef types.
};

/// Instances of the Type class are immutable, uniqued, immortal, and owned by
/// MLIRContext.  As such, they are passed around by raw non-const pointer.
///
class Type {
public:

  /// Return the classification for this type.
  TypeKind getKind() const {
    return kind;
  }

  /// Return true if this type is the specified kind.
  bool is(TypeKind k) const {
    return kind == k;
  }

  /// Return the LLVMContext in which this type was uniqued.
  MLIRContext *getContext() const { return context; }

  /// Print the current type.
  void print(raw_ostream &os) const;
  void dump() const;

  // Convenience factories.
  static IntegerType *getInt(unsigned width, MLIRContext *ctx);
  static PrimitiveType *getAffineInt(MLIRContext *ctx);
  static PrimitiveType *getBF16(MLIRContext *ctx);
  static PrimitiveType *getF16(MLIRContext *ctx);
  static PrimitiveType *getF32(MLIRContext *ctx);
  static PrimitiveType *getF64(MLIRContext *ctx);

protected:
  explicit Type(TypeKind kind, MLIRContext *context)
    : context(context), kind(kind), subclassData(0) {
  }
  explicit Type(TypeKind kind, MLIRContext *context, unsigned subClassData)
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
  /// This refers to the MLIRContext in which this type was uniqued.
  MLIRContext *const context;

  /// Classification of the subclass, used for type checking.
  TypeKind kind : 8;

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
  static PrimitiveType *get(TypeKind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Type *type) {
    return type->getKind() <= TypeKind::LAST_PRIMITIVE_TYPE;
  }
private:
  PrimitiveType(TypeKind kind, MLIRContext *context);
};


inline PrimitiveType *Type::getAffineInt(MLIRContext *ctx) {
  return PrimitiveType::get(TypeKind::AffineInt, ctx);
}
inline PrimitiveType *Type::getBF16(MLIRContext *ctx) {
  return PrimitiveType::get(TypeKind::BF16, ctx);
}
inline PrimitiveType *Type::getF16(MLIRContext *ctx) {
  return PrimitiveType::get(TypeKind::F16, ctx);
}
inline PrimitiveType *Type::getF32(MLIRContext *ctx) {
  return PrimitiveType::get(TypeKind::F32, ctx);
}
inline PrimitiveType *Type::getF64(MLIRContext *ctx) {
  return PrimitiveType::get(TypeKind::F64, ctx);
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
    return type->getKind() == TypeKind::Integer;
  }
private:
  unsigned width;
  IntegerType(unsigned width, MLIRContext *context);
};

inline IntegerType *Type::getInt(unsigned width, MLIRContext *ctx) {
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
    return type->getKind() == TypeKind::Function;
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
    return type->getKind() == TypeKind::Vector;
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
    return type->getKind() == TypeKind::RankedTensor ||
           type->getKind() == TypeKind::UnrankedTensor;
  }

protected:
  /// The type of each scalar element of the tensor.
  Type *elementType;

  TensorType(TypeKind kind, Type *elementType, MLIRContext *context);
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
    return type->getKind() == TypeKind::RankedTensor;
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
    return type->getKind() == TypeKind::UnrankedTensor;
  }

private:
  UnrankedTensorType(Type *elementType, MLIRContext *context);
};

} // end namespace mlir

#endif  // MLIR_IR_TYPES_H
