//===- Attributes.h - MLIR Attribute Classes --------------------*- C++ -*-===//
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

#ifndef MLIR_IR_ATTRIBUTES_H
#define MLIR_IR_ATTRIBUTES_H

#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Function;
class FunctionType;
class MLIRContext;
class Type;
class VectorOrTensorType;

/// Attributes are known-constant values of operations and functions.
///
/// Instances of the Attribute class are immutable, uniqued, immortal, and owned
/// by MLIRContext.  As such, they are passed around by raw non-const pointer.
class Attribute {
public:
  enum class Kind {
    Bool,
    Integer,
    Float,
    String,
    Type,
    Array,
    AffineMap,
    Function,

    SplatElements,
    DenseIntElements,
    DenseFPElements,
    SparseElements,
    FIRST_ELEMENTS_ATTR = SplatElements,
    LAST_ELEMENTS_ATTR = SparseElements,
  };

  /// Return the classification for this attribute.
  Kind getKind() const { return kind; }

  /// Return true if this field is, or contains, a function attribute.
  bool isOrContainsFunction() const { return isOrContainsFunctionCache; }

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  explicit Attribute(Kind kind, bool isOrContainsFunction)
      : kind(kind), isOrContainsFunctionCache(isOrContainsFunction) {}
  ~Attribute() {}

private:
  /// Classification of the subclass, used for type checking.
  Kind kind : 8;

  /// This field is true if this is, or contains, a function attribute.
  bool isOrContainsFunctionCache : 1;

  Attribute(const Attribute &) = delete;
  void operator=(const Attribute &) = delete;
};

inline raw_ostream &operator<<(raw_ostream &os, const Attribute &attr) {
  attr.print(os);
  return os;
}

class BoolAttr : public Attribute {
public:
  static BoolAttr *get(bool value, MLIRContext *context);

  bool getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Bool;
  }

private:
  BoolAttr(bool value)
      : Attribute(Kind::Bool, /*isOrContainsFunction=*/false), value(value) {}
  ~BoolAttr() = delete;
  bool value;
};

class IntegerAttr : public Attribute {
public:
  static IntegerAttr *get(int64_t value, MLIRContext *context);

  int64_t getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Integer;
  }

private:
  IntegerAttr(int64_t value)
      : Attribute(Kind::Integer, /*isOrContainsFunction=*/false), value(value) {
  }
  ~IntegerAttr() = delete;
  int64_t value;
};

class FloatAttr : public Attribute {
public:
  static FloatAttr *get(double value, MLIRContext *context);

  // TODO: This should really be implemented in terms of APFloat for
  // correctness, otherwise constant folding will be done with host math.  This
  // is completely incorrect for BF16 and other datatypes, and subtly wrong
  // for float32.
  double getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Float;
  }

private:
  FloatAttr(double value)
      : Attribute(Kind::Float, /*isOrContainsFunction=*/false), value(value) {}
  ~FloatAttr() = delete;
  double value;
};

class StringAttr : public Attribute {
public:
  static StringAttr *get(StringRef bytes, MLIRContext *context);

  StringRef getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::String;
  }

private:
  StringAttr(StringRef value)
      : Attribute(Kind::String, /*isOrContainsFunction=*/false), value(value) {}
  ~StringAttr() = delete;
  StringRef value;
};

/// Array attributes are lists of other attributes.  They are not necessarily
/// type homogenous given that attributes don't, in general, carry types.
class ArrayAttr : public Attribute {
public:
  static ArrayAttr *get(ArrayRef<Attribute *> value, MLIRContext *context);

  ArrayRef<Attribute *> getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Array;
  }

private:
  ArrayAttr(ArrayRef<Attribute *> value, bool isOrContainsFunction)
      : Attribute(Kind::Array, isOrContainsFunction), value(value) {}
  ~ArrayAttr() = delete;
  ArrayRef<Attribute *> value;
};

class AffineMapAttr : public Attribute {
public:
  static AffineMapAttr *get(AffineMap value);

  AffineMap getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::AffineMap;
  }

private:
  AffineMapAttr(AffineMap value)
      : Attribute(Kind::AffineMap, /*isOrContainsFunction=*/false),
        value(value) {}
  ~AffineMapAttr() = delete;
  AffineMap value;
};

class TypeAttr : public Attribute {
public:
  static TypeAttr *get(Type *type, MLIRContext *context);

  Type *getValue() const { return value; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Type;
  }

private:
  TypeAttr(Type *value)
      : Attribute(Kind::Type, /*isOrContainsFunction=*/false), value(value) {}
  ~TypeAttr() = delete;
  Type *value;
};

/// A function attribute represents a reference to a function object.
///
/// When working with IR, it is important to know that a function attribute can
/// exist with a null Function inside of it, which occurs when a function object
/// is deleted that had an attribute which referenced it.  No references to this
/// attribute should persist across the transformation, but that attribute will
/// remain in MLIRContext.
class FunctionAttr : public Attribute {
public:
  static FunctionAttr *get(const Function *value, MLIRContext *context);

  Function *getValue() const { return value; }

  FunctionType *getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Function;
  }

  /// This function is used by the internals of the Function class to null out
  /// attributes refering to functions that are about to be deleted.
  static void dropFunctionReference(Function *value);

private:
  FunctionAttr(Function *value)
      : Attribute(Kind::Function, /*isOrContainsFunction=*/true), value(value) {
  }
  ~FunctionAttr() = delete;
  Function *value;
};

/// A base attribute represents a reference to a vector or tensor constant.
class ElementsAttr : public Attribute {
public:
  ElementsAttr(Kind kind, VectorOrTensorType *type)
      : Attribute(kind, /*isOrContainsFunction=*/false), type(type) {}

  VectorOrTensorType *getType() const { return type; }

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() >= Kind::FIRST_ELEMENTS_ATTR &&
           attr->getKind() <= Kind::LAST_ELEMENTS_ATTR;
  }

private:
  VectorOrTensorType *type;
};

/// An attribute represents a reference to a splat vecctor or tensor constant,
/// meaning all of the elements have the same value.
class SplatElementsAttr : public ElementsAttr {
public:
  static SplatElementsAttr *get(VectorOrTensorType *type, Attribute *elt);
  Attribute *getValue() const { return elt; }

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::SplatElements;
  }

private:
  SplatElementsAttr(VectorOrTensorType *type, Attribute *elt)
      : ElementsAttr(Kind::SplatElements, type), elt(elt) {}
  Attribute *elt;
};

/// An attribute represents a reference to a dense vector or tensor object.
///
/// This class is designed to store elements with any bit widths equal or less
/// than 64.
class DenseElementsAttr : public ElementsAttr {
public:
  /// It assumes the elements in the input array have been truncated to the bits
  /// width specified by the element type (note all float type are 64 bits).
  /// When the value is retrieved, the bits are read from the storage and extend
  /// to 64 bits if necessary.
  static DenseElementsAttr *get(VectorOrTensorType *type, ArrayRef<char> data);

  // TODO: Read the data from the attribute list and compress them
  // to a character array. Then call the above method to construct the
  // attribute.
  static DenseElementsAttr *get(VectorOrTensorType *type,
                                ArrayRef<Attribute *> values);

  void getValues(SmallVectorImpl<Attribute *> &values) const;

  ArrayRef<char> getRawData() const { return data; }

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::DenseIntElements ||
           attr->getKind() == Kind::DenseFPElements;
  }

protected:
  DenseElementsAttr(Kind kind, VectorOrTensorType *type, ArrayRef<char> data)
      : ElementsAttr(kind, type), data(data) {}

private:
  ArrayRef<char> data;
};

/// An attribute represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseElementsAttr {
public:
  DenseIntElementsAttr(VectorOrTensorType *type, ArrayRef<char> data,
                       size_t bitsWidth)
      : DenseElementsAttr(Kind::DenseIntElements, type, data),
        bitsWidth(bitsWidth) {}

  // TODO: returns APInts instead of IntegerAttr.
  void getValues(SmallVectorImpl<Attribute *> &values) const;

  APInt getValue(ArrayRef<unsigned> indices) const;

  /// Writes the lowest `bitWidth` bits of `value` to the bit position `bitPos`
  /// in array `rawData`.
  static void writeBits(char *rawData, size_t bitPos, size_t bitWidth,
                        uint64_t value);

  /// Reads the next `bitWidth` bits from the bit position `bitPos` in array
  /// `rawData` and return them as the lowest bits of an uint64 integer.
  static uint64_t readBits(const char *rawData, size_t bitPos,
                           size_t bitsWidth);

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::DenseIntElements;
  }

private:
  ~DenseIntElementsAttr() = delete;

  size_t bitsWidth;
};

/// An attribute represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseElementsAttr {
public:
  DenseFPElementsAttr(VectorOrTensorType *type, ArrayRef<char> data)
      : DenseElementsAttr(Kind::DenseFPElements, type, data) {}

  // TODO: returns APFPs instead of FloatAttr.
  void getValues(SmallVectorImpl<Attribute *> &values) const;

  APFloat getValue(ArrayRef<unsigned> indices) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::DenseFPElements;
  }

private:
  ~DenseFPElementsAttr() = delete;
};

/// An attribute represents a reference to a sparse vector or tensor object.
///
/// This class uses COO (coordinate list) encoding to represent the sparse
/// elements in an element attribute. Specifically, the sparse vector/tensor
/// stores the indices and values as two separate dense elements attributes. The
/// dense elements attribute indices is a 2-D tensor with shape [N, ndims],
/// which specifies the indices of the elements in the sparse tensor that
/// contains nonzero values. The dense elements attribute values is a 1-D tensor
/// with shape [N], and it supplies the corresponding values for the indices.
///
/// For example,
/// `sparse<tensor<3x4xi32>, [[0, 0], [1, 2]], [1, 5]>` represents tensor
/// [[1, 0, 0, 0],
///  [0, 0, 5, 0],
///  [0, 0, 0, 0]].
class SparseElementsAttr : public ElementsAttr {
public:
  static SparseElementsAttr *get(VectorOrTensorType *type,
                                 DenseIntElementsAttr *indices,
                                 DenseElementsAttr *values);

  DenseIntElementsAttr *getIndices() const { return indices; }

  DenseElementsAttr *getValues() const { return values; }

  /// Return the value at the given index.
  Attribute *getValue(ArrayRef<unsigned> index) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::SparseElements;
  }

private:
  SparseElementsAttr(VectorOrTensorType *type, DenseIntElementsAttr *indices,
                     DenseElementsAttr *values)
      : ElementsAttr(Kind::SparseElements, type), indices(indices),
        values(values) {}
  ~SparseElementsAttr() = delete;

  DenseIntElementsAttr *const indices;
  DenseElementsAttr *const values;
};
} // end namespace mlir.

#endif
