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

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class AffineMap;
class Function;
class FunctionAttr;
class FunctionType;
class IntegerSet;
class Location;
class MLIRContext;
class Type;
class VectorOrTensorType;

namespace detail {

struct AttributeStorage;
struct BoolAttributeStorage;
struct IntegerAttributeStorage;
struct FloatAttributeStorage;
struct StringAttributeStorage;
struct ArrayAttributeStorage;
struct AffineMapAttributeStorage;
struct IntegerSetAttributeStorage;
struct TypeAttributeStorage;
struct FunctionAttributeStorage;
struct ElementsAttributeStorage;
struct SplatElementsAttributeStorage;
struct DenseElementsAttributeStorage;
struct DenseIntElementsAttributeStorage;
struct DenseFPElementsAttributeStorage;
struct OpaqueElementsAttributeStorage;
struct SparseElementsAttributeStorage;

} // namespace detail

/// Attributes are known-constant values of operations and functions.
///
/// Instances of the Attribute class are immutable, uniqued, immortal, and owned
/// by MLIRContext.  As such, an Attribute is a POD interface to an underlying
/// storage pointer.
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
    IntegerSet,
    Function,

    SplatElements,
    DenseIntElements,
    DenseFPElements,
    OpaqueElements,
    SparseElements,
    FIRST_ELEMENTS_ATTR = SplatElements,
    LAST_ELEMENTS_ATTR = SparseElements,
  };

  using ImplType = detail::AttributeStorage;
  using ValueType = void;

  Attribute() : attr(nullptr) {}
  /* implicit */ Attribute(const ImplType *attr)
      : attr(const_cast<ImplType *>(attr)) {}

  Attribute(const Attribute &other) : attr(other.attr) {}
  Attribute &operator=(Attribute other) {
    attr = other.attr;
    return *this;
  }

  bool operator==(Attribute other) const { return attr == other.attr; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return attr; }

  bool operator!() const { return attr == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  /// Return the classification for this attribute.
  Kind getKind() const;

  /// Return true if this field is, or contains, a function attribute.
  bool isOrContainsFunction() const;

  /// Replace a function attribute or function attributes nested in an array
  /// attribute with another function attribute as defined by the provided
  /// remapping table.  Return the original attribute if it (or any of nested
  /// attributes) is not present in the table.
  Attribute remapFunctionAttrs(
      const llvm::DenseMap<Attribute, FunctionAttr> &remappingTable,
      MLIRContext *context) const;

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(Attribute arg);

protected:
  ImplType *attr;
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

/// Numeric attributes are (vector/tensor of) bool, integer, or floating-point
/// constants. For all the attributes, we can only build constant op out of
/// numeric attributes.
class NumericAttr : public Attribute {
public:
  using Attribute::Attribute;

  Type getType() const;

  static bool kindof(Kind kind);
};

class BoolAttr : public NumericAttr {
public:
  using NumericAttr::NumericAttr;
  using ImplType = detail::BoolAttributeStorage;
  using ValueType = bool;

  static BoolAttr get(bool value, MLIRContext *context);

  bool getValue() const;

  Type getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Bool; }
};

class IntegerAttr : public NumericAttr {
public:
  using NumericAttr::NumericAttr;
  using ImplType = detail::IntegerAttributeStorage;
  using ValueType = APInt;

  static IntegerAttr get(Type type, int64_t value);
  static IntegerAttr get(Type type, const APInt &value);

  APInt getValue() const;
  // TODO(jpienaar): Change callers to use getValue instead.
  int64_t getInt() const;

  Type getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Integer; }
};

class FloatAttr : public NumericAttr {
public:
  using NumericAttr::NumericAttr;
  using ImplType = detail::FloatAttributeStorage;
  using ValueType = APFloat;

  /// Return a float attribute for the specified value in the specified type.
  /// These methods should only be used for simple constant values, e.g 1.0/2.0,
  /// that are known-valid both as host double and the 'type' format.
  static FloatAttr get(Type type, double value);
  static FloatAttr getChecked(Type type, double value, Location loc);

  /// Return a float attribute for the specified value in the specified type.
  static FloatAttr get(Type type, const APFloat &value);

  APFloat getValue() const;

  /// This function is used to convert the value to a double, even if it loses
  /// precision.
  double getValueAsDouble() const;

  Type getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Float; }
};

class StringAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ImplType = detail::StringAttributeStorage;
  using ValueType = StringRef;

  static StringAttr get(StringRef bytes, MLIRContext *context);

  StringRef getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::String; }
};

/// Array attributes are lists of other attributes.  They are not necessarily
/// type homogenous given that attributes don't, in general, carry types.
class ArrayAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ImplType = detail::ArrayAttributeStorage;
  using ValueType = ArrayRef<Attribute>;

  static ArrayAttr get(ArrayRef<Attribute> value, MLIRContext *context);

  ArrayRef<Attribute> getValue() const;

  size_t size() const { return getValue().size(); }

  using iterator = llvm::ArrayRef<Attribute>::iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Array; }
};

class AffineMapAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ImplType = detail::AffineMapAttributeStorage;
  using ValueType = AffineMap;

  static AffineMapAttr get(AffineMap value);

  AffineMap getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::AffineMap; }
};

class IntegerSetAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ImplType = detail::IntegerSetAttributeStorage;
  using ValueType = IntegerSet;

  static IntegerSetAttr get(IntegerSet value);

  IntegerSet getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::IntegerSet; }
};

class TypeAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ImplType = detail::TypeAttributeStorage;
  using ValueType = Type;

  static TypeAttr get(Type type, MLIRContext *context);

  Type getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Type; }
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
  using Attribute::Attribute;
  using ImplType = detail::FunctionAttributeStorage;
  using ValueType = Function *;

  static FunctionAttr get(const Function *value, MLIRContext *context);

  Function *getValue() const;

  FunctionType getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::Function; }

  /// This function is used by the internals of the Function class to null out
  /// attributes refering to functions that are about to be deleted.
  static void dropFunctionReference(Function *value);
};

/// A base attribute represents a reference to a vector or tensor constant.
class ElementsAttr : public NumericAttr {
public:
  using NumericAttr::NumericAttr;
  using ImplType = detail::ElementsAttributeStorage;

  VectorOrTensorType getType() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) {
    return kind >= Kind::FIRST_ELEMENTS_ATTR &&
           kind <= Kind::LAST_ELEMENTS_ATTR;
  }
};

/// An attribute represents a reference to a splat vecctor or tensor constant,
/// meaning all of the elements have the same value.
class SplatElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;
  using ImplType = detail::SplatElementsAttributeStorage;
  using ValueType = Attribute;

  static SplatElementsAttr get(VectorOrTensorType type, Attribute elt);
  Attribute getValue() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::SplatElements; }
};

/// An attribute represents a reference to a dense vector or tensor object.
///
/// This class is designed to store elements with any bit widths equal or less
/// than 64.
class DenseElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;
  using ImplType = detail::DenseElementsAttributeStorage;

  /// It assumes the elements in the input array have been truncated to the bits
  /// width specified by the element type (note all float type are 64 bits).
  static DenseElementsAttr get(VectorOrTensorType type, ArrayRef<char> data);

  // Constructs a dense elements attribute from an array of element values. Each
  // element attribute value is expected to be an element of 'type'.
  static DenseElementsAttr get(VectorOrTensorType type,
                               ArrayRef<Attribute> values);

  /// Return the value at the given index. If index does not refer to a valid
  /// element, then a null attribute is returned.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  void getValues(SmallVectorImpl<Attribute> &values) const;

  ArrayRef<char> getRawData() const;

  /// Writes value to the bit position `bitPos` in array `rawData`. 'rawData' is
  /// expected to be a 64-bit aligned storage address.
  static void writeBits(char *rawData, size_t bitPos, APInt value);

  /// Reads the next `bitWidth` bits from the bit position `bitPos` in array
  /// `rawData`. 'rawData' is expected to be a 64-bit aligned storage address.
  static APInt readBits(const char *rawData, size_t bitPos, size_t bitWidth);

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) {
    return kind == Kind::DenseIntElements || kind == Kind::DenseFPElements;
  }

protected:
  /// Parses the raw integer internal value for each dense element into
  /// 'values'.
  void getRawValues(SmallVectorImpl<APInt> &values) const;
};

/// An attribute represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;
  using DenseElementsAttr::getValues;
  using DenseElementsAttr::ImplType;

  /// Gets the integer value of each of the dense elements.
  void getValues(SmallVectorImpl<APInt> &values) const;

  APInt getValue(ArrayRef<unsigned> indices) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::DenseIntElements; }
};

/// An attribute represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;
  using DenseElementsAttr::getValues;
  using DenseElementsAttr::ImplType;

  /// Gets the float value of each of the dense elements.
  void getValues(SmallVectorImpl<APFloat> &values) const;

  APFloat getValue(ArrayRef<unsigned> indices) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::DenseFPElements; }
};

/// An attribute represents a reference to a tensor constant with opaque
/// content. This respresentation is for tensor constants which the compiler
/// doesn't need to interpret.
class OpaqueElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;
  using ImplType = detail::OpaqueElementsAttributeStorage;
  using ValueType = StringRef;

  static OpaqueElementsAttr get(VectorOrTensorType type, StringRef bytes);

  StringRef getValue() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::OpaqueElements; }
};

/// An attribute represents a reference to a sparse vector or tensor object.
///
/// This class uses COO (coordinate list) encoding to represent the sparse
/// elements in an element attribute. Specifically, the sparse vector/tensor
/// stores the indices and values as two separate dense elements attributes of
/// tensor type (even if the sparse attribute is of vector type, in order to
/// support empty lists). The dense elements attribute indices is a 2-D tensor
/// of 64-bit integer elements with shape [N, ndims], which specifies the
/// indices of the elements in the sparse tensor that contains nonzero values.
/// The dense elements attribute values is a 1-D tensor with shape [N], and it
/// supplies the corresponding values for the indices.
///
/// For example,
/// `sparse<tensor<3x4xi32>, [[0, 0], [1, 2]], [1, 5]>` represents tensor
/// [[1, 0, 0, 0],
///  [0, 0, 5, 0],
///  [0, 0, 0, 0]].
class SparseElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;
  using ImplType = detail::SparseElementsAttributeStorage;

  static SparseElementsAttr get(VectorOrTensorType type,
                                DenseIntElementsAttr indices,
                                DenseElementsAttr values);

  DenseIntElementsAttr getIndices() const;

  DenseElementsAttr getValues() const;

  /// Return the value of the element at the given index.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(Kind kind) { return kind == Kind::SparseElements; }
};

template <typename U> bool Attribute::isa() const {
  assert(attr && "isa<> used on a null attribute.");
  return U::kindof(getKind());
}
template <typename U> U Attribute::dyn_cast() const {
  return isa<U>() ? U(attr) : U(nullptr);
}
template <typename U> U Attribute::dyn_cast_or_null() const {
  return (attr && isa<U>()) ? U(attr) : U(nullptr);
}
template <typename U> U Attribute::cast() const {
  assert(isa<U>());
  return U(attr);
}

// Make Attribute hashable.
inline ::llvm::hash_code hash_value(Attribute arg) {
  return ::llvm::hash_value(arg.attr);
}

} // end namespace mlir.

namespace llvm {

// Attribute hash just like pointers
template <> struct DenseMapInfo<mlir::Attribute> {
  static mlir::Attribute getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static mlir::Attribute getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Attribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Attribute LHS, mlir::Attribute RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif
