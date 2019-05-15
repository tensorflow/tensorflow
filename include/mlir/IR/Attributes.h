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

#include "mlir/IR/AttributeSupport.h"
#include "llvm/ADT/APFloat.h"

namespace mlir {
class AffineMap;
class Dialect;
class Function;
class FunctionAttr;
class FunctionType;
class Identifier;
class IntegerSet;
class Location;
class MLIRContext;
class Type;
class VectorOrTensorType;

namespace detail {

struct OpaqueAttributeStorage;
struct BoolAttributeStorage;
struct IntegerAttributeStorage;
struct FloatAttributeStorage;
struct StringAttributeStorage;
struct ArrayAttributeStorage;
struct AffineMapAttributeStorage;
struct IntegerSetAttributeStorage;
struct TypeAttributeStorage;
struct FunctionAttributeStorage;
struct SplatElementsAttributeStorage;
struct DenseElementsAttributeStorage;
struct DenseIntElementsAttributeStorage;
struct DenseFPElementsAttributeStorage;
struct OpaqueElementsAttributeStorage;
struct SparseElementsAttributeStorage;

class AttributeListStorage;

} // namespace detail

/// Attributes are known-constant values of operations and functions.
///
/// Instances of the Attribute class are references to immutable, uniqued,
/// and immortal values owned by MLIRContext. As such, an Attribute is a thin
/// wrapper around an underlying storage pointer. Attributes are usually passed
/// by value.
class Attribute {
public:
  /// Integer identifier for all the concrete attribute kinds.
  enum Kind {
  // Reserve attribute kinds for dialect specific extensions.
#define DEFINE_SYM_KIND_RANGE(Dialect)                                         \
  FIRST_##Dialect##_ATTR, LAST_##Dialect##_ATTR = FIRST_##Dialect##_ATTR + 0xff,
#include "DialectSymbolRegistry.def"
  };

  /// Utility class for implementing attributes.
  template <typename ConcreteType, typename BaseType = Attribute,
            typename StorageType = AttributeStorage>
  using AttrBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::AttributeUniquer>;

  using ImplType = AttributeStorage;
  using ValueType = void;

  Attribute() : impl(nullptr) {}
  /* implicit */ Attribute(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Attribute(const Attribute &other) : impl(other.impl) {}
  Attribute &operator=(Attribute other) {
    impl = other.impl;
    return *this;
  }

  bool operator==(Attribute other) const { return impl == other.impl; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  // Support dyn_cast'ing Attribute to itself.
  static bool kindof(unsigned) { return true; }

  /// Return the classification for this attribute.
  unsigned getKind() const { return impl->getKind(); }

  /// Return the type of this attribute.
  Type getType() const;

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  const Dialect &getDialect() const;

  /// Return true if this field is, or contains, a function attribute.
  bool isOrContainsFunction() const;

  /// Replace a function attribute or function attributes nested in an array
  /// attribute with another function attribute as defined by the provided
  /// remapping table.  Return the original attribute if it (or any of nested
  /// attributes) is not present in the table.
  Attribute remapFunctionAttrs(
      const llvm::DenseMap<Attribute, FunctionAttr> &remappingTable) const;

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Get an opaque pointer to the attribute.
  const void *getAsOpaquePointer() const { return impl; }
  /// Construct an attribute from the opaque pointer representation.
  static Attribute getFromOpaquePointer(const void *ptr) {
    return Attribute(
        const_cast<ImplType *>(reinterpret_cast<const ImplType *>(ptr)));
  }

  friend ::llvm::hash_code hash_value(Attribute arg);

protected:
  ImplType *impl;
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

namespace StandardAttributes {
enum Kind {
  Unit = Attribute::FIRST_STANDARD_ATTR,
  Opaque,
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
} // namespace StandardAttributes

/// Unit attributes are attributes that hold no specific value and are given
/// meaning by their existence.
class UnitAttr : public Attribute::AttrBase<UnitAttr> {
public:
  using Base::Base;

  static UnitAttr get(MLIRContext *context) {
    return Base::get(context, StandardAttributes::Unit);
  }

  static bool kindof(unsigned kind) { return kind == StandardAttributes::Unit; }
};

/// Opaque attributes represent attributes of non-registered dialects. These are
/// attribute represented in their raw string form, and can only usefully be
/// tested for attribute equality.
class OpaqueAttr : public Attribute::AttrBase<OpaqueAttr, Attribute,
                                              detail::OpaqueAttributeStorage> {
public:
  using Base::Base;

  /// Get or create a new OpaqueAttr with the provided dialect and string data.
  static OpaqueAttr get(Identifier dialect, StringRef attrData,
                        MLIRContext *context);

  /// Get or create a new OpaqueAttr with the provided dialect and string data.
  /// If the given identifier is not a valid namespace for a dialect, then a
  /// null attribute is returned.
  static OpaqueAttr getChecked(Identifier dialect, StringRef attrData,
                               MLIRContext *context, Location location);

  /// Returns the dialect namespace of the opaque attribute.
  Identifier getDialectNamespace() const;

  /// Returns the raw attribute data of the opaque attribute.
  StringRef getAttrData() const;

  /// Verify the construction of an opaque attribute.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc,
                               MLIRContext *context, Identifier dialect,
                               StringRef attrData);

  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Opaque;
  }
};

class BoolAttr : public Attribute::AttrBase<BoolAttr, Attribute,
                                            detail::BoolAttributeStorage> {
public:
  using Base::Base;
  using ValueType = bool;

  static BoolAttr get(bool value, MLIRContext *context);

  bool getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardAttributes::Bool; }
};

class IntegerAttr
    : public Attribute::AttrBase<IntegerAttr, Attribute,
                                 detail::IntegerAttributeStorage> {
public:
  using Base::Base;
  using ValueType = APInt;

  static IntegerAttr get(Type type, int64_t value);
  static IntegerAttr get(Type type, const APInt &value);

  APInt getValue() const;
  // TODO(jpienaar): Change callers to use getValue instead.
  int64_t getInt() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Integer;
  }
};

class FloatAttr : public Attribute::AttrBase<FloatAttr, Attribute,
                                             detail::FloatAttributeStorage> {
public:
  using Base::Base;
  using ValueType = APFloat;

  /// Return a float attribute for the specified value in the specified type.
  /// These methods should only be used for simple constant values, e.g 1.0/2.0,
  /// that are known-valid both as host double and the 'type' format.
  static FloatAttr get(Type type, double value);
  static FloatAttr getChecked(Type type, double value, Location loc);

  /// Return a float attribute for the specified value in the specified type.
  static FloatAttr get(Type type, const APFloat &value);
  static FloatAttr getChecked(Type type, const APFloat &value, Location loc);

  APFloat getValue() const;

  /// This function is used to convert the value to a double, even if it loses
  /// precision.
  double getValueAsDouble() const;
  static double getValueAsDouble(APFloat val);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Float;
  }

  /// Verify the construction invariants for a double value.
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc, MLIRContext *ctx,
                               Type type, double value);
  static LogicalResult
  verifyConstructionInvariants(llvm::Optional<Location> loc, MLIRContext *ctx,
                               Type type, const APFloat &value);
};

class StringAttr : public Attribute::AttrBase<StringAttr, Attribute,
                                              detail::StringAttributeStorage> {
public:
  using Base::Base;
  using ValueType = StringRef;

  static StringAttr get(StringRef bytes, MLIRContext *context);

  StringRef getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::String;
  }
};

/// Array attributes are lists of other attributes.  They are not necessarily
/// type homogenous given that attributes don't, in general, carry types.
class ArrayAttr : public Attribute::AttrBase<ArrayAttr, Attribute,
                                             detail::ArrayAttributeStorage> {
public:
  using Base::Base;
  using ValueType = ArrayRef<Attribute>;

  static ArrayAttr get(ArrayRef<Attribute> value, MLIRContext *context);

  ArrayRef<Attribute> getValue() const;

  size_t size() const { return getValue().size(); }

  using iterator = llvm::ArrayRef<Attribute>::iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Array;
  }
};

class AffineMapAttr
    : public Attribute::AttrBase<AffineMapAttr, Attribute,
                                 detail::AffineMapAttributeStorage> {
public:
  using Base::Base;
  using ValueType = AffineMap;

  static AffineMapAttr get(AffineMap value);

  AffineMap getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::AffineMap;
  }
};

class IntegerSetAttr
    : public Attribute::AttrBase<IntegerSetAttr, Attribute,
                                 detail::IntegerSetAttributeStorage> {
public:
  using Base::Base;
  using ValueType = IntegerSet;

  static IntegerSetAttr get(IntegerSet value);

  IntegerSet getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::IntegerSet;
  }
};

class TypeAttr : public Attribute::AttrBase<TypeAttr, Attribute,
                                            detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = Type;

  static TypeAttr get(Type value);

  Type getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == StandardAttributes::Type; }
};

/// A function attribute represents a reference to a function object.
///
/// When working with IR, it is important to know that a function attribute can
/// exist with a null Function inside of it, which occurs when a function object
/// is deleted that had an attribute which referenced it.  No references to this
/// attribute should persist across the transformation, but that attribute will
/// remain in MLIRContext.
class FunctionAttr
    : public Attribute::AttrBase<FunctionAttr, Attribute,
                                 detail::FunctionAttributeStorage> {
public:
  using Base::Base;
  using ValueType = Function *;

  static FunctionAttr get(Function *value);

  Function *getValue() const;

  FunctionType getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Function;
  }

  /// This function is used by the internals of the Function class to null out
  /// attributes referring to functions that are about to be deleted.
  static void dropFunctionReference(Function *value);
};

/// A base attribute that represents a reference to a vector or tensor constant.
class ElementsAttr : public Attribute {
public:
  using Attribute::Attribute;

  VectorOrTensorType getType() const;

  /// Return the value at the given index. If index does not refer to a valid
  /// element, then a null attribute is returned.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind >= StandardAttributes::FIRST_ELEMENTS_ATTR &&
           kind <= StandardAttributes::LAST_ELEMENTS_ATTR;
  }
};

/// An attribute that represents a reference to a splat vecctor or tensor
/// constant, meaning all of the elements have the same value.
class SplatElementsAttr
    : public Attribute::AttrBase<SplatElementsAttr, ElementsAttr,
                                 detail::SplatElementsAttributeStorage> {
public:
  using Base::Base;
  using ValueType = Attribute;

  static SplatElementsAttr get(VectorOrTensorType type, Attribute elt);
  Attribute getValue() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::SplatElements;
  }
};

/// An attribute that represents a reference to a dense vector or tensor object.
///
class DenseElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;
  using ImplType = detail::DenseElementsAttributeStorage;

  /// It assumes the elements in the input array have been truncated to the bits
  /// width specified by the element type.
  static DenseElementsAttr get(VectorOrTensorType type, ArrayRef<char> data);

  // Constructs a dense elements attribute from an array of element values. Each
  // element attribute value is expected to be an element of 'type'.
  static DenseElementsAttr get(VectorOrTensorType type,
                               ArrayRef<Attribute> values);

  /// Returns the number of elements held by this attribute.
  size_t size() const;

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
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::DenseIntElements ||
           kind == StandardAttributes::DenseFPElements;
  }

protected:
  /// A utility iterator that allows walking over the internal raw APInt values.
  class RawElementIterator
      : public llvm::iterator_facade_base<RawElementIterator,
                                          std::bidirectional_iterator_tag,
                                          APInt, std::ptrdiff_t, APInt, APInt> {
  public:
    /// Iterator movement.
    RawElementIterator &operator++() {
      ++index;
      return *this;
    }
    RawElementIterator &operator--() {
      --index;
      return *this;
    }

    /// Accesses the raw APInt value at this iterator position.
    APInt operator*() const;

    /// Iterator equality.
    bool operator==(const RawElementIterator &rhs) const {
      return rawData == rhs.rawData && index == rhs.index;
    }
    bool operator!=(const RawElementIterator &rhs) const {
      return !(*this == rhs);
    }

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    RawElementIterator(DenseElementsAttr attr, size_t index);

    /// The base address of the raw data buffer.
    const char *rawData;

    /// The current element index.
    size_t index;

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// Raw element iterators for this attribute.
  RawElementIterator raw_begin() const { return RawElementIterator(*this, 0); }
  RawElementIterator raw_end() const {
    return RawElementIterator(*this, size());
  }

  // Constructs a dense elements attribute from an array of raw APInt values.
  // Each APInt value is expected to have the same bitwidth as the element type
  // of 'type'.
  static DenseElementsAttr get(VectorOrTensorType type, ArrayRef<APInt> values);
};

/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr
    : public Attribute::AttrBase<DenseIntElementsAttr, DenseElementsAttr,
                                 detail::DenseElementsAttributeStorage> {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::RawElementIterator;

  using Base::Base;
  using DenseElementsAttr::get;
  using DenseElementsAttr::getValues;

  /// Constructs a dense integer elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'.
  static DenseIntElementsAttr get(VectorOrTensorType type,
                                  ArrayRef<APInt> values);

  /// Constructs a dense integer elements attribute from an array of integer
  /// values. Each value is expected to be within the bitwidth of the element
  /// type of 'type'.
  static DenseIntElementsAttr get(VectorOrTensorType type,
                                  ArrayRef<int64_t> values);

  /// Gets the integer value of each of the dense elements.
  void getValues(SmallVectorImpl<APInt> &values) const;

  /// Iterator access to the integer element values.
  iterator begin() const { return raw_begin(); }
  iterator end() const { return raw_end(); }

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::DenseIntElements;
  }
};

/// An attribute that represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr
    : public Attribute::AttrBase<DenseFPElementsAttr, DenseElementsAttr,
                                 detail::DenseElementsAttributeStorage> {
public:
  /// DenseFPElementsAttr iterates on APFloat, so we need to wrap the raw
  /// element iterator.
  class ElementIterator final
      : public llvm::mapped_iterator<RawElementIterator,
                                     std::function<APFloat(const APInt &)>> {
    friend DenseFPElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    ElementIterator(const llvm::fltSemantics &smt, RawElementIterator it);
  };
  using iterator = ElementIterator;

  using Base::Base;
  using DenseElementsAttr::get;
  using DenseElementsAttr::getValues;

  // Constructs a dense float elements attribute from an array of APFloat
  // values. Each APFloat value is expected to have the same bitwidth as the
  // element type of 'type'.
  static DenseFPElementsAttr get(VectorOrTensorType type,
                                 ArrayRef<APFloat> values);

  /// Gets the float value of each of the dense elements.
  void getValues(SmallVectorImpl<APFloat> &values) const;

  /// Iterator access to the float element values.
  iterator begin() const;
  iterator end() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::DenseFPElements;
  }
};

/// An opaque attribute that represents a reference to a vector or tensor
/// constant with opaque content. This respresentation is for tensor constants
/// which the compiler may not need to interpret. This attribute is always
/// associated with a particular dialect, which provides a method to convert
/// tensor representation to a non-opaque format.
class OpaqueElementsAttr
    : public Attribute::AttrBase<OpaqueElementsAttr, ElementsAttr,
                                 detail::OpaqueElementsAttributeStorage> {
public:
  using Base::Base;
  using ValueType = StringRef;

  static OpaqueElementsAttr get(Dialect *dialect, VectorOrTensorType type,
                                StringRef bytes);

  StringRef getValue() const;

  /// Return the value at the given index. If index does not refer to a valid
  /// element, then a null attribute is returned.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Decodes the attribute value using dialect-specific decoding hook.
  /// Returns false if decoding is successful. If not, returns true and leaves
  /// 'result' argument unspecified.
  bool decode(ElementsAttr &result);

  /// Returns dialect associated with this opaque constant.
  Dialect *getDialect() const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::OpaqueElements;
  }
};

/// An attribute that represents a reference to a sparse vector or tensor
/// object.
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
class SparseElementsAttr
    : public Attribute::AttrBase<SparseElementsAttr, ElementsAttr,
                                 detail::SparseElementsAttributeStorage> {
public:
  using Base::Base;

  static SparseElementsAttr get(VectorOrTensorType type,
                                DenseIntElementsAttr indices,
                                DenseElementsAttr values);

  DenseIntElementsAttr getIndices() const;

  DenseElementsAttr getValues() const;

  /// Return the value of the element at the given index.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::SparseElements;
  }
};

template <typename U> bool Attribute::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::kindof(getKind());
}
template <typename U> U Attribute::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::cast() const {
  assert(isa<U>());
  return U(impl);
}

// Make Attribute hashable.
inline ::llvm::hash_code hash_value(Attribute arg) {
  return ::llvm::hash_value(arg.impl);
}

/// NamedAttribute is used for named attribute lists, it holds an identifier for
/// the name and a value for the attribute. The attribute pointer should always
/// be non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

/// A NamedAttributeList is used to manage a list of named attributes. This
/// provides simple interfaces for adding/removing/finding attributes from
/// within a raw AttributeListStorage.
///
/// We assume there will be relatively few attributes on a given function
/// (maybe a dozen or so, but not hundreds or thousands) so we use linear
/// searches for everything.
class NamedAttributeList {
public:
  NamedAttributeList() : attrs(nullptr) {}
  NamedAttributeList(ArrayRef<NamedAttribute> attributes);

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() const;

  /// Replace the held attributes with ones provided in 'newAttrs'.
  void setAttrs(ArrayRef<NamedAttribute> attributes);

  /// Return the specified attribute if present, null otherwise.
  Attribute get(StringRef name) const;
  Attribute get(Identifier name) const;

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void set(Identifier name, Attribute value);

  enum class RemoveResult { Removed, NotFound };

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  RemoveResult remove(Identifier name);

private:
  detail::AttributeListStorage *attrs;
};

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
