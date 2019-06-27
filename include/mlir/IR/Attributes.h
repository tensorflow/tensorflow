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
class FunctionType;
class Identifier;
class IntegerSet;
class Location;
class MLIRContext;
class ShapedType;
class Type;

namespace detail {

struct AffineMapAttributeStorage;
struct ArrayAttributeStorage;
struct BoolAttributeStorage;
struct DictionaryAttributeStorage;
struct IntegerAttributeStorage;
struct IntegerSetAttributeStorage;
struct FloatAttributeStorage;
struct OpaqueAttributeStorage;
struct StringAttributeStorage;
struct TypeAttributeStorage;

/// Elements Attributes.
struct DenseElementsAttributeStorage;
struct OpaqueElementsAttributeStorage;
struct SparseElementsAttributeStorage;
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
  static bool classof(Attribute) { return true; }

  /// Return the classification for this attribute.
  unsigned getKind() const { return impl->getKind(); }

  /// Return the type of this attribute.
  Type getType() const;

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  Dialect &getDialect() const;

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Get an opaque pointer to the attribute.
  const void *getAsOpaquePointer() const { return impl; }
  /// Construct an attribute from the opaque pointer representation.
  static Attribute getFromOpaquePointer(const void *ptr) {
    return Attribute(reinterpret_cast<const ImplType *>(ptr));
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
  AffineMap = Attribute::FIRST_STANDARD_ATTR,
  Array,
  Bool,
  Dictionary,
  Float,
  Function,
  Integer,
  IntegerSet,
  Opaque,
  String,
  Type,
  Unit,

  /// Elements Attributes.
  DenseElements,
  OpaqueElements,
  SparseElements,
  FIRST_ELEMENTS_ATTR = DenseElements,
  LAST_ELEMENTS_ATTR = SparseElements,

  /// Locations.
  CallSiteLocation,
  FileLineColLocation,
  FusedLocation,
  NameLocation,
  UnknownLocation,

  // Represents a location as a 'void*' pointer to a front-end's opaque
  // location information, which must live longer than the MLIR objects that
  // refer to it.  OpaqueLocation's are never serialized.
  //
  // TODO: OpaqueLocation,

  // Represents a value inlined through a function call.
  // TODO: InlinedLocation,

  FIRST_LOCATION_ATTR = CallSiteLocation,
  LAST_LOCATION_ATTR = UnknownLocation,
};
} // namespace StandardAttributes

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

/// Array attributes are lists of other attributes.  They are not necessarily
/// type homogenous given that attributes don't, in general, carry types.
class ArrayAttr : public Attribute::AttrBase<ArrayAttr, Attribute,
                                             detail::ArrayAttributeStorage> {
public:
  using Base::Base;
  using ValueType = ArrayRef<Attribute>;

  static ArrayAttr get(ArrayRef<Attribute> value, MLIRContext *context);

  ArrayRef<Attribute> getValue() const;

  /// Support range iteration.
  using iterator = llvm::ArrayRef<Attribute>::iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }
  size_t size() const { return getValue().size(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Array;
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

/// NamedAttribute is used for dictionary attributes, it holds an identifier for
/// the name and a value for the attribute. The attribute pointer should always
/// be non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

/// Dictionary attribute is an attribute that represents a sorted collection of
/// named attribute values. The elements are sorted by name, and each name must
/// be unique within the collection.
class DictionaryAttr
    : public Attribute::AttrBase<DictionaryAttr, Attribute,
                                 detail::DictionaryAttributeStorage> {
public:
  using Base::Base;
  using ValueType = ArrayRef<NamedAttribute>;

  static DictionaryAttr get(ArrayRef<NamedAttribute> value,
                            MLIRContext *context);

  ArrayRef<NamedAttribute> getValue() const;

  /// Return the specified attribute if present, null otherwise.
  Attribute get(StringRef name) const;
  Attribute get(Identifier name) const;

  /// Support range iteration.
  using iterator = llvm::ArrayRef<NamedAttribute>::iterator;
  iterator begin() const;
  iterator end() const;
  bool empty() const { return size() == 0; }
  size_t size() const;

  /// Methods for supporting type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Dictionary;
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

/// A function attribute represents a reference to a function object.
class FunctionAttr
    : public Attribute::AttrBase<FunctionAttr, Attribute,
                                 detail::StringAttributeStorage> {
public:
  using Base::Base;
  using ValueType = Function *;

  static FunctionAttr get(Function *value);
  static FunctionAttr get(StringRef value, MLIRContext *ctx);

  /// Returns the name of the held function reference.
  StringRef getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == StandardAttributes::Function;
  }
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

/// Unit attributes are attributes that hold no specific value and are given
/// meaning by their existence.
class UnitAttr : public Attribute::AttrBase<UnitAttr> {
public:
  using Base::Base;

  static UnitAttr get(MLIRContext *context);

  static bool kindof(unsigned kind) { return kind == StandardAttributes::Unit; }
};

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

/// A base attribute that represents a reference to a static shaped tensor or
/// vector constant.
class ElementsAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Return the type of this ElementsAttr, guaranteed to be a vector or tensor
  /// with static shape.
  ShapedType getType() const;

  /// Return the value at the given index. If index does not refer to a valid
  /// element, then a null attribute is returned.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Generates a new ElementsAttr by mapping each int value to a new
  /// underlying APInt. The new values can represent either a integer or float.
  /// This ElementsAttr should contain integers.
  ElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APInt &)> mapping) const;

  /// Generates a new ElementsAttr by mapping each float value to a new
  /// underlying APInt. The new values can represent either a integer or float.
  /// This ElementsAttr should contain floats.
  ElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APFloat &)> mapping) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    return attr.getKind() >= StandardAttributes::FIRST_ELEMENTS_ATTR &&
           attr.getKind() <= StandardAttributes::LAST_ELEMENTS_ATTR;
  }
};

/// An attribute that represents a reference to a dense vector or tensor object.
///
class DenseElementsAttr
    : public Attribute::AttrBase<DenseElementsAttr, ElementsAttr,
                                 detail::DenseElementsAttributeStorage> {
public:
  using Base::Base;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    return attr.getKind() == StandardAttributes::DenseElements;
  }

  /// Constructs a dense elements attribute from an array of element values.
  /// Each element attribute value is expected to be an element of 'type'.
  /// 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<Attribute> values);

  /// Constructs a dense integer elements attribute from an array of integer
  /// or floating-point values. Each value is expected to be the same bitwidth
  /// of the element type of 'type'. 'type' must be a vector or tensor with
  /// static shape.
  template <typename T, typename = typename std::enable_if<
                            std::numeric_limits<T>::is_integer ||
                            llvm::is_one_of<T, float, double>::value>::type>
  static DenseElementsAttr get(const ShapedType &type, ArrayRef<T> values) {
    const char *data = reinterpret_cast<const char *>(values.data());
    return getRawIntOrFloat(
        type, ArrayRef<char>(data, values.size() * sizeof(T)), sizeof(T),
        /*isInt=*/std::numeric_limits<T>::is_integer);
  }

  /// Constructs a dense integer elements attribute from a single element.
  template <typename T, typename = typename std::enable_if<
                            std::numeric_limits<T>::is_integer ||
                            llvm::is_one_of<T, float, double>::value>::type>
  static DenseElementsAttr get(const ShapedType &type, T value) {
    return get(type, llvm::makeArrayRef(value));
  }

  /// Overload of the above 'get' method that is specialized for boolean values.
  static DenseElementsAttr get(ShapedType type, ArrayRef<bool> values);

  /// Constructs a dense integer elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APInt> values);

  /// Constructs a dense float elements attribute from an array of APFloat
  /// values. Each APFloat value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APFloat> values);

  /// Construct a dense elements attribute for an initializer_list of values.
  /// Each value is expected to be the same bitwidth of the element type of
  /// 'type'. 'type' must be a vector or tensor with static shape.
  template <typename T>
  static DenseElementsAttr get(const ShapedType &type,
                               const std::initializer_list<T> &list) {
    return get(type, ArrayRef<T>(list));
  }

  //===--------------------------------------------------------------------===//
  // Iterators
  //===--------------------------------------------------------------------===//

  /// A utility iterator that allows walking over the internal Attribute values
  /// of a DenseElementsAttr.
  class AttributeElementIterator
      : public indexed_accessor_iterator<AttributeElementIterator, const void *,
                                         Attribute, Attribute, Attribute> {
  public:
    /// Accesses the Attribute value at this iterator position.
    Attribute operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    AttributeElementIterator(DenseElementsAttr attr, size_t index);
  };

  /// A utility iterator that allows walking over the internal raw APInt values.
  class IntElementIterator
      : public indexed_accessor_iterator<IntElementIterator, const char *,
                                         APInt, APInt, APInt> {
  public:
    /// Accesses the raw APInt value at this iterator position.
    APInt operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    IntElementIterator(DenseElementsAttr attr, size_t index);

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// Iterator for walking over APFloat values.
  class FloatElementIterator final
      : public llvm::mapped_iterator<IntElementIterator,
                                     std::function<APFloat(const APInt &)>> {
    friend DenseElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    FloatElementIterator(const llvm::fltSemantics &smt, IntElementIterator it);

  public:
    using reference = APFloat;
  };

  //===--------------------------------------------------------------------===//
  // Value Querying
  //===--------------------------------------------------------------------===//

  /// Returns the number of raw elements held by this attribute.
  size_t rawSize() const;

  /// Returns if this attribute corresponds to a splat, i.e. if all element
  /// values are the same.
  bool isSplat() const;

  /// If this attribute corresponds to a splat, then get the splat value.
  /// Otherwise, return null.
  Attribute getSplatValue() const;

  /// Return the value at the given index. If index does not refer to a valid
  /// element, then a null attribute is returned.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Return the held element values as an array of integer or floating-point
  /// values.
  template <typename T, typename = typename std::enable_if<
                            (!std::is_same<T, bool>::value &&
                             std::numeric_limits<T>::is_integer) ||
                            llvm::is_one_of<T, float, double>::value>::type>
  ArrayRef<T> getValues() const {
    assert(isValidIntOrFloat(sizeof(T), std::numeric_limits<T>::is_integer));
    auto rawData = getRawData();
    return ArrayRef<T>(reinterpret_cast<const T *>(rawData.data()),
                       rawData.size() / sizeof(T));
  }

  /// Return the held element values as a range of Attributes.
  llvm::iterator_range<AttributeElementIterator> getAttributeValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, Attribute>::value>::type>
  llvm::iterator_range<AttributeElementIterator> getValues() const {
    return getAttributeValues();
  }
  AttributeElementIterator attr_value_begin() const;
  AttributeElementIterator attr_value_end() const;

  /// Return the held element values as a range of APInts. The element type of
  /// this attribute must be of integer type.
  llvm::iterator_range<IntElementIterator> getIntValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, APInt>::value>::type>
  llvm::iterator_range<IntElementIterator> getValues() const {
    return getIntValues();
  }
  IntElementIterator int_value_begin() const;
  IntElementIterator int_value_end() const;

  /// Return the held element values as a range of APFloat. The element type of
  /// this attribute must be of float type.
  llvm::iterator_range<FloatElementIterator> getFloatValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, APFloat>::value>::type>
  llvm::iterator_range<FloatElementIterator> getValues() const {
    return getFloatValues();
  }
  FloatElementIterator float_value_begin() const;
  FloatElementIterator float_value_end() const;

  //===--------------------------------------------------------------------===//
  // Mutation Utilities
  //===--------------------------------------------------------------------===//

  /// Return a new DenseElementsAttr that has the same data as the current
  /// attribute, but has been reshaped to 'newType'. The new type must have the
  /// same total number of elements as well as element type.
  DenseElementsAttr reshape(ShapedType newType);

  /// Generates a new DenseElementsAttr by mapping each int value to a new
  /// underlying APInt. The new values can represent either a integer or float.
  /// This underlying type must be an DenseIntElementsAttr.
  DenseElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APInt &)> mapping) const;

  /// Generates a new DenseElementsAttr by mapping each float value to a new
  /// underlying APInt. the new values can represent either a integer or float.
  /// This underlying type must be an DenseFPElementsAttr.
  DenseElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APFloat &)> mapping) const;

protected:
  /// Return the raw storage data held by this attribute.
  ArrayRef<char> getRawData() const;

  /// Get iterators to the raw APInt values for each element in this attribute.
  IntElementIterator raw_int_begin() const {
    return IntElementIterator(*this, 0);
  }
  IntElementIterator raw_int_end() const {
    return IntElementIterator(*this, rawSize());
  }

  /// Constructs a dense elements attribute from an array of raw APInt values.
  /// Each APInt value is expected to have the same bitwidth as the element type
  /// of 'type'. 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr getRaw(ShapedType type, ArrayRef<APInt> values);

  /// Get or create a new dense elements attribute instance with the given raw
  /// data buffer. 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr getRaw(ShapedType type, ArrayRef<char> data,
                                  bool isSplat);

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// integer or floating-point type. This method is used to verify type
  /// invariants that the templatized 'get' method cannot.
  static DenseElementsAttr getRawIntOrFloat(ShapedType type,
                                            ArrayRef<char> data,
                                            int64_t dataEltSize, bool isInt);

  /// Check the information for a c++ data type, check if this type is valid for
  /// the current attribute. This method is used to verify specific type
  /// invariants that the templatized 'getValues' method cannot.
  bool isValidIntOrFloat(int64_t dataEltSize, bool isInt) const;
};

/// An attribute that represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseElementsAttr {
public:
  using iterator = DenseElementsAttr::FloatElementIterator;

  using DenseElementsAttr::DenseElementsAttr;

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APFloat &)> mapping) const;

  /// Iterator access to the float element values.
  iterator begin() const { return float_value_begin(); }
  iterator end() const { return float_value_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseElementsAttr {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::IntElementIterator;

  using DenseElementsAttr::DenseElementsAttr;

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr
  mapValues(Type newElementType,
            llvm::function_ref<APInt(const APInt &)> mapping) const;

  /// Iterator access to the integer element values.
  iterator begin() const { return raw_int_begin(); }
  iterator end() const { return raw_int_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
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

  static OpaqueElementsAttr get(Dialect *dialect, ShapedType type,
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

  /// 'type' must be a vector or tensor with static shape.
  static SparseElementsAttr get(ShapedType type, DenseElementsAttr indices,
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

/// An attribute that represents a reference to a splat vector or tensor
/// constant, meaning all of the elements have the same value.
class SplatElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    auto denseAttr = attr.dyn_cast<DenseElementsAttr>();
    return denseAttr && denseAttr.isSplat();
  }
};

template <typename U> bool Attribute::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::classof(*this);
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

/// A NamedAttributeList is used to manage a list of named attributes. This
/// provides simple interfaces for adding/removing/finding attributes from
/// within a DictionaryAttr.
///
/// We assume there will be relatively few attributes on a given operation
/// (maybe a dozen or so, but not hundreds or thousands) so we use linear
/// searches for everything.
class NamedAttributeList {
public:
  NamedAttributeList(DictionaryAttr attrs = nullptr)
      : attrs((attrs && !attrs.empty()) ? attrs : nullptr) {}
  NamedAttributeList(ArrayRef<NamedAttribute> attributes);

  /// Return the underlying dictionary attribute. This may be null, if this list
  /// has no attributes.
  DictionaryAttr getDictionary() const { return attrs; }

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
  DictionaryAttr attrs;
};

} // end namespace mlir.

namespace llvm {

// Attribute hash just like pointers.
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

/// Allow LLVM to steal the low bits of Attributes.
template <> struct PointerLikeTypeTraits<mlir::Attribute> {
public:
  static inline void *getAsVoidPointer(mlir::Attribute attr) {
    return const_cast<void *>(attr.getAsOpaquePointer());
  }
  static inline mlir::Attribute getFromVoidPointer(void *ptr) {
    return mlir::Attribute::getFromOpaquePointer(ptr);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif
