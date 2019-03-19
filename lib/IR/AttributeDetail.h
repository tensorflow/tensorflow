//===- AttributeDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
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
//
// This holds implementation details of Attribute.
//
//===----------------------------------------------------------------------===//

#ifndef ATTRIBUTEDETAIL_H_
#define ATTRIBUTEDETAIL_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

/// Base storage class appearing in an attribute.
struct AttributeStorage {
  Attribute::Kind kind : 8;

  /// This field is true if this is, or contains, a function attribute.
  bool isOrContainsFunctionCache : 1;
};

/// An attribute representing a boolean value.
struct BoolAttributeStorage : public AttributeStorage {
  BoolAttributeStorage(AttributeStorage base, Type type, bool value)
      : AttributeStorage(base), type(type), value(value) {}
  const Type type;
  bool value;
};

/// An attribute representing a integral value.
struct IntegerAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<IntegerAttributeStorage, uint64_t> {
  IntegerAttributeStorage(AttributeStorage &&as, Type type, size_t numObjects)
      : AttributeStorage(as), type(type), numObjects(numObjects) {
    assert((type.isIndex() || type.isa<IntegerType>()) && "invalid type");
  }

  const Type type;
  size_t numObjects;

  /// Returns an APInt representing the stored value.
  APInt getValue() const {
    if (type.isIndex())
      return APInt(64, {getTrailingObjects<uint64_t>(), numObjects});
    return APInt(type.getIntOrFloatBitWidth(),
                 {getTrailingObjects<uint64_t>(), numObjects});
  }
};

/// An attribute representing a floating point value.
struct FloatAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<FloatAttributeStorage, uint64_t> {
  FloatAttributeStorage(AttributeStorage &&as,
                        const llvm::fltSemantics &semantics, Type type,
                        size_t numObjects)
      : AttributeStorage(as), semantics(semantics),
        type(type.cast<FloatType>()), numObjects(numObjects) {}
  const llvm::fltSemantics &semantics;
  const FloatType type;
  size_t numObjects;

  /// Returns an APFloat representing the stored value.
  APFloat getValue() const {
    auto val = APInt(APFloat::getSizeInBits(semantics),
                     {getTrailingObjects<uint64_t>(), numObjects});
    return APFloat(semantics, val);
  }
};

/// An attribute representing a string value.
struct StringAttributeStorage : public AttributeStorage {
  StringAttributeStorage(AttributeStorage base, StringRef value)
      : AttributeStorage(base), value(value) {}
  StringRef value;
};

/// An attribute representing an array of other attributes.
struct ArrayAttributeStorage : public AttributeStorage {
  ArrayAttributeStorage(AttributeStorage base, ArrayRef<Attribute> value)
      : AttributeStorage(base), value(value) {}
  ArrayRef<Attribute> value;
};

// An attribute representing a reference to an affine map.
struct AffineMapAttributeStorage : public AttributeStorage {
  AffineMapAttributeStorage(AttributeStorage base, AffineMap value)
      : AttributeStorage(base), value(value) {}
  AffineMap value;
};

// An attribute representing a reference to an integer set.
struct IntegerSetAttributeStorage : public AttributeStorage {
  IntegerSetAttributeStorage(AttributeStorage base, IntegerSet value)
      : AttributeStorage(base), value(value) {}
  IntegerSet value;
};

/// An attribute representing a reference to a type.
struct TypeAttributeStorage : public AttributeStorage {
  TypeAttributeStorage(AttributeStorage base, Type value)
      : AttributeStorage(base), value(value) {}
  Type value;
};

/// An attribute representing a reference to a function.
struct FunctionAttributeStorage : public AttributeStorage {
  FunctionAttributeStorage(AttributeStorage base, Function *value)
      : AttributeStorage(base), value(value) {}
  Function *value;
};

/// A base attribute representing a reference to a vector or tensor constant.
struct ElementsAttributeStorage : public AttributeStorage {
  ElementsAttributeStorage(AttributeStorage base, VectorOrTensorType type)
      : AttributeStorage(base), type(type) {}
  VectorOrTensorType type;
};

/// An attribute representing a reference to a vector or tensor constant,
/// inwhich all elements have the same value.
struct SplatElementsAttributeStorage : public ElementsAttributeStorage {
  SplatElementsAttributeStorage(ElementsAttributeStorage base, Attribute elt)
      : ElementsAttributeStorage(base), elt(elt) {}
  Attribute elt;
};

/// An attribute representing a reference to a dense vector or tensor object.
struct DenseElementsAttributeStorage : public ElementsAttributeStorage {
  DenseElementsAttributeStorage(ElementsAttributeStorage base,
                                ArrayRef<char> data)
      : ElementsAttributeStorage(base), data(data) {}
  ArrayRef<char> data;
};

/// An attribute representing a reference to a tensor constant with opaque
/// content.
struct OpaqueElementsAttributeStorage : public ElementsAttributeStorage {
  OpaqueElementsAttributeStorage(ElementsAttributeStorage base,
                                 Dialect *dialect, StringRef bytes)
      : ElementsAttributeStorage(base), dialect(dialect), bytes(bytes) {}
  Dialect *dialect;
  StringRef bytes;
};

/// An attribute representing a reference to a sparse vector or tensor object.
struct SparseElementsAttributeStorage : public ElementsAttributeStorage {
  SparseElementsAttributeStorage(ElementsAttributeStorage base,
                                 DenseIntElementsAttr indices,
                                 DenseElementsAttr values)
      : ElementsAttributeStorage(base), indices(indices), values(values) {}
  DenseIntElementsAttr indices;
  DenseElementsAttr values;
};

/// A raw list of named attributes stored as a trailing array.
class AttributeListStorage final
    : private llvm::TrailingObjects<AttributeListStorage, NamedAttribute> {
  friend class llvm::TrailingObjects<AttributeListStorage, NamedAttribute>;

public:
  /// Given a list of NamedAttribute's, canonicalize the list (sorting
  /// by name) and return the unique'd result.  Note that the empty list is
  /// represented with a null pointer.
  static AttributeListStorage *get(ArrayRef<NamedAttribute> attrs,
                                   MLIRContext *context);

  /// Return the element constants for this aggregate constant.  These are
  /// known to all be constants.
  ArrayRef<NamedAttribute> getElements() const {
    return {getTrailingObjects<NamedAttribute>(), numElements};
  }

private:
  // This is used by the llvm::TrailingObjects base class.
  size_t numTrailingObjects(OverloadToken<NamedAttribute>) const {
    return numElements;
  }
  AttributeListStorage() = delete;
  AttributeListStorage(const AttributeListStorage &) = delete;
  AttributeListStorage(unsigned numElements) : numElements(numElements) {}

  /// This is the number of attributes.
  const unsigned numElements;
};
} // namespace detail
} // namespace mlir

#endif // ATTRIBUTEDETAIL_H_
