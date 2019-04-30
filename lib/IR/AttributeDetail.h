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
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {
/// Base storage class appearing in an attribute.
struct AttributeStorage : public StorageUniquer::BaseStorage {
  AttributeStorage(bool isOrContainsFunctionCache = false)
      : isOrContainsFunctionCache(isOrContainsFunctionCache) {}

  /// This field is true if this is, or contains, a function attribute.
  bool isOrContainsFunctionCache : 1;
};

// A utility class to get, or create, unique instances of attributes within an
// MLIRContext. This class manages all creation and uniquing of attributes.
class AttributeUniquer {
public:
  /// Get an uniqued instance of attribute T. This overload is used for
  /// derived attributes that have complex storage or uniquing constraints.
  template <typename T, typename... Args>
  static typename std::enable_if<
      !std::is_same<typename T::ImplType, AttributeStorage>::value, T>::type
  get(MLIRContext *ctx, Attribute::Kind kind, Args &&... args) {
    return ctx->getAttributeUniquer().getComplex<typename T::ImplType>(
        /*initFn=*/{}, static_cast<unsigned>(kind),
        std::forward<Args>(args)...);
  }

  /// Get an uniqued instance of attribute T. This overload is used for
  /// derived attributes that use the AttributeStorage directly and thus need no
  /// additional storage or uniquing.
  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_same<typename T::ImplType, AttributeStorage>::value, T>::type
  get(MLIRContext *ctx, Attribute::Kind kind) {
    return ctx->getAttributeUniquer().getSimple<AttributeStorage>(
        /*initFn=*/{}, static_cast<unsigned>(kind));
  }

  /// Erase a uniqued instance of attribute T. This overload is used for
  /// derived attributes that have complex storage or uniquing constraints.
  template <typename T, typename... Args>
  static typename std::enable_if<
      !std::is_same<typename T::ImplType, AttributeStorage>::value>::type
  erase(MLIRContext *ctx, Attribute::Kind kind, Args &&... args) {
    return ctx->getAttributeUniquer().eraseComplex<typename T::ImplType>(
        static_cast<unsigned>(kind), std::forward<Args>(args)...);
  }
};

using AttributeStorageAllocator = StorageUniquer::StorageAllocator;

/// An attribute representing a boolean value.
struct BoolAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<MLIRContext *, bool>;

  BoolAttributeStorage(Type type, bool value) : type(type), value(value) {}

  /// We only check equality for and hash with the boolean key parameter.
  bool operator==(const KeyTy &key) const { return key.second == value; }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_value(key.second);
  }

  static BoolAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<BoolAttributeStorage>())
        BoolAttributeStorage(IntegerType::get(1, key.first), key.second);
  }

  Type type;
  bool value;
};

/// An attribute representing a integral value.
struct IntegerAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<IntegerAttributeStorage, uint64_t> {
  using KeyTy = std::pair<Type, APInt>;

  IntegerAttributeStorage(Type type, size_t numObjects)
      : type(type), numObjects(numObjects) {
    assert((type.isIndex() || type.isa<IntegerType>()) && "invalid type");
  }

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(type, getValue());
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, llvm::hash_value(key.second));
  }

  /// Construct a new storage instance.
  static IntegerAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    Type type;
    APInt value;
    std::tie(type, value) = key;

    auto elements = ArrayRef<uint64_t>(value.getRawData(), value.getNumWords());
    auto size =
        IntegerAttributeStorage::totalSizeToAlloc<uint64_t>(elements.size());
    auto rawMem = allocator.allocate(size, alignof(IntegerAttributeStorage));
    auto result = ::new (rawMem) IntegerAttributeStorage(type, elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(),
                            result->getTrailingObjects<uint64_t>());
    return result;
  }

  /// Returns an APInt representing the stored value.
  APInt getValue() const {
    if (type.isIndex())
      return APInt(64, {getTrailingObjects<uint64_t>(), numObjects});
    return APInt(type.getIntOrFloatBitWidth(),
                 {getTrailingObjects<uint64_t>(), numObjects});
  }

  Type type;
  size_t numObjects;
};

/// An attribute representing a floating point value.
struct FloatAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<FloatAttributeStorage, uint64_t> {
  using KeyTy = std::pair<Type, APFloat>;

  FloatAttributeStorage(const llvm::fltSemantics &semantics, Type type,
                        size_t numObjects)
      : semantics(semantics), type(type.cast<FloatType>()),
        numObjects(numObjects) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key.first == type && key.second.bitwiseIsEqual(getValue());
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, llvm::hash_value(key.second));
  }

  /// Construct a new storage instance.
  static FloatAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    const auto &apint = key.second.bitcastToAPInt();

    // Here one word's bitwidth equals to that of uint64_t.
    auto elements = ArrayRef<uint64_t>(apint.getRawData(), apint.getNumWords());

    auto byteSize =
        FloatAttributeStorage::totalSizeToAlloc<uint64_t>(elements.size());
    auto rawMem = allocator.allocate(byteSize, alignof(FloatAttributeStorage));
    auto result = ::new (rawMem) FloatAttributeStorage(
        key.second.getSemantics(), key.first, elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(),
                            result->getTrailingObjects<uint64_t>());
    return result;
  }

  /// Returns an APFloat representing the stored value.
  APFloat getValue() const {
    auto val = APInt(APFloat::getSizeInBits(semantics),
                     {getTrailingObjects<uint64_t>(), numObjects});
    return APFloat(semantics, val);
  }

  const llvm::fltSemantics &semantics;
  FloatType type;
  size_t numObjects;
};

/// An attribute representing a string value.
struct StringAttributeStorage : public AttributeStorage {
  using KeyTy = StringRef;

  StringAttributeStorage(StringRef value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static StringAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<StringAttributeStorage>())
        StringAttributeStorage(allocator.copyInto(key));
  }

  StringRef value;
};

/// An attribute representing an array of other attributes.
struct ArrayAttributeStorage : public AttributeStorage {
  using KeyTy = ArrayRef<Attribute>;

  ArrayAttributeStorage(bool hasFunctionAttr, ArrayRef<Attribute> value)
      : AttributeStorage(hasFunctionAttr), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static ArrayAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    // Check to see if any of the elements have a function attr.
    bool hasFunctionAttr = llvm::any_of(
        key, [](Attribute elt) { return elt.isOrContainsFunction(); });

    // Initialize the memory using placement new.
    return new (allocator.allocate<ArrayAttributeStorage>())
        ArrayAttributeStorage(hasFunctionAttr, allocator.copyInto(key));
  }

  ArrayRef<Attribute> value;
};

// An attribute representing a reference to an affine map.
struct AffineMapAttributeStorage : public AttributeStorage {
  using KeyTy = AffineMap;

  AffineMapAttributeStorage(AffineMap value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static AffineMapAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<AffineMapAttributeStorage>())
        AffineMapAttributeStorage(key);
  }

  AffineMap value;
};

// An attribute representing a reference to an integer set.
struct IntegerSetAttributeStorage : public AttributeStorage {
  using KeyTy = IntegerSet;

  IntegerSetAttributeStorage(IntegerSet value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static IntegerSetAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<IntegerSetAttributeStorage>())
        IntegerSetAttributeStorage(key);
  }

  IntegerSet value;
};

/// An attribute representing a reference to a type.
struct TypeAttributeStorage : public AttributeStorage {
  using KeyTy = Type;

  TypeAttributeStorage(Type value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static TypeAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                         KeyTy key) {
    return new (allocator.allocate<TypeAttributeStorage>())
        TypeAttributeStorage(key);
  }

  Type value;
};

/// An attribute representing a reference to a function.
struct FunctionAttributeStorage : public AttributeStorage {
  using KeyTy = Function *;

  FunctionAttributeStorage(Function *value)
      : AttributeStorage(/*isOrContainsFunctionCache=*/true), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static FunctionAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<FunctionAttributeStorage>())
        FunctionAttributeStorage(key);
  }

  /// Storage cleanup function.
  void cleanup() {
    // Null out the function reference in the attribute to avoid dangling
    // pointers.
    value = nullptr;
  }

  Function *value;
};

/// A base attribute representing a reference to a vector or tensor constant.
struct ElementsAttributeStorage : public AttributeStorage {
  ElementsAttributeStorage(VectorOrTensorType type) : type(type) {}
  VectorOrTensorType type;
};

/// An attribute representing a reference to a vector or tensor constant,
/// inwhich all elements have the same value.
struct SplatElementsAttributeStorage : public ElementsAttributeStorage {
  using KeyTy = std::pair<VectorOrTensorType, Attribute>;

  SplatElementsAttributeStorage(VectorOrTensorType type, Attribute elt)
      : ElementsAttributeStorage(type), elt(elt) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == std::make_pair(type, elt);
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Construct a new storage instance.
  static SplatElementsAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<SplatElementsAttributeStorage>())
        SplatElementsAttributeStorage(key.first, key.second);
  }

  Attribute elt;
};

/// An attribute representing a reference to a dense vector or tensor object.
struct DenseElementsAttributeStorage : public ElementsAttributeStorage {
  using KeyTy = std::pair<VectorOrTensorType, ArrayRef<char>>;

  DenseElementsAttributeStorage(VectorOrTensorType ty, ArrayRef<char> data)
      : ElementsAttributeStorage(ty), data(data) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const { return key == KeyTy(type, data); }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Construct a new storage instance.
  static DenseElementsAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator.
    ArrayRef<char> data = key.second;
    if (!data.empty()) {
      // Rounding up the allocate size to multiples of APINT_WORD_SIZE, so
      // the `readBits` will not fail when it accesses multiples of
      // APINT_WORD_SIZE each time.
      size_t sizeToAllocate =
          llvm::alignTo(data.size(), APInt::APINT_WORD_SIZE);
      auto *rawCopy = (char *)allocator.allocate(sizeToAllocate, 64);
      std::uninitialized_copy(data.begin(), data.end(), rawCopy);
      data = {rawCopy, data.size()};
    }
    return new (allocator.allocate<DenseElementsAttributeStorage>())
        DenseElementsAttributeStorage(key.first, data);
  }

  ArrayRef<char> data;
};

/// An attribute representing a reference to a tensor constant with opaque
/// content.
struct OpaqueElementsAttributeStorage : public ElementsAttributeStorage {
  using KeyTy = std::tuple<VectorOrTensorType, Dialect *, StringRef>;

  OpaqueElementsAttributeStorage(VectorOrTensorType type, Dialect *dialect,
                                 StringRef bytes)
      : ElementsAttributeStorage(type), dialect(dialect), bytes(bytes) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(type, dialect, bytes);
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  /// Construct a new storage instance.
  static OpaqueElementsAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // TODO(b/131468830): Provide a way to avoid copying content of large opaque
    // tensors This will likely require a new reference attribute kind.
    return new (allocator.allocate<OpaqueElementsAttributeStorage>())
        OpaqueElementsAttributeStorage(std::get<0>(key), std::get<1>(key),
                                       allocator.copyInto(std::get<2>(key)));
  }

  Dialect *dialect;
  StringRef bytes;
};

/// An attribute representing a reference to a sparse vector or tensor object.
struct SparseElementsAttributeStorage : public ElementsAttributeStorage {
  using KeyTy =
      std::tuple<VectorOrTensorType, DenseIntElementsAttr, DenseElementsAttr>;

  SparseElementsAttributeStorage(VectorOrTensorType type,
                                 DenseIntElementsAttr indices,
                                 DenseElementsAttr values)
      : ElementsAttributeStorage(type), indices(indices), values(values) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(type, indices, values);
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  /// Construct a new storage instance.
  static SparseElementsAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<SparseElementsAttributeStorage>())
        SparseElementsAttributeStorage(std::get<0>(key), std::get<1>(key),
                                       std::get<2>(key));
  }

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
