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
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {
// An attribute representing a reference to an affine map.
struct AffineMapAttributeStorage : public AttributeStorage {
  using KeyTy = AffineMap;

  AffineMapAttributeStorage(AffineMap value)
      : AttributeStorage(IndexType::get(value.getContext())), value(value) {}

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

/// An attribute representing an array of other attributes.
struct ArrayAttributeStorage : public AttributeStorage {
  using KeyTy = ArrayRef<Attribute>;

  ArrayAttributeStorage(ArrayRef<Attribute> value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static ArrayAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<ArrayAttributeStorage>())
        ArrayAttributeStorage(allocator.copyInto(key));
  }

  ArrayRef<Attribute> value;
};

/// An attribute representing a boolean value.
struct BoolAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<MLIRContext *, bool>;

  BoolAttributeStorage(Type type, bool value)
      : AttributeStorage(type), value(value) {}

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

  bool value;
};

/// An attribute representing a dictionary of sorted named attributes.
struct DictionaryAttributeStorage final
    : public AttributeStorage,
      private llvm::TrailingObjects<DictionaryAttributeStorage,
                                    NamedAttribute> {
  using KeyTy = ArrayRef<NamedAttribute>;

  /// Given a list of NamedAttribute's, canonicalize the list (sorting
  /// by name) and return the unique'd result.
  static DictionaryAttributeStorage *get(ArrayRef<NamedAttribute> attrs);

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == getElements(); }

  /// Construct a new storage instance.
  static DictionaryAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto size = DictionaryAttributeStorage::totalSizeToAlloc<NamedAttribute>(
        key.size());
    auto rawMem = allocator.allocate(size, alignof(NamedAttribute));

    // Initialize the storage and trailing attribute list.
    auto result = ::new (rawMem) DictionaryAttributeStorage(key.size());
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<NamedAttribute>());
    return result;
  }

  /// Return the elements of this dictionary attribute.
  ArrayRef<NamedAttribute> getElements() const {
    return {getTrailingObjects<NamedAttribute>(), numElements};
  }

private:
  friend class llvm::TrailingObjects<DictionaryAttributeStorage,
                                     NamedAttribute>;

  // This is used by the llvm::TrailingObjects base class.
  size_t numTrailingObjects(OverloadToken<NamedAttribute>) const {
    return numElements;
  }
  DictionaryAttributeStorage(unsigned numElements) : numElements(numElements) {}

  /// This is the number of attributes.
  const unsigned numElements;
};

/// An attribute representing a floating point value.
struct FloatAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<FloatAttributeStorage, uint64_t> {
  using KeyTy = std::pair<Type, APFloat>;

  FloatAttributeStorage(const llvm::fltSemantics &semantics, Type type,
                        size_t numObjects)
      : AttributeStorage(type), semantics(semantics), numObjects(numObjects) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key.first == getType() && key.second.bitwiseIsEqual(getValue());
  }
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, llvm::hash_value(key.second));
  }

  /// Construct a key with a type and double.
  static KeyTy getKey(Type type, double value) {
    // Treat BF16 as double because it is not supported in LLVM's APFloat.
    // TODO(b/121118307): add BF16 support to APFloat?
    if (type.isBF16() || type.isF64())
      return KeyTy(type, APFloat(value));

    // This handles, e.g., F16 because there is no APFloat constructor for it.
    bool unused;
    APFloat val(value);
    val.convert(type.cast<FloatType>().getFloatSemantics(),
                APFloat::rmNearestTiesToEven, &unused);
    return KeyTy(type, val);
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
  size_t numObjects;
};

/// An attribute representing a integral value.
struct IntegerAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<IntegerAttributeStorage, uint64_t> {
  using KeyTy = std::pair<Type, APInt>;

  IntegerAttributeStorage(Type type, size_t numObjects)
      : AttributeStorage(type), numObjects(numObjects) {
    assert((type.isIndex() || type.isa<IntegerType>()) && "invalid type");
  }

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getType(), getValue());
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
    if (getType().isIndex())
      return APInt(64, {getTrailingObjects<uint64_t>(), numObjects});
    return APInt(getType().getIntOrFloatBitWidth(),
                 {getTrailingObjects<uint64_t>(), numObjects});
  }

  size_t numObjects;
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

/// Opaque Attribute Storage and Uniquing.
struct OpaqueAttributeStorage : public AttributeStorage {
  OpaqueAttributeStorage(Identifier dialectNamespace, StringRef attrData,
                         Type type)
      : AttributeStorage(type), dialectNamespace(dialectNamespace),
        attrData(attrData) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<Identifier, StringRef, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dialectNamespace, attrData, getType());
  }

  static OpaqueAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<OpaqueAttributeStorage>())
        OpaqueAttributeStorage(std::get<0>(key),
                               allocator.copyInto(std::get<1>(key)),
                               std::get<2>(key));
  }

  // The dialect namespace.
  Identifier dialectNamespace;

  // The parser attribute data for this opaque attribute.
  StringRef attrData;
};

/// An attribute representing a string value.
struct StringAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<StringRef, Type>;

  StringAttributeStorage(StringRef value, Type type)
      : AttributeStorage(type), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(value, getType());
  }

  /// Construct a new storage instance.
  static StringAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<StringAttributeStorage>())
        StringAttributeStorage(allocator.copyInto(key.first), key.second);
  }

  StringRef value;
};

/// An attribute representing a symbol reference.
struct SymbolRefAttributeStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<SymbolRefAttributeStorage,
                                   FlatSymbolRefAttr> {
  using KeyTy = std::pair<StringRef, ArrayRef<FlatSymbolRefAttr>>;

  SymbolRefAttributeStorage(StringRef value, size_t numNestedRefs)
      : value(value), numNestedRefs(numNestedRefs) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(value, getNestedRefs());
  }

  /// Construct a new storage instance.
  static SymbolRefAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto size = SymbolRefAttributeStorage::totalSizeToAlloc<FlatSymbolRefAttr>(
        key.second.size());
    auto rawMem = allocator.allocate(size, alignof(SymbolRefAttributeStorage));
    auto result = ::new (rawMem) SymbolRefAttributeStorage(
        allocator.copyInto(key.first), key.second.size());
    std::uninitialized_copy(key.second.begin(), key.second.end(),
                            result->getTrailingObjects<FlatSymbolRefAttr>());
    return result;
  }

  /// Returns the set of nested references.
  ArrayRef<FlatSymbolRefAttr> getNestedRefs() const {
    return {getTrailingObjects<FlatSymbolRefAttr>(), numNestedRefs};
  }

  StringRef value;
  size_t numNestedRefs;
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

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

/// An attribute representing a reference to a dense vector or tensor object.
struct DenseElementsAttributeStorage : public AttributeStorage {
  struct KeyTy {
    KeyTy(ShapedType type, ArrayRef<char> data, llvm::hash_code hashCode,
          bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    /// The type of the dense elements.
    ShapedType type;

    /// The raw buffer for the data storage.
    ArrayRef<char> data;

    /// The computed hash code for the storage data.
    llvm::hash_code hashCode;

    /// A boolean that indicates if this data is a splat or not.
    bool isSplat;
  };

  DenseElementsAttributeStorage(ShapedType ty, ArrayRef<char> data,
                                bool isSplat = false)
      : AttributeStorage(ty), data(data), isSplat(isSplat) {}

  /// Compare this storage instance with the provided key.
  bool operator==(const KeyTy &key) const {
    if (key.type != getType())
      return false;

    // For boolean splats we need to explicitly check that the first bit is the
    // same. Boolean values are packed at the bit level, and even though a splat
    // is detected the rest of the bits in the first byte may differ from the
    // splat value.
    if (key.type.getElementTypeBitWidth() == 1) {
      if (key.isSplat != isSplat)
        return false;
      if (isSplat)
        return (key.data.front() & 1) == data.front();
    }

    // Otherwise, we can default to just checking the data.
    return key.data == data;
  }

  /// Construct a key from a shaped type, raw data buffer, and a flag that
  /// signals if the data is already known to be a splat. Callers to this
  /// function are expected to tag preknown splat values when possible, e.g. one
  /// element shapes.
  static KeyTy getKey(ShapedType ty, ArrayRef<char> data, bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    if (isKnownSplat)
      return KeyTy(ty, data, llvm::hash_value(data), isKnownSplat);

    // Otherwise, we need to check if the data corresponds to a splat or not.

    // Handle the simple case of only one element.
    size_t numElements = ty.getNumElements();
    assert(numElements != 1 && "splat of 1 element should already be detected");

    // Handle boolean values directly as they are packed to 1-bit.
    size_t elementWidth = ty.getElementTypeBitWidth();
    if (elementWidth == 1)
      return getKeyForBoolData(ty, data, numElements);

    // FIXME(b/121118307): using 64 bits for BF16 because it is currently stored
    // with double semantics.
    if (ty.getElementType().isBF16())
      elementWidth = 64;

    // Non 1-bit dense elements are padded to 8-bits.
    size_t storageSize = llvm::divideCeil(elementWidth, CHAR_BIT);
    assert(((data.size() / storageSize) == numElements) &&
           "data does not hold expected number of elements");

    // Create the initial hash value with just the first element.
    auto firstElt = data.take_front(storageSize);
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = storageSize, e = data.size(); i != e; i += storageSize)
      if (memcmp(data.data(), &data[i], storageSize))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, firstElt, hashVal, /*isSplat=*/true);
  }

  /// Construct a key with a set of boolean data.
  static KeyTy getKeyForBoolData(ShapedType ty, ArrayRef<char> data,
                                 size_t numElements) {
    ArrayRef<char> splatData = data;
    bool splatValue = splatData.front() & 1;

    // Helper functor to generate a KeyTy for a boolean splat value.
    auto generateSplatKey = [=] {
      return KeyTy(ty, data.take_front(1),
                   llvm::hash_value(ArrayRef<char>(splatValue ? 1 : 0)),
                   /*isSplat=*/true);
    };

    // Handle the case where the potential splat value is 1 and the number of
    // elements is non 8-bit aligned.
    size_t numOddElements = numElements % CHAR_BIT;
    if (splatValue && numOddElements != 0) {
      // Check that all bits are set in the last value.
      char lastElt = splatData.back();
      if (lastElt != llvm::maskTrailingOnes<unsigned char>(numOddElements))
        return KeyTy(ty, data, llvm::hash_value(data));

      // If this is the only element, the data is known to be a splat.
      if (splatData.size() == 1)
        return generateSplatKey();
      splatData = splatData.drop_back();
    }

    // Check that the data buffer corresponds to a splat of the proper mask.
    char mask = splatValue ? ~0 : 0;
    return llvm::all_of(splatData, [mask](char c) { return c == mask; })
               ? generateSplatKey()
               : KeyTy(ty, data, llvm::hash_value(data));
  }

  /// Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  /// Construct a new storage instance.
  static DenseElementsAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    ArrayRef<char> copy, data = key.data;
    if (!data.empty()) {
      char *rawData = reinterpret_cast<char *>(
          allocator.allocate(data.size(), alignof(uint64_t)));
      std::memcpy(rawData, data.data(), data.size());

      // If this is a boolean splat, make sure only the first bit is used.
      if (key.isSplat && key.type.getElementTypeBitWidth() == 1)
        rawData[0] &= 1;
      copy = ArrayRef<char>(rawData, data.size());
    }

    return new (allocator.allocate<DenseElementsAttributeStorage>())
        DenseElementsAttributeStorage(key.type, copy, key.isSplat);
  }

  ArrayRef<char> data;
  bool isSplat;
};

/// An attribute representing a reference to a tensor constant with opaque
/// content.
struct OpaqueElementsAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, Dialect *, StringRef>;

  OpaqueElementsAttributeStorage(Type type, Dialect *dialect, StringRef bytes)
      : AttributeStorage(type), dialect(dialect), bytes(bytes) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(getType(), dialect, bytes);
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
struct SparseElementsAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, DenseIntElementsAttr, DenseElementsAttr>;

  SparseElementsAttributeStorage(Type type, DenseIntElementsAttr indices,
                                 DenseElementsAttr values)
      : AttributeStorage(type), indices(indices), values(values) {}

  /// Key equality and hash functions.
  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(getType(), indices, values);
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
} // namespace detail
} // namespace mlir

#endif // ATTRIBUTEDETAIL_H_
