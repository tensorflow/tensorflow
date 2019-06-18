//===- SPIRVTypes.cpp - MLIR SPIR-V Types ---------------------------------===//
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
// This file defines the types in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/SPIRV/SPIRVTypes.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::spirv;

// Pull in all enum utility function definitions
#include "mlir/SPIRV/SPIRVEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::ArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, int64_t>;

  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, elementCount);
  }

  ArrayTypeStorage(const KeyTy &key)
      : elementType(key.first), elementCount(key.second) {}

  Type elementType;
  int64_t elementCount;
};

ArrayType ArrayType::get(Type elementType, int64_t elementCount) {
  return Base::get(elementType.getContext(), TypeKind::Array, elementType,
                   elementCount);
}

Type ArrayType::getElementType() { return getImpl()->elementType; }

int64_t ArrayType::getElementCount() { return getImpl()->elementCount; }

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

struct spirv::detail::PointerTypeStorage : public TypeStorage {
  // (Type, StorageClass) as the key: Type stored in this struct, and
  // StorageClass stored as TypeStorage's subclass data.
  using KeyTy = std::pair<Type, StorageClass>;

  static PointerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PointerTypeStorage>())
        PointerTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(pointeeType, getStorageClass());
  }

  PointerTypeStorage(const KeyTy &key)
      : TypeStorage(static_cast<unsigned>(key.second)), pointeeType(key.first) {
  }

  StorageClass getStorageClass() const {
    return static_cast<StorageClass>(getSubclassData());
  }

  Type pointeeType;
};

PointerType PointerType::get(Type pointeeType, StorageClass storageClass) {
  return Base::get(pointeeType.getContext(), TypeKind::Pointer, pointeeType,
                   storageClass);
}

Type PointerType::getPointeeType() { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() {
  return getImpl()->getStorageClass();
}

StringRef PointerType::getStorageClassStr() {
  return stringifyStorageClass(getStorageClass());
}

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::RuntimeArrayTypeStorage : public TypeStorage {
  using KeyTy = Type;

  static RuntimeArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    return new (allocator.allocate<RuntimeArrayTypeStorage>())
        RuntimeArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return elementType == key; }

  RuntimeArrayTypeStorage(const KeyTy &key) : elementType(key) {}

  Type elementType;
};

RuntimeArrayType RuntimeArrayType::get(Type elementType) {
  return Base::get(elementType.getContext(), TypeKind::RuntimeArray,
                   elementType);
}

Type RuntimeArrayType::getElementType() { return getImpl()->elementType; }
