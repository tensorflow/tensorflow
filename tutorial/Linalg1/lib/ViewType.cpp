//===- ViewType.h - Implementation of the ViewType custom type ------------===//
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
// This file implements a custom ViewType in the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/ViewType.h"

using mlir::MLIRContext;
using mlir::Type;
using mlir::TypeStorage;
using mlir::TypeStorageAllocator;

namespace linalg {

struct ViewTypeStorage : public mlir::TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  struct Key {
    Key(Type elementType, unsigned rank)
        : elementType(elementType), rank(rank) {}
    Type elementType;
    unsigned rank;
  };
  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the llvm::BumpPtrAllocator given a key.
  static ViewTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const Key &key) {
    return new (allocator.allocate<ViewTypeStorage>()) ViewTypeStorage(key);
  }

  /// Equality operator for hashing.
  bool operator==(const Key &key) const {
    return elementType == key.elementType && rank == key.rank;
  }

  /// Hashing for unique'ing.
  static unsigned hashKey(const Key &key) {
    return llvm::hash_combine(key.elementType, key.rank);
  }

  unsigned getRank() { return rank; };
  Type getElementType() { return elementType; };

private:
  ViewTypeStorage(const Key &key)
      : elementType(key.elementType), rank(key.rank) {}

  Type elementType;
  unsigned rank;
};

ViewType linalg::ViewType::get(MLIRContext *context, Type elementType,
                               unsigned rank) {
  return Base::get(context, LinalgTypes::View, elementType, rank);
}

Type linalg::ViewType::getElementType() { return getImpl()->getElementType(); }

unsigned linalg::ViewType::getRank() { return getImpl()->getRank(); }

} // namespace linalg
