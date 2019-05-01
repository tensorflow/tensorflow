//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
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
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

mlir::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect("linalg", context) {
  addTypes<BufferType, RangeType, ViewType>();
  addOperations<BufferAllocOp, BufferDeallocOp, RangeOp, SliceOp, ViewOp>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Linalg/IR/LinalgOps.cpp.inc"
      >();
}

struct mlir::BufferTypeStorage : public mlir::TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  struct Key {
    Key(Type elementType) : elementType(elementType) {}
    Type elementType;
  };
  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the llvm::BumpPtrAllocator given a key.
  static BufferTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const Key &key) {
    return new (allocator.allocate<BufferTypeStorage>()) BufferTypeStorage(key);
  }

  /// Equality operator for hashing.
  bool operator==(const Key &key) const {
    return elementType == key.elementType;
  }

  /// Hashing for unique'ing.
  static unsigned hashKey(const Key &key) {
    return llvm::hash_combine(key.elementType);
  }

  Type getElementType() { return elementType; };

private:
  BufferTypeStorage(const Key &key) : elementType(key.elementType) {}

  Type elementType;
};

BufferType mlir::BufferType::get(MLIRContext *context, Type elementType) {
  return Base::get(context, LinalgTypes::Buffer, elementType);
}

Type mlir::BufferType::getElementType() { return getImpl()->getElementType(); }

Type mlir::LinalgDialect::parseType(StringRef spec, Location loc) const {
  MLIRContext *context = getContext();
  if (spec == "range")
    return RangeType::get(getContext());
  // TODO(ntv): reuse mlir Parser once exposed.
  if (spec == "buffer<f32>")
    return BufferType::get(getContext(), FloatType::getF32(getContext()));
  // TODO(ntv): reuse mlir Parser once exposed.
  if (spec.startswith("view")) {
    spec.consume_front("view");
    // Just count the number of ? to get the rank, the type must be f32 for now.
    unsigned rank = 0;
    for (unsigned i = 0, e = spec.size(); i < e; ++i)
      if (spec[i] == '?')
        ++rank;
    return ViewType::get(context, FloatType::getF32(context), rank);
  }
  return (context->emitError(loc, "unknown Linalg type: " + spec), Type());
}

struct mlir::ViewTypeStorage : public mlir::TypeStorage {
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

ViewType mlir::ViewType::get(MLIRContext *context, Type elementType,
                             unsigned rank) {
  return Base::get(context, LinalgTypes::View, elementType, rank);
}

Type mlir::ViewType::getElementType() { return getImpl()->getElementType(); }

unsigned mlir::ViewType::getRank() { return getImpl()->getRank(); }

/// BufferType prints as "buffer<element_type>".
static void print(BufferType bt, raw_ostream &os) {
  os << "buffer<" << bt.getElementType() << ">";
}

/// RangeType prints as just "range".
static void print(RangeType rt, raw_ostream &os) { os << "range"; }

/// ViewType prints as:
///
/// ```{.mlir}
///   view<?x?xf32>
/// ```
///
/// or
///
/// ```{.mlir}
///   view<?xf32>
/// ```
///
/// for 0-D views (a.k.a pointer to a scalar value).
static void print(mlir::ViewType rt, raw_ostream &os) {
  os << "view<";
  for (unsigned i = 0, e = rt.getRank(); i < e; ++i) {
    os << "?x";
  }
  os << rt.getElementType();
  os << ">";
}

void mlir::LinalgDialect::printType(Type type, raw_ostream &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Linalg type");
  case LinalgTypes::Buffer:
    print(type.cast<BufferType>(), os);
    break;
  case LinalgTypes::Range:
    print(type.cast<RangeType>(), os);
    break;
  case LinalgTypes::View:
    print(type.cast<ViewType>(), os);
    break;
  }
}
