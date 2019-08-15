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
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

mlir::linalg::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<BufferType, RangeType, ViewType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Linalg/IR/LinalgLibraryOps.cpp.inc"
      >();
}

struct mlir::linalg::BufferTypeStorage : public TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  struct Key {
    Key(Type elementType, int64_t bufferSize = -1)
        : elementType(elementType), bufferSize(bufferSize) {}
    Type elementType;
    int64_t bufferSize;
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
    return elementType == key.elementType && bufferSize == key.bufferSize;
  }

  /// Hashing for unique'ing.
  static unsigned hashKey(const Key &key) {
    return llvm::hash_combine(key.elementType, key.bufferSize);
  }

  Type getElementType() { return elementType; }
  bool hasConstantSize() { return bufferSize >= 0; }
  Optional<int64_t> getBufferSize() {
    if (hasConstantSize()) {
      return bufferSize;
    }
    return llvm::None;
  }

private:
  BufferTypeStorage(const Key &key)
      : elementType(key.elementType), bufferSize(key.bufferSize) {}

  Type elementType;
  int64_t bufferSize;
};

BufferType mlir::linalg::BufferType::get(MLIRContext *context, Type elementType,
                                         int64_t bufferSize) {
  return Base::get(context, LinalgTypes::Buffer, elementType, bufferSize);
}

Type mlir::linalg::BufferType::getElementType() {
  return getImpl()->getElementType();
}

bool mlir::linalg::BufferType::hasConstantSize() {
  return getImpl()->hasConstantSize();
}

Optional<int64_t> mlir::linalg::BufferType::getBufferSize() {
  return getImpl()->getBufferSize();
}

Type mlir::linalg::LinalgDialect::parseType(StringRef spec,
                                            Location loc) const {
  StringRef origSpec = spec;
  MLIRContext *context = getContext();
  if (spec == "range")
    return RangeType::get(getContext());
  else if (spec.consume_front("buffer")) {
    if (spec.consume_front("<") && spec.consume_back(">")) {
      StringRef sizeSpec, typeSpec;
      std::tie(sizeSpec, typeSpec) = spec.split('x');
      if (typeSpec.empty()) {
        emitError(loc, "expected 'x' followed by element type");
        return Type();
      }
      // Check for '?'
      int64_t bufferSize = -1;
      if (!sizeSpec.consume_front("?")) {
        if (sizeSpec.consumeInteger(10, bufferSize)) {
          emitError(loc, "expected buffer size to be an unsigned integer");
          return Type();
        }
      }
      if (!sizeSpec.empty()) {
        emitError(loc, "unexpected token '") << sizeSpec << "'";
      }

      typeSpec = typeSpec.trim();
      auto t = mlir::parseType(typeSpec, context);
      if (!t) {
        emitError(loc, "invalid type specification: '") << typeSpec << "'";
        return Type();
      }
      return (bufferSize == -1 ? BufferType::get(getContext(), t)
                               : BufferType::get(getContext(), t, bufferSize));
    }
  } else if (spec.consume_front("view")) {
    if (spec.consume_front("<") && spec.consume_back(">")) {
      // Just count the number of ? to get the rank.
      unsigned rank = 0;
      for (unsigned i = 0, e = spec.size(); i < e; ++i) {
        if (spec.consume_front("?")) {
          ++rank;
          if (!spec.consume_front("x")) {
            emitError(loc, "expected a list of '?x' dimension specifiers: ")
                << spec;
            return Type();
          }
        }
      }
      if (auto t = mlir::parseType(spec, context))
        return ViewType::get(context, t, rank);
    }
  }
  return (emitError(loc, "unknown Linalg type: " + origSpec), Type());
}

struct mlir::linalg::ViewTypeStorage : public TypeStorage {
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

ViewType mlir::linalg::ViewType::get(MLIRContext *context, Type elementType,
                                     unsigned rank) {
  return Base::get(context, LinalgTypes::View, elementType, rank);
}

Type mlir::linalg::ViewType::getElementType() {
  return getImpl()->getElementType();
}

unsigned mlir::linalg::ViewType::getRank() { return getImpl()->getRank(); }

/// BufferType prints as "buffer<element_type>".
static void print(BufferType bt, raw_ostream &os) {
  os << "buffer<";
  auto bs = bt.getBufferSize();
  if (bs) {
    os << bs.getValue();
  } else {
    os << "?";
  }
  os << "x" << bt.getElementType() << ">";
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
static void print(mlir::linalg::ViewType rt, raw_ostream &os) {
  os << "view<";
  for (unsigned i = 0, e = rt.getRank(); i < e; ++i) {
    os << "?x";
  }
  os << rt.getElementType();
  os << ">";
}

void mlir::linalg::LinalgDialect::printType(Type type, raw_ostream &os) const {
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
