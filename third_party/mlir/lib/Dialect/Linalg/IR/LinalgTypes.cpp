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

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

mlir::linalg::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<BufferType, RangeType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc"
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

Type mlir::linalg::LinalgDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "range")
    return RangeType::get(context);

  // Handle 'buffer' types.
  if (keyword == "buffer") {
    llvm::SMLoc dimensionLoc;
    SmallVector<int64_t, 1> size;
    Type type;
    if (parser.parseLess() || parser.getCurrentLocation(&dimensionLoc) ||
        parser.parseDimensionList(size) || parser.parseType(type) ||
        parser.parseGreater())
      return Type();

    if (size.size() != 1) {
      parser.emitError(dimensionLoc, "expected single element in size list");
      return Type();
    }

    return (size.front() == -1 ? BufferType::get(context, type)
                               : BufferType::get(context, type, size.front()));
  }

  parser.emitError(parser.getNameLoc(), "unknown Linalg type: " + keyword);
  return Type();
}

/// BufferType prints as "buffer<size x element_type>".
static void print(BufferType bt, DialectAsmPrinter &os) {
  os << "buffer<";
  if (Optional<int64_t> bs = bt.getBufferSize())
    os << bs.getValue();
  else
    os << "?";
  os << "x" << bt.getElementType() << ">";
}

/// RangeType prints as just "range".
static void print(RangeType rt, DialectAsmPrinter &os) { os << "range"; }

void mlir::linalg::LinalgDialect::printType(Type type,
                                            DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Linalg type");
  case LinalgTypes::Buffer:
    print(type.cast<BufferType>(), os);
    break;
  case LinalgTypes::Range:
    print(type.cast<RangeType>(), os);
    break;
  }
}
