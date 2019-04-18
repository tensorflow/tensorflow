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

#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Linalg/LinalgOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

mlir::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect("linalg", context) {
  addTypes<BufferType, RangeType>();
  addOperations<BufferAllocOp, BufferDeallocOp, RangeOp>();
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
  return (context->emitError(loc, "unknown Linalg type: " + spec), Type());
}

/// RangeType prints as just "range".
static void print(BufferType bt, raw_ostream &os) {
  os << "buffer<" << bt.getElementType() << ">";
}
static void print(RangeType rt, raw_ostream &os) { os << "range"; }

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
  }
}
