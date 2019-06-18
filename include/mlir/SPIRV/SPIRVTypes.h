//===- SPIRVTypes.h - MLIR SPIR-V Types -------------------------*- C++ -*-===//
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
// This file declares the types in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SPIRV_SPIRVTYPES_H_
#define MLIR_SPIRV_SPIRVTYPES_H_

#include "mlir/IR/Types.h"

// Pull in all enum type definitions and utility function declarations
#include "mlir/SPIRV/SPIRVEnums.h.inc"

namespace mlir {
namespace spirv {

namespace detail {
struct ArrayTypeStorage;
struct PointerTypeStorage;
struct RuntimeArrayTypeStorage;
} // namespace detail

namespace TypeKind {
enum Kind {
  Array = Type::FIRST_SPIRV_TYPE,
  Pointer,
  RuntimeArray,
};
}

// SPIR-V array type
class ArrayType
    : public Type::TypeBase<ArrayType, Type, detail::ArrayTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Array; }

  static ArrayType get(Type elementType, int64_t elementCount);

  Type getElementType();

  int64_t getElementCount();
};

// SPIR-V pointer type
class PointerType
    : public Type::TypeBase<PointerType, Type, detail::PointerTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Pointer; }

  static PointerType get(Type pointeeType, StorageClass storageClass);

  Type getPointeeType();

  StorageClass getStorageClass();
  StringRef getStorageClassStr();
};

// SPIR-V run-time array type
class RuntimeArrayType
    : public Type::TypeBase<RuntimeArrayType, Type,
                            detail::RuntimeArrayTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::RuntimeArray; }

  static RuntimeArrayType get(Type elementType);

  Type getElementType();
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_SPIRV_SPIRVTYPES_H_
