//===- LinalgTypes.h - Linalg Types ---------------------------------------===//
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

#ifndef MLIR_LINALG_LINALGTYPES_H_
#define MLIR_LINALG_LINALGTYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
class MLIRContext;

enum LinalgTypes {
  Buffer = Type::FIRST_LINALG_TYPE,
  Range,
  LAST_USED_LINALG_TYPE = Range,
};

class LinalgDialect : public Dialect {
public:
  explicit LinalgDialect(MLIRContext *context);

  /// Parse a type registered to this dialect.
  Type parseType(llvm::StringRef spec, Location loc) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, llvm::raw_ostream &os) const override;
};

/// A BufferType represents a minimal range abstraction (min, max, step).
class BufferTypeStorage;
class BufferType : public Type::TypeBase<BufferType, Type, BufferTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  /// Construction hook.
  static BufferType get(MLIRContext *context, Type elementType);
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Buffer; }
  //////////////////////////////////////////////////////////////////////////////
  // Type-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  Type getElementType();
};

/// A RangeType represents a minimal range abstraction (min, max, step).
class RangeType : public Type::TypeBase<RangeType, Type> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  /// Construction hook.
  static RangeType get(MLIRContext *context) {
    /// Custom, uniq'ed construction in the MLIRContext.
    return Base::get(context, LinalgTypes::Range);
  }
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Range; }
};

} // namespace mlir

#endif // MLIR_LINALG_LINALGTYPES_H_
