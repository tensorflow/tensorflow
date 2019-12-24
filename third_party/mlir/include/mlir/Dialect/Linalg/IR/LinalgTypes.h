//===- LinalgTypes.h - Linalg Types ---------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LINALGTYPES_H_
#define MLIR_DIALECT_LINALG_LINALGTYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
class MLIRContext;

namespace linalg {
enum LinalgTypes {
  Range = Type::FIRST_LINALG_TYPE,
  LAST_USED_LINALG_TYPE = Range,
};

class LinalgDialect : public Dialect {
public:
  explicit LinalgDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "linalg"; }

  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

/// A RangeType represents a minimal range abstraction (min, max, step).
/// It is constructed by calling the linalg.range op with three values index of
/// index type:
///
/// ```mlir
///    func @foo(%arg0 : index, %arg1 : index, %arg2 : index) {
///      %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
///    }
/// ```
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

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTYPES_H_
