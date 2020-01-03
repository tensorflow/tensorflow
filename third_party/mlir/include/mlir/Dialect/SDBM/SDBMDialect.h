//===- SDBMDialect.h - Dialect for striped DBMs -----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SDBM_SDBMDIALECT_H
#define MLIR_DIALECT_SDBM_SDBMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {
class MLIRContext;

class SDBMDialect : public Dialect {
public:
  SDBMDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context) {}

  static StringRef getDialectNamespace() { return "sdbm"; }

  /// Get the uniquer for SDBM expressions. This should not be used directly.
  StorageUniquer &getUniquer() { return uniquer; }

private:
  StorageUniquer uniquer;
};
} // namespace mlir

#endif // MLIR_DIALECT_SDBM_SDBMDIALECT_H
