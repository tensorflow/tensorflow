//===- SDBMDialect.h - Dialect for striped DBMs -----------------*- C++ -*-===//
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
