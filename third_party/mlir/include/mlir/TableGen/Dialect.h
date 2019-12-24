//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_DIALECT_H_
#define MLIR_TABLEGEN_DIALECT_H_

#include "mlir/Support/LLVM.h"

namespace llvm {
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {
// Wrapper class that contains a MLIR dialect's information defined in TableGen
// and provides helper methods for accessing them.
class Dialect {
public:
  explicit Dialect(const llvm::Record *def) : def(def) {}

  // Returns the name of this dialect.
  StringRef getName() const;

  // Returns the C++ namespaces that ops of this dialect should be placed into.
  StringRef getCppNamespace() const;

  // Returns the summary description of the dialect. Returns empty string if
  // none.
  StringRef getSummary() const;

  // Returns the description of the dialect. Returns empty string if none.
  StringRef getDescription() const;

  // Returns whether two dialects are equal by checking the equality of the
  // underlying record.
  bool operator==(const Dialect &other) const;

  // Compares two dialects by comparing the names of the dialects.
  bool operator<(const Dialect &other) const;

  // Returns whether the dialect is defined.
  operator bool() const { return def != nullptr; }

private:
  const llvm::Record *def;
};
} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_DIALECT_H_
