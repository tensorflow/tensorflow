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
