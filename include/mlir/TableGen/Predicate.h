//===- Predicate.h - Predicate class ----------------------------*- C++ -*-===//
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
// Wrapper around predicates defined in TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PREDICATE_H_
#define MLIR_TABLEGEN_PREDICATE_H_

#include "mlir/Support/LLVM.h"

namespace llvm {
class Init;
class ListInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// A logical predicate.
class Pred {
public:
  // Construct a Predicate from a record.
  explicit Pred(const llvm::Record *def);
  // Construct a Predicate from an initializer.
  explicit Pred(const llvm::Init *init);

  // Get the predicate condition.  The predicate must not be null.
  StringRef getCondition() const;

  // Check if the predicate is defined.  Callers may use this to interpret the
  // missing predicate as either true (e.g. in filters) or false (e.g. in
  // precondition verification).
  bool isNull() const { return def == nullptr; }

private:
  // The TableGen definition of this predicate.
  const llvm::Record *def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PREDICATE_H_
