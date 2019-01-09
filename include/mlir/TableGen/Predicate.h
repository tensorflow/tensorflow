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

// Predicate in Conjunctive Normal Form (CNF).
//
// CNF is an AND of ORs. That means there are two levels of lists: the inner
// list contains predicate atoms, which are ORed. Then outer list ANDs its inner
// lists.
// An empty CNF is defined as always true, thus matching everything.
class PredCNF {
public:
  // Constructs an empty predicate CNF.
  explicit PredCNF() : def(nullptr) {}

  explicit PredCNF(const llvm::Record *def) : def(def) {}

  // Constructs a predicate CNF out of the given TableGen initializer.
  // The initializer is allowed to be unset initializer (?); then we are
  // constructing an empty predicate CNF.
  explicit PredCNF(const llvm::Init *init);

  // Returns true if this is an empty predicate CNF.
  bool isEmpty() const { return !def; }

  // Returns the conditions inside this predicate CNF. Returns nullptr if
  // this is an empty predicate CNF.
  const llvm::ListInit *getConditions() const;

  // Returns the template string to construct the matcher corresponding to this
  // predicate CNF. The string uses '{0}' to represent the type.
  std::string createTypeMatcherTemplate() const;

private:
  // The TableGen definition of this predicate CNF. nullptr means an empty
  // predicate CNF.
  const llvm::Record *def;
};

} // end namespace mlir

#endif // MLIR_TABLEGEN_PREDICATE_H_
