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
class ListInit;
} // end namespace llvm

namespace mlir {

// Predicate in conjunctive normal form.
class PredCNF {
public:
  PredCNF(llvm::ListInit *conditions) : conditions(conditions) {}

  // Return template string to construct matcher corresponding to predicate in
  // CNF form with '{0}' representing the type.
  std::string createTypeMatcherTemplate() const;

private:
  llvm::ListInit *conditions;
};

} // end namespace mlir

#endif // MLIR_TABLEGEN_PREDICATE_H_
