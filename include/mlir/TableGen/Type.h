//===- Type.h - Type class --------------------------------------*- C++ -*-===//
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
// Type wrapper to simplify using TableGen Record defining a MLIR Type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_TYPE_H_
#define MLIR_TABLEGEN_TYPE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing Type constraints defined in
// TableGen.
class TypeConstraint {
public:
  explicit TypeConstraint(const llvm::Record &record);
  explicit TypeConstraint(const llvm::DefInit &init);

  // Returns the predicate that can be used to check if a type satisfies this
  // type constraint.
  Pred getPredicate() const;

  // Returns the condition template that can be used to check if a type
  // satisfies this type constraint.  The template may contain "{0}" that must
  // be substituted with an expression returning an mlir::Type.
  StringRef getConditionTemplate() const;

  // Returns the user-readable description of the constraint.  If the
  // description is not provided, returns an empty string.
  StringRef getDescription() const;

protected:
  // The TableGen definition of this type.
  const llvm::Record &def;
};

// Wrapper class providing helper methods for accessing MLIR Type defined
// in TableGen. This class should closely reflect what is defined as
// class Type in TableGen.
class Type : public TypeConstraint {
public:
  explicit Type(const llvm::Record &record);
  explicit Type(const llvm::Record *record) : Type(*record) {}
  explicit Type(const llvm::DefInit *init);

  // Returns the TableGen def name for this type.
  StringRef getTableGenDefName() const;

  // Returns the method call to invoke upon a MLIR pattern rewriter to
  // construct this type. Returns an empty StringRef if the method call
  // is undefined or unset.
  StringRef getBuilderCall() const;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPE_H_
