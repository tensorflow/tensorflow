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
#include "mlir/TableGen/Constraint.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing Type constraints defined in
// TableGen.
class TypeConstraint : public Constraint {
public:
  explicit TypeConstraint(const llvm::Record *record);
  explicit TypeConstraint(const llvm::DefInit *init);

  static bool classof(const Constraint *c) { return c->getKind() == CK_Type; }

  // Returns true if this is a variadic type constraint.
  bool isVariadic() const;
};

// Wrapper class providing helper methods for accessing MLIR Type defined
// in TableGen. This class should closely reflect what is defined as
// class Type in TableGen.
class Type : public TypeConstraint {
public:
  explicit Type(const llvm::Record *record);
  explicit Type(const llvm::DefInit *init);

  // Returns the TableGen def name for this type.
  StringRef getTableGenDefName() const;

  // Gets the base type of this variadic type constraint.
  // Precondition: isVariadic() is true.
  Type getVariadicBaseType() const;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPE_H_
