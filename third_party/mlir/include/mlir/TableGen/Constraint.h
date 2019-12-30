//===- Constraint.h - Constraint class --------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint wrapper to simplify using TableGen Record for constraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CONSTRAINT_H_
#define MLIR_TABLEGEN_CONSTRAINT_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing Constraint defined in
// TableGen.
class Constraint {
public:
  Constraint(const llvm::Record *record);

  bool operator==(const Constraint &that) { return def == that.def; }
  bool operator!=(const Constraint &that) { return def != that.def; }

  // Returns the predicate for this constraint.
  Pred getPredicate() const;

  // Returns the condition template that can be used to check if a type or
  // attribute satisfies this constraint.  The template may contain "{0}" that
  // must be substituted with an expression returning an mlir::Type or
  // mlir::Attribute.
  std::string getConditionTemplate() const;

  // Returns the user-readable description of this constraint. If the
  // description is not provided, returns the TableGen def name.
  StringRef getDescription() const;

  // Constraint kind
  enum Kind { CK_Attr, CK_Region, CK_Type, CK_Uncategorized };

  Kind getKind() const { return kind; }

protected:
  Constraint(Kind kind, const llvm::Record *record);

  // The TableGen definition of this constraint.
  const llvm::Record *def;

private:
  // What kind of constraint this is.
  Kind kind;
};

// An constraint and the concrete entities to place the constraint on.
struct AppliedConstraint {
  AppliedConstraint(Constraint &&constraint, StringRef self,
                    std::vector<std::string> &&entities);

  Constraint constraint;
  // The symbol to replace `$_self` special placeholder in the constraint.
  std::string self;
  // The symbols to replace `$N` positional placeholders in the constraint.
  std::vector<std::string> entities;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_CONSTRAINT_H_
