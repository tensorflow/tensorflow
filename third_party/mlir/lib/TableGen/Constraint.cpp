//===- Constraint.cpp - Constraint class ----------------------------------===//
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

#include "mlir/TableGen/Constraint.h"
#include "llvm/TableGen/Record.h"

using namespace mlir::tblgen;

Constraint::Constraint(const llvm::Record *record)
    : def(record), kind(CK_Uncategorized) {
  if (record->isSubClassOf("TypeConstraint")) {
    kind = CK_Type;
  } else if (record->isSubClassOf("AttrConstraint")) {
    kind = CK_Attr;
  } else if (record->isSubClassOf("RegionConstraint")) {
    kind = CK_Region;
  } else {
    assert(record->isSubClassOf("Constraint"));
  }
}

Constraint::Constraint(Kind kind, const llvm::Record *record)
    : def(record), kind(kind) {}

Pred Constraint::getPredicate() const {
  auto *val = def->getValue("predicate");

  // If no predicate is specified, then return the null predicate (which
  // corresponds to true).
  if (!val)
    return Pred();

  const auto *pred = dyn_cast<llvm::DefInit>(val->getValue());
  return Pred(pred);
}

std::string Constraint::getConditionTemplate() const {
  return getPredicate().getCondition();
}

llvm::StringRef Constraint::getDescription() const {
  auto doc = def->getValueAsString("description");
  if (doc.empty())
    return def->getName();
  return doc;
}

AppliedConstraint::AppliedConstraint(Constraint &&constraint,
                                     llvm::StringRef self,
                                     std::vector<std::string> &&entities)
    : constraint(constraint), self(self), entities(std::move(entities)) {}
