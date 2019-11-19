//===- Constraint.cpp - Constraint class ----------------------------------===//
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
