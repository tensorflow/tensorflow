//===- Type.cpp - Type class ----------------------------------------------===//
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

#include "mlir/TableGen/Type.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

TypeConstraint::TypeConstraint(const llvm::Record *record)
    : Constraint(Constraint::CK_Type, record) {
  assert(def->isSubClassOf("TypeConstraint") &&
         "must be subclass of TableGen 'TypeConstraint' class");
}

TypeConstraint::TypeConstraint(const llvm::DefInit *init)
    : TypeConstraint(init->getDef()) {}

bool TypeConstraint::isVariadic() const {
  return def->isSubClassOf("Variadic");
}

Type::Type(const llvm::Record *record) : TypeConstraint(record) {}

StringRef Type::getTypeDescription() const {
  return def->getValueAsString("typeDescription");
}

Dialect Type::getDialect() const {
  return Dialect(def->getValueAsDef("dialect"));
}
