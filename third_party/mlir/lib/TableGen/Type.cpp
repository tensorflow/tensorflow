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

tblgen::TypeConstraint::TypeConstraint(const llvm::Record *record)
    : Constraint(Constraint::CK_Type, record) {
  assert(def->isSubClassOf("TypeConstraint") &&
         "must be subclass of TableGen 'TypeConstraint' class");
}

tblgen::TypeConstraint::TypeConstraint(const llvm::DefInit *init)
    : TypeConstraint(init->getDef()) {}

bool tblgen::TypeConstraint::isVariadic() const {
  return def->isSubClassOf("Variadic");
}

tblgen::Type::Type(const llvm::Record *record) : TypeConstraint(record) {
  assert(def->isSubClassOf("Type") &&
         "must be subclass of TableGen 'Type' class");
}

tblgen::Type::Type(const llvm::DefInit *init) : Type(init->getDef()) {}

StringRef tblgen::Type::getTableGenDefName() const { return def->getName(); }

tblgen::Type tblgen::Type::getVariadicBaseType() const {
  assert(isVariadic() && "must be variadic type constraint");
  return Type(def->getValueAsDef("baseType"));
}
