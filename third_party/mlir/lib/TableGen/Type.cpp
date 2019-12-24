//===- Type.cpp - Type class ----------------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
