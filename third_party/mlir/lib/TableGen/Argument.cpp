//===- Argument.cpp - Argument definitions --------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Argument.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

bool tblgen::NamedTypeConstraint::hasPredicate() const {
  return !constraint.getPredicate().isNull();
}

bool tblgen::NamedTypeConstraint::isVariadic() const {
  return constraint.isVariadic();
}
