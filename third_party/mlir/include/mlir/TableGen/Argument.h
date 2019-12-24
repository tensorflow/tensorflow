//===- Argument.h - Argument definitions ------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file contains definitions for TableGen operation's arguments.
// Operation arguments fall into two categories:
//
// 1. Operands: SSA values operated on by the operation
// 2. Attributes: compile-time known properties that have influence over
//    the operation's behavior
//
// These two categories are modelled with the unified argument concept in
// TableGen because we need similar pattern matching mechanisms for them.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ARGUMENT_H_
#define MLIR_TABLEGEN_ARGUMENT_H_

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/PointerUnion.h"
#include <string>

namespace llvm {
class StringRef;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// A struct wrapping an op attribute and its name together
struct NamedAttribute {
  llvm::StringRef name;
  Attribute attr;
};

// A struct wrapping an op operand/result's constraint and its name together
struct NamedTypeConstraint {
  // Returns true if this operand/result has constraint to be satisfied.
  bool hasPredicate() const;
  // Returns true if this operand/result is variadic.
  bool isVariadic() const;

  llvm::StringRef name;
  TypeConstraint constraint;
};

// Operation argument: either attribute or operand
using Argument = llvm::PointerUnion<NamedAttribute *, NamedTypeConstraint *>;

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_ARGUMENT_H_
