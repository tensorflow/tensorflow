//===- Argument.h - Argument definitions ------------------------*- C++ -*-===//
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

// A struct wrapping an attribute and its name together
struct NamedAttribute {
  // Returns the MLIR attribute name.
  std::string getName() const;

  llvm::StringRef name;
  Attribute attr;
};

// A struct wrapping an operand and its name together
struct Operand {
  // Returns true if this operand has constraint that need to be satisfied.
  bool hasMatcher() const;

  llvm::StringRef name;
  Type type;
};

// Operation argument: either attribute or operand
using Argument = llvm::PointerUnion<NamedAttribute *, Operand *>;

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_ARGUMENT_H_
