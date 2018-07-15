//===- StandardOps.cpp - Standard MLIR Operations -------------------------===//
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

#include "mlir/IR/StandardOps.h"
#include "mlir/IR/OperationSet.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

void AddFOp::print(raw_ostream &os) const {
  os << "addf xx, yy : sometype";
}

// Return an error message on failure.
const char *AddFOp::verify() const {
  // TODO: Check that the types of the LHS and RHS match.
  // TODO: This should be a refinement of TwoOperands.
  // TODO: There should also be a OneResultWhoseTypeMatchesFirstOperand.
  return nullptr;
}

void DimOp::print(raw_ostream &os) const {
  os << "dim xxx, " << getIndex() << " : sometype";
}

const char *DimOp::verify() const {
  // TODO: Check that the operand has tensor or memref type.

  // Check that we have an integer index operand.
  auto indexAttr = getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return "'dim' op requires an integer attribute named 'index'";

  // TODO: Check that the index is in range.

  return nullptr;
}

/// Install the standard operations in the specified operation set.
void mlir::registerStandardOperations(OperationSet &opSet) {
  opSet.addOperations<AddFOp, DimOp>(/*prefix=*/ "");
}
