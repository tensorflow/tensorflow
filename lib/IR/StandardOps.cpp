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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
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

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
const char *ConstantOp::verify() const {
  auto *value = getValue();
  if (!value)
    return "requires a 'value' attribute";

  auto *type = this->getType();
  if (isa<IntegerType>(type)) {
    if (!isa<IntegerAttr>(value))
      return "requires 'value' to be an integer for an integer result type";
    return nullptr;
  }

  if (isa<FunctionType>(type)) {
    // TODO: Verify a function attr.
  }

  return "requires a result type that aligns with the 'value' attribute";
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<IntegerType>(op->getResult(0)->getType());
}

void DimOp::print(raw_ostream &os) const {
  os << "dim xxx, " << getIndex() << " : sometype";
}

const char *DimOp::verify() const {
  // Check that we have an integer index operand.
  auto indexAttr = getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return "requires an integer attribute named 'index'";
  uint64_t index = (uint64_t)indexAttr->getValue();

  auto *type = getOperand()->getType();
  if (auto *tensorType = dyn_cast<RankedTensorType>(type)) {
    if (index >= tensorType->getRank())
      return "index is out of range";
  } else if (auto *memrefType = dyn_cast<MemRefType>(type)) {
    if (index >= memrefType->getRank())
      return "index is out of range";

  } else if (isa<UnrankedTensorType>(type)) {
    // ok, assumed to be in-range.
  } else {
    return "requires an operand with tensor or memref type";
  }

  return nullptr;
}

void AffineApplyOp::print(raw_ostream &os) const {
  os << "affine_apply map: ";
  getAffineMap()->print(os);
}

const char *AffineApplyOp::verify() const {
  // TODO: Check input and output dimensions match.

  // Check that affine map attribute was specified
  auto affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return "requires an affine map.";

  return nullptr;
}

/// Install the standard operations in the specified operation set.
void mlir::registerStandardOperations(OperationSet &opSet) {
  opSet.addOperations<AddFOp, ConstantOp, DimOp, AffineApplyOp>(/*prefix=*/"");
}
