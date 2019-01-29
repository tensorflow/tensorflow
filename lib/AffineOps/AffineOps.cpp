//===- AffineOps.cpp - MLIR Affine Operations -----------------------------===//
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

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// AffineOpsDialect
//===----------------------------------------------------------------------===//

AffineOpsDialect::AffineOpsDialect(MLIRContext *context)
    : Dialect(/*namePrefix=*/"", context) {
  addOperations<AffineIfOp>();
}

//===----------------------------------------------------------------------===//
// AffineIfOp
//===----------------------------------------------------------------------===//

void AffineIfOp::build(Builder *builder, OperationState *result,
                       IntegerSet condition,
                       ArrayRef<Value *> conditionOperands) {
  result->addAttribute(getConditionAttrName(), IntegerSetAttr::get(condition));
  result->addOperands(conditionOperands);

  // Reserve 2 block lists, one for the 'then' and one for the 'else' regions.
  result->reserveBlockLists(2);
}

bool AffineIfOp::verify() const {
  // Verify that we have a condition attribute.
  auto conditionAttr = getAttrOfType<IntegerSetAttr>(getConditionAttrName());
  if (!conditionAttr)
    return emitOpError("requires an integer set attribute named 'condition'");

  // Verify that the operands are valid dimension/symbols.
  IntegerSet condition = conditionAttr.getValue();
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const Value *operand = getOperand(i);
    if (i < condition.getNumDims() && !operand->isValidDim())
      return emitOpError("operand cannot be used as a dimension id");
    if (i >= condition.getNumDims() && !operand->isValidSymbol())
      return emitOpError("operand cannot be used as a symbol");
  }

  // Verify that the entry of each child blocklist does not have arguments.
  for (const auto &blockList : getInstruction()->getBlockLists()) {
    if (blockList.empty())
      continue;

    // TODO(riverriddle) We currently do not allow multiple blocks in child
    // block lists.
    if (std::next(blockList.begin()) != blockList.end())
      return emitOpError(
          "expects only one block per 'if' or 'else' block list");
    if (blockList.front().getTerminator())
      return emitOpError("expects region block to not have a terminator");

    for (const auto &b : blockList)
      if (b.getNumArguments() != 0)
        return emitOpError(
            "requires that child entry blocks have no arguments");
  }
  return false;
}

bool AffineIfOp::parse(OpAsmParser *parser, OperationState *result) {
  // Parse the condition attribute set.
  IntegerSetAttr conditionAttr;
  unsigned numDims;
  if (parser->parseAttribute(conditionAttr, getConditionAttrName().data(),
                             result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims))
    return true;

  // Verify the condition operands.
  auto set = conditionAttr.getValue();
  if (set.getNumDims() != numDims)
    return parser->emitError(
        parser->getNameLoc(),
        "dim operand count and integer set dim count must match");
  if (numDims + set.getNumSymbols() != result->operands.size())
    return parser->emitError(
        parser->getNameLoc(),
        "symbol operand count and integer set symbol count must match");

  // Parse the 'then' block list.
  if (parser->parseBlockList())
    return true;

  // If we find an 'else' keyword then parse the else block list.
  if (!parser->parseOptionalKeyword("else")) {
    if (parser->parseBlockList())
      return true;
  }

  // Reserve 2 block lists, one for the 'then' and one for the 'else' regions.
  result->reserveBlockLists(2);
  return false;
}

void AffineIfOp::print(OpAsmPrinter *p) const {
  auto conditionAttr = getAttrOfType<IntegerSetAttr>(getConditionAttrName());
  *p << "if " << conditionAttr;
  printDimAndSymbolList(operand_begin(), operand_end(),
                        conditionAttr.getValue().getNumDims(), p);
  p->printBlockList(getInstruction()->getBlockList(0));

  // Print the 'else' block list if it has any blocks.
  const auto &elseBlockList = getInstruction()->getBlockList(1);
  if (!elseBlockList.empty()) {
    *p << " else";
    p->printBlockList(elseBlockList);
  }
}

IntegerSet AffineIfOp::getIntegerSet() const {
  return getAttrOfType<IntegerSetAttr>(getConditionAttrName()).getValue();
}
void AffineIfOp::setIntegerSet(IntegerSet newSet) {
  setAttr(
      Identifier::get(getConditionAttrName(), getInstruction()->getContext()),
      IntegerSetAttr::get(newSet));
}

/// Returns the list of 'then' blocks.
BlockList &AffineIfOp::getThenBlocks() {
  return getInstruction()->getBlockList(0);
}

/// Returns the list of 'else' blocks.
BlockList &AffineIfOp::getElseBlocks() {
  return getInstruction()->getBlockList(1);
}
