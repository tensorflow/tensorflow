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
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

// TODO: Have verify functions return std::string to enable more descriptive
// error messages.
OpAsmParserResult AddFOp::parse(OpAsmParser *parser) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type *type;
  SSAValue *lhs, *rhs;
  if (parser->parseOperandList(ops, 2) || parser->parseColonType(type) ||
      parser->resolveOperand(ops[0], type, lhs) ||
      parser->resolveOperand(ops[1], type, rhs))
    return {};

  return OpAsmParserResult({lhs, rhs}, type);
}

void AddFOp::print(OpAsmPrinter *p) const {
  *p << "addf " << *getOperand(0) << ", " << *getOperand(1) << " : "
     << *getType();
}

// Return an error message on failure.
const char *AddFOp::verify() const {
  // TODO: Check that the types of the LHS and RHS match.
  // TODO: This should be a refinement of TwoOperands.
  // TODO: There should also be a OneResultWhoseTypeMatchesFirstOperand.
  return nullptr;
}

OpAsmParserResult AffineApplyOp::parse(OpAsmParser *parser) {
  SmallVector<OpAsmParser::OperandType, 2> opInfos;
  SmallVector<SSAValue *, 4> operands;

  auto &builder = parser->getBuilder();
  auto *affineIntTy = builder.getAffineIntType();

  AffineMapAttr *mapAttr;
  if (parser->parseAttribute(mapAttr) ||
      parser->parseOperandList(opInfos, -1,
                               OpAsmParser::Delimeter::ParenDelimeter))
    return {};
  unsigned numDims = opInfos.size();

  if (parser->parseOperandList(
          opInfos, -1, OpAsmParser::Delimeter::OptionalSquareDelimeter) ||
      parser->resolveOperands(opInfos, affineIntTy, operands))
    return {};

  auto *map = mapAttr->getValue();
  if (map->getNumDims() != numDims ||
      numDims + map->getNumSymbols() != opInfos.size()) {
    parser->emitError(parser->getNameLoc(),
                      "dimension or symbol index mismatch");
    return {};
  }

  SmallVector<Type *, 4> resultTypes(map->getNumResults(), affineIntTy);
  return OpAsmParserResult(
      operands, resultTypes,
      NamedAttribute(builder.getIdentifier("map"), mapAttr));
}

void AffineApplyOp::print(OpAsmPrinter *p) const {
  auto *map = getAffineMap();
  *p << "affine_apply " << *map;

  auto opit = operand_begin();
  *p << '(';
  p->printOperands(opit, opit + map->getNumDims());
  *p << ')';

  if (map->getNumSymbols()) {
    *p << '[';
    p->printOperands(opit + map->getNumDims(), operand_end());
    *p << ']';
  }
}

const char *AffineApplyOp::verify() const {
  // Check that affine map attribute was specified.
  auto *affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return "requires an affine map.";

  // Check input and output dimensions match.
  auto *map = affineMapAttr->getValue();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != map->getNumDims() + map->getNumSymbols())
    return "operand count and affine map dimension and symbol count must match";

  // Verify that result count matches affine map result count.
  if (getNumResults() != map->getNumResults())
    return "result count and affine map result count must match";

  return nullptr;
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
const char *ConstantOp::verify() const {
  auto *value = getValue();
  if (!value)
    return "requires a 'value' attribute";

  auto *type = this->getType();
  if (isa<IntegerType>(type) || type->isAffineInt()) {
    if (!isa<IntegerAttr>(value))
      return "requires 'value' to be an integer for an integer result type";
    return nullptr;
  }

  if (isa<FunctionType>(type)) {
    // TODO: Verify a function attr.
  }

  return "requires a result type that aligns with the 'value' attribute";
}

/// ConstantIntOp only matches values whose result type is an IntegerType or
/// AffineInt.
bool ConstantIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         (isa<IntegerType>(op->getResult(0)->getType()) ||
          op->getResult(0)->getType()->isAffineInt());
}

void DimOp::print(OpAsmPrinter *p) const {
  *p << "dim " << *getOperand() << ", " << getIndex() << " : "
     << *getOperand()->getType();
}

OpAsmParserResult DimOp::parse(OpAsmParser *parser) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr *indexAttr;
  Type *type;
  SSAValue *operand;
  if (parser->parseOperand(operandInfo) || parser->parseComma() ||
      parser->parseAttribute(indexAttr) || parser->parseColonType(type) ||
      parser->resolveOperand(operandInfo, type, operand))
    return {};

  auto &builder = parser->getBuilder();
  return OpAsmParserResult(
      operand, builder.getAffineIntType(),
      NamedAttribute(builder.getIdentifier("index"), indexAttr));
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

void LoadOp::print(OpAsmPrinter *p) const {
  *p << "load " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << "] : " << *getMemRef()->getType();
}

OpAsmParserResult LoadOp::parse(OpAsmParser *parser) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *type;
  SmallVector<SSAValue *, 4> operands;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  if (parser->parseOperand(memrefInfo) ||
      parser->parseOperandList(indexInfo, -1,
                               OpAsmParser::Delimeter::SquareDelimeter) ||
      parser->parseColonType(type) ||
      parser->resolveOperands(memrefInfo, type, operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, operands))
    return {};

  return OpAsmParserResult(operands, type->getElementType());
}

const char *LoadOp::verify() const {
  if (getNumOperands() == 0)
    return "expected a memref to load from";

  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return "first operand must be a memref";

  for (auto *idx : getIndices())
    if (!idx->getType()->isAffineInt())
      return "index to load must have 'affineint' type";

  // TODO: Verify we have the right number of indices.

  // TODO: in MLFunction verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return nullptr;
}

/// Install the standard operations in the specified operation set.
void mlir::registerStandardOperations(OperationSet &opSet) {
  opSet.addOperations<AddFOp, AffineApplyOp, ConstantOp, DimOp, LoadOp>(
      /*prefix=*/"");
}
