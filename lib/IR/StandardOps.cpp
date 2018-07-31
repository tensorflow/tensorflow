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

static void printDimAndSymbolList(Operation::const_operand_iterator begin,
                                  Operation::const_operand_iterator end,
                                  unsigned numDims, OpAsmPrinter *p) {
  *p << '(';
  p->printOperands(begin, begin + numDims);
  *p << ')';

  if (begin + numDims != end) {
    *p << '[';
    p->printOperands(begin + numDims, end);
    *p << ']';
  }
}

// Parses dimension and symbol list, and sets 'numDims' to the number of
// dimension operands parsed.
// Returns 'false' on success and 'true' on error.
static bool
parseDimAndSymbolList(OpAsmParser *parser,
                      SmallVectorImpl<OpAsmParser::OperandType> &opInfos,
                      SmallVector<SSAValue *, 4> &operands, unsigned &numDims) {
  if (parser->parseOperandList(opInfos, -1,
                               OpAsmParser::Delimeter::ParenDelimeter))
    return true;
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto *affineIntTy = parser->getBuilder().getAffineIntType();
  if (parser->parseOperandList(
          opInfos, -1, OpAsmParser::Delimeter::OptionalSquareDelimeter) ||
      parser->resolveOperands(opInfos, affineIntTy, operands))
    return true;
  return false;
}

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
  unsigned numDims;
  if (parser->parseAttribute(mapAttr) ||
      parseDimAndSymbolList(parser, opInfos, operands, numDims))
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
  printDimAndSymbolList(operand_begin(), operand_end(), map->getNumDims(), p);
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

void AllocOp::print(OpAsmPrinter *p) const {
  MemRefType *type = cast<MemRefType>(getMemRef()->getType());
  *p << "alloc";
  // Print dynamic dimension operands.
  printDimAndSymbolList(operand_begin(), operand_end(),
                        type->getNumDynamicDims(), p);
  // Print memref type.
  *p << " : " << *type;
}

OpAsmParserResult AllocOp::parse(OpAsmParser *parser) {
  MemRefType *type;
  SmallVector<SSAValue *, 4> operands;
  SmallVector<OpAsmParser::OperandType, 4> operandsInfo;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, operandsInfo, operands, numDimOperands) ||
      parser->parseColonType(type))
    return {};

  // Check numDynamicDims against number of question marks in memref type.
  if (numDimOperands != type->getNumDynamicDims()) {
    parser->emitError(parser->getNameLoc(),
                      "Dynamic dimensions count mismatch: dimension operand "
                      "count does not equal memref dynamic dimension count.");
    return {};
  }

  // Check that the number of symbol operands matches the number of symbols in
  // the first affinemap of the memref's affine map composition.
  // Note that a memref must specify at least one affine map in the composition.
  if ((operandsInfo.size() - numDimOperands) !=
      type->getAffineMaps()[0]->getNumSymbols()) {
    parser->emitError(parser->getNameLoc(),
                      "AffineMap symbol count mismatch: symbol operand "
                      "count does not equal memref affine map symbol count.");
    return {};
  }

  return OpAsmParserResult(operands, type);
}

const char *AllocOp::verify() const {
  // TODO(andydavis): Verify alloc.
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

void StoreOp::print(OpAsmPrinter *p) const {
  *p << "store " << *getValueToStore();
  *p << ", " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << "] : " << *getMemRef()->getType();
}

OpAsmParserResult StoreOp::parse(OpAsmParser *parser) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  SmallVector<SSAValue *, 4> operands;
  MemRefType *memrefType;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  if (parser->parseOperand(storeValueInfo) || parser->parseComma() ||
      parser->parseOperand(memrefInfo) ||
      parser->parseOperandList(indexInfo, -1,
                               OpAsmParser::Delimeter::SquareDelimeter) ||
      parser->parseColonType(memrefType) ||
      parser->resolveOperands(storeValueInfo, memrefType->getElementType(),
                              operands) ||
      parser->resolveOperands(memrefInfo, memrefType, operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, operands))
    return {};

  return OpAsmParserResult(operands, {});
}

const char *StoreOp::verify() const {
  if (getNumOperands() < 2)
    return "expected a value to store and a memref";

  // Second operand is a memref type.
  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return "second operand must be a memref";

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != memRefType->getElementType())
    return "first operand must have same type memref element type ";

  if (getNumOperands() != 2 + memRefType->getRank())
    return "store index operand count not equal to memref rank";

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
  opSet.addOperations<AddFOp, AffineApplyOp, AllocOp, ConstantOp, DimOp, LoadOp,
                      StoreOp>(
      /*prefix=*/"");
}
