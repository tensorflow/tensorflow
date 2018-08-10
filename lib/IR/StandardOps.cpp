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
#include "mlir/Support/STLExtras.h"
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
                      SmallVector<SSAValue *, 4> &operands, unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser->parseOperandList(opInfos, -1, OpAsmParser::Delimiter::Paren))
    return true;
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto *affineIntTy = parser->getBuilder().getAffineIntType();
  if (parser->parseOperandList(opInfos, -1,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->resolveOperands(opInfos, affineIntTy, operands))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

bool AddFOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type *type;
  return parser->parseOperandList(ops, 2) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperands(ops, type, result->operands) ||
         parser->addTypeToList(type, result->types);
}

void AddFOp::print(OpAsmPrinter *p) const {
  *p << "addf " << *getOperand(0) << ", " << *getOperand(1);
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getType();
}

// TODO: Have verify functions return std::string to enable more descriptive
// error messages.
// Return an error message on failure.
const char *AddFOp::verify() const {
  // TODO: Check that the types of the LHS and RHS match.
  // TODO: This should be a refinement of TwoOperands.
  // TODO: There should also be a OneResultWhoseTypeMatchesFirstOperand.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

bool AffineApplyOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  auto *affineIntTy = builder.getAffineIntType();

  AffineMapAttr *mapAttr;
  unsigned numDims;
  if (parser->parseAttribute(mapAttr, "map", result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  auto *map = mapAttr->getValue();

  if (map->getNumDims() != numDims ||
      numDims + map->getNumSymbols() != result->operands.size()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension or symbol index mismatch");
  }

  result->types.append(map->getNumResults(), affineIntTy);
  return false;
}

void AffineApplyOp::print(OpAsmPrinter *p) const {
  auto *map = getAffineMap();
  *p << "affine_apply " << *map;
  printDimAndSymbolList(operand_begin(), operand_end(), map->getNumDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
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

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocOp::print(OpAsmPrinter *p) const {
  MemRefType *type = cast<MemRefType>(getMemRef()->getType());
  *p << "alloc";
  // Print dynamic dimension operands.
  printDimAndSymbolList(operand_begin(), operand_end(),
                        type->getNumDynamicDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
  *p << " : " << *type;
}

bool AllocOp::parse(OpAsmParser *parser, OperationState *result) {
  MemRefType *type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, result->operands, numDimOperands) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  // Check numDynamicDims against number of question marks in memref type.
  if (numDimOperands != type->getNumDynamicDims()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension operand count does not equal memref "
                             "dynamic dimension count");
  }

  // Check that the number of symbol operands matches the number of symbols in
  // the first affinemap of the memref's affine map composition.
  // Note that a memref must specify at least one affine map in the composition.
  if (result->operands.size() - numDimOperands !=
      type->getAffineMaps()[0]->getNumSymbols()) {
    return parser->emitError(
        parser->getNameLoc(),
        "affine map symbol operand count does not equal memref affine map "
        "symbol count");
  }

  result->types.push_back(type);
  return false;
}

const char *AllocOp::verify() const {
  // TODO(andydavis): Verify alloc.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter *p) const {
  *p << "constant " << *getValue();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"value");
  *p << " : " << *getType();
}

bool ConstantOp::parse(OpAsmParser *parser, OperationState *result) {
  Attribute *valueAttr;
  Type *type;

  return parser->parseAttribute(valueAttr, "value", result->attributes) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->addTypeToList(type, result->types);
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

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<IntegerType>(op->getResult(0)->getType());
}

OperationState ConstantIntOp::build(Builder *builder, int64_t value,
                                    unsigned width) {
  OperationState result(builder->getIdentifier("constant"));
  result.attributes.push_back(
      {builder->getIdentifier("value"), builder->getIntegerAttr(value)});
  result.types.push_back(builder->getIntegerType(width));
  return result;
}

/// ConstantAffineIntOp only matches values whose result type is AffineInt.
bool ConstantAffineIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType()->isAffineInt();
}

OperationState ConstantAffineIntOp::build(Builder *builder, int64_t value) {
  OperationState result(builder->getIdentifier("constant"));
  result.attributes.push_back(
      {builder->getIdentifier("value"), builder->getIntegerAttr(value)});
  result.types.push_back(builder->getAffineIntType());
  return result;
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::print(OpAsmPrinter *p) const {
  *p << "dim " << *getOperand() << ", " << getIndex();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"index");
  *p << " : " << *getOperand()->getType();
}

bool DimOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr *indexAttr;
  Type *type;

  return parser->parseOperand(operandInfo) || parser->parseComma() ||
         parser->parseAttribute(indexAttr, "index", result->attributes) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(operandInfo, type, result->operands) ||
         parser->addTypeToList(parser->getBuilder().getAffineIntType(),
                               result->types);
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

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::print(OpAsmPrinter *p) const {
  *p << "load " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getMemRef()->getType();
}

bool LoadOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *type;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  return parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type->getElementType(), result->types);
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

bool ReturnOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type *, 2> types;

  return parser->parseOperandList(opInfo, -1, OpAsmParser::Delimiter::None) ||
         (!opInfo.empty() && parser->parseColonTypeList(types)) ||
         parser->resolveOperands(opInfo, types, result->operands);
}

void ReturnOp::print(OpAsmPrinter *p) const {
  *p << "return";
  if (getNumOperands() > 0) {
    *p << " ";
    p->printOperands(operand_begin(), operand_end());
    *p << " : ";
    interleave(operand_begin(), operand_end(),
               [&](auto *e) { p->printType(e->getType()); },
               [&]() { *p << ", "; });
  }
}

const char *ReturnOp::verify() const {
  // ReturnOp must be part of an ML function.
  if (auto *stmt = dyn_cast<OperationStmt>(getOperation())) {
    MLFunction *func = dyn_cast_or_null<MLFunction>(stmt->getBlock());
    if (!func || &func->back() != stmt)
      return "must be the last statement in the ML function";

    // Return success. Checking that operand types match those in the function
    // signature is performed in the ML function verifier.
    return nullptr;
  }
  return "cannot occur in a CFG function";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::print(OpAsmPrinter *p) const {
  *p << "store " << *getValueToStore();
  *p << ", " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getMemRef()->getType();
}

bool StoreOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *memrefType;

  auto affineIntTy = parser->getBuilder().getAffineIntType();
  return parser->parseOperand(storeValueInfo) || parser->parseComma() ||
         parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(memrefType) ||
         parser->resolveOperand(storeValueInfo, memrefType->getElementType(),
                                result->operands) ||
         parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands);
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

//===----------------------------------------------------------------------===//
// Register operations.
//===----------------------------------------------------------------------===//

/// Install the standard operations in the specified operation set.
void mlir::registerStandardOperations(OperationSet &opSet) {
  opSet.addOperations<AddFOp, AffineApplyOp, AllocOp, ConstantOp, DimOp, LoadOp,
                      ReturnOp, StoreOp>(
      /*prefix=*/"");
}
