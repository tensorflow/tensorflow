//===- BuiltinOps.cpp - Builtin MLIR Operations -------------------------===//
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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

BuiltinDialect::BuiltinDialect(MLIRContext *context)
    : Dialect(/*opPrefix=*/"", context) {
  addOperations<AffineApplyOp, BranchOp, CondBranchOp, ConstantOp, ReturnOp>();
}

void mlir::printDimAndSymbolList(OperationInst::const_operand_iterator begin,
                                 OperationInst::const_operand_iterator end,
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
bool mlir::parseDimAndSymbolList(OpAsmParser *parser,
                                 SmallVector<Value *, 4> &operands,
                                 unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser->parseOperandList(opInfos, -1, OpAsmParser::Delimiter::Paren))
    return true;
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto affineIntTy = parser->getBuilder().getIndexType();
  if (parser->parseOperandList(opInfos, -1,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->resolveOperands(opInfos, affineIntTy, operands))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

void AffineApplyOp::build(Builder *builder, OperationState *result,
                          AffineMap map, ArrayRef<Value *> operands) {
  result->addOperands(operands);
  result->types.append(map.getNumResults(), builder->getIndexType());
  result->addAttribute("map", builder->getAffineMapAttr(map));
}

bool AffineApplyOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  auto affineIntTy = builder.getIndexType();

  AffineMapAttr mapAttr;
  unsigned numDims;
  if (parser->parseAttribute(mapAttr, "map", result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  auto map = mapAttr.getValue();

  if (map.getNumDims() != numDims ||
      numDims + map.getNumSymbols() != result->operands.size()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension or symbol index mismatch");
  }

  result->types.append(map.getNumResults(), affineIntTy);
  return false;
}

void AffineApplyOp::print(OpAsmPrinter *p) const {
  auto map = getAffineMap();
  *p << "affine_apply " << map;
  printDimAndSymbolList(operand_begin(), operand_end(), map.getNumDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
}

bool AffineApplyOp::verify() const {
  // Check that affine map attribute was specified.
  auto affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return emitOpError("requires an affine map");

  // Check input and output dimensions match.
  auto map = affineMapAttr.getValue();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != map.getNumDims() + map.getNumSymbols())
    return emitOpError(
        "operand count and affine map dimension and symbol count must match");

  // Verify that result count matches affine map result count.
  if (getNumResults() != map.getNumResults())
    return emitOpError("result count and affine map result count must match");

  return false;
}

// The result of the affine apply operation can be used as a dimension id if it
// is a CFG value or if it is an Value, and all the operands are valid
// dimension ids.
bool AffineApplyOp::isValidDim() const {
  for (auto *op : getOperands()) {
    if (!op->isValidDim())
      return false;
  }
  return true;
}

// The result of the affine apply operation can be used as a symbol if it is
// a CFG value or if it is an Value, and all the operands are symbols.
bool AffineApplyOp::isValidSymbol() const {
  for (auto *op : getOperands()) {
    if (!op->isValidSymbol())
      return false;
  }
  return true;
}

bool AffineApplyOp::constantFold(ArrayRef<Attribute> operandConstants,
                                 SmallVectorImpl<Attribute> &results,
                                 MLIRContext *context) const {
  auto map = getAffineMap();
  if (map.constantFold(operandConstants, results))
    return true;
  // Return false on success.
  return false;
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

void BranchOp::build(Builder *builder, OperationState *result, BasicBlock *dest,
                     ArrayRef<Value *> operands) {
  result->addSuccessor(dest, operands);
}

bool BranchOp::parse(OpAsmParser *parser, OperationState *result) {
  BasicBlock *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser->parseSuccessorAndUseList(dest, destOperands))
    return true;
  result->addSuccessor(dest, destOperands);
  return false;
}

void BranchOp::print(OpAsmPrinter *p) const {
  *p << "br ";
  p->printSuccessorAndUseList(getOperation(), 0);
}

bool BranchOp::verify() const {
  // ML functions do not have branching terminators.
  if (getOperation()->getFunction()->isML())
    return (emitOpError("cannot occur in a ML function"), true);
  return false;
}

BasicBlock *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(BasicBlock *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

void CondBranchOp::build(Builder *builder, OperationState *result,
                         Value *condition, BasicBlock *trueDest,
                         ArrayRef<Value *> trueOperands, BasicBlock *falseDest,
                         ArrayRef<Value *> falseOperands) {
  result->addOperands(condition);
  result->addSuccessor(trueDest, trueOperands);
  result->addSuccessor(falseDest, falseOperands);
}

bool CondBranchOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<Value *, 4> destOperands;
  BasicBlock *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type int1Ty = parser->getBuilder().getI1Type();
  if (parser->parseOperand(condInfo) || parser->parseComma() ||
      parser->resolveOperand(condInfo, int1Ty, result->operands)) {
    return parser->emitError(parser->getNameLoc(),
                             "expected condition type was boolean (i1)");
  }

  // Parse the true successor.
  if (parser->parseSuccessorAndUseList(dest, destOperands))
    return true;
  result->addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (parser->parseComma() ||
      parser->parseSuccessorAndUseList(dest, destOperands))
    return true;
  result->addSuccessor(dest, destOperands);

  // Return false on success.
  return false;
}

void CondBranchOp::print(OpAsmPrinter *p) const {
  *p << "cond_br ";
  p->printOperand(getCondition());
  *p << ", ";
  p->printSuccessorAndUseList(getOperation(), trueIndex);
  *p << ", ";
  p->printSuccessorAndUseList(getOperation(), falseIndex);
}

bool CondBranchOp::verify() const {
  // ML functions do not have branching terminators.
  if (getOperation()->getFunction()->isML())
    return (emitOpError("cannot occur in a ML function"), true);
  if (!getCondition()->getType().isInteger(1))
    return emitOpError("expected condition type was boolean (i1)");
  return false;
}

BasicBlock *CondBranchOp::getTrueDest() {
  return getOperation()->getSuccessor(trueIndex);
}

BasicBlock *CondBranchOp::getFalseDest() {
  return getOperation()->getSuccessor(falseIndex);
}

unsigned CondBranchOp::getNumTrueOperands() const {
  return getOperation()->getNumSuccessorOperands(trueIndex);
}

void CondBranchOp::eraseTrueOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(trueIndex, index);
}

unsigned CondBranchOp::getNumFalseOperands() const {
  return getOperation()->getNumSuccessorOperands(falseIndex);
}

void CondBranchOp::eraseFalseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(falseIndex, index);
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

/// Builds a constant op with the specified attribute value and result type.
void ConstantOp::build(Builder *builder, OperationState *result,
                       Attribute value, Type type) {
  result->addAttribute("value", value);
  result->types.push_back(type);
}

void ConstantOp::print(OpAsmPrinter *p) const {
  *p << "constant ";
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"value");

  if (getAttrs().size() > 1)
    *p << ' ';
  *p << getValue();
  if (!getValue().isa<FunctionAttr>())
    *p << " : " << getType();
}

bool ConstantOp::parse(OpAsmParser *parser, OperationState *result) {
  Attribute valueAttr;
  Type type;

  if (parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseAttribute(valueAttr, "value", result->attributes))
    return true;

  // 'constant' taking a function reference doesn't get a redundant type
  // specifier.  The attribute itself carries it.
  if (auto fnAttr = valueAttr.dyn_cast<FunctionAttr>())
    return parser->addTypeToList(fnAttr.getValue()->getType(), result->types);

  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
    type = intAttr.getType();
  } else if (auto fpAttr = valueAttr.dyn_cast<FloatAttr>()) {
    type = fpAttr.getType();
  } else if (parser->parseColonType(type)) {
    return true;
  }
  return parser->addTypeToList(type, result->types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
bool ConstantOp::verify() const {
  auto value = getValue();
  if (!value)
    return emitOpError("requires a 'value' attribute");

  auto type = this->getType();
  if (type.isa<IntegerType>() || type.isIndex()) {
    auto intAttr = value.dyn_cast<IntegerAttr>();
    if (!intAttr)
      return emitOpError(
          "requires 'value' to be an integer for an integer result type");

    // If the type has a known bitwidth we verify that the value can be
    // represented with the given bitwidth.
    if (!type.isIndex()) {
      auto bitwidth = type.cast<IntegerType>().getWidth();
      auto intVal = intAttr.getValue();
      if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
        return emitOpError("requires 'value' to be an integer within the range "
                           "of the integer result type");
    }
    return false;
  }

  if (type.isa<FloatType>()) {
    if (!value.isa<FloatAttr>())
      return emitOpError("requires 'value' to be a floating point constant");
    return false;
  }

  if (type.isa<VectorOrTensorType>()) {
    if (!value.isa<ElementsAttr>())
      return emitOpError("requires 'value' to be a vector/tensor constant");
    return false;
  }

  if (type.isTFString()) {
    if (!value.isa<StringAttr>())
      return emitOpError("requires 'value' to be a string constant");
    return false;
  }

  if (type.isa<FunctionType>()) {
    if (!value.isa<FunctionAttr>())
      return emitOpError("requires 'value' to be a function reference");
    return false;
  }

  return emitOpError(
      "requires a result type that aligns with the 'value' attribute");
}

Attribute ConstantOp::constantFold(ArrayRef<Attribute> operands,
                                   MLIRContext *context) const {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantFloatOp::build(Builder *builder, OperationState *result,
                            const APFloat &value, FloatType type) {
  ConstantOp::build(builder, result, builder->getFloatAttr(type, value), type);
}

bool ConstantFloatOp::isClassFor(const OperationInst *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType().isa<FloatType>();
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const OperationInst *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType().isa<IntegerType>();
}

void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, unsigned width) {
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, builder->getIntegerAttr(type, value),
                    type);
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, Type type) {
  assert(type.isa<IntegerType>() && "ConstantIntOp can only have integer type");
  ConstantOp::build(builder, result, builder->getIntegerAttr(type, value),
                    type);
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::isClassFor(const OperationInst *op) {
  return ConstantOp::isClassFor(op) && op->getResult(0)->getType().isIndex();
}

void ConstantIndexOp::build(Builder *builder, OperationState *result,
                            int64_t value) {
  Type type = builder->getIndexType();
  ConstantOp::build(builder, result, builder->getIntegerAttr(type, value),
                    type);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::build(Builder *builder, OperationState *result,
                     ArrayRef<Value *> results) {
  result->addOperands(results);
}

bool ReturnOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc;
  return parser->getCurrentLocation(&loc) || parser->parseOperandList(opInfo) ||
         (!opInfo.empty() && parser->parseColonTypeList(types)) ||
         parser->resolveOperands(opInfo, types, loc, result->operands);
}

void ReturnOp::print(OpAsmPrinter *p) const {
  *p << "return";
  if (getNumOperands() > 0) {
    *p << ' ';
    p->printOperands(operand_begin(), operand_end());
    *p << " : ";
    interleave(
        operand_begin(), operand_end(),
        [&](const Value *e) { p->printType(e->getType()); },
        [&]() { *p << ", "; });
  }
}

bool ReturnOp::verify() const {
  auto *function = cast<OperationInst>(getOperation())->getFunction();

  // The operand number and types must match the function signature.
  const auto &results = function->getType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has " + Twine(getNumOperands()) +
                       " operands, but enclosing function returns " +
                       Twine(results.size()));

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i)->getType() != results[i])
      return emitError("type of return operand " + Twine(i) +
                       " doesn't match function result type");

  return false;
}
