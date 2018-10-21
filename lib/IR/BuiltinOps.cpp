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
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

void mlir::printDimAndSymbolList(Operation::const_operand_iterator begin,
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
bool mlir::parseDimAndSymbolList(OpAsmParser *parser,
                                 SmallVector<SSAValue *, 4> &operands,
                                 unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser->parseOperandList(opInfos, -1, OpAsmParser::Delimiter::Paren))
    return true;
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto *affineIntTy = parser->getBuilder().getIndexType();
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
                          AffineMap map, ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->types.append(map.getNumResults(), builder->getIndexType());
  result->addAttribute("map", builder->getAffineMapAttr(map));
}

bool AffineApplyOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  auto *affineIntTy = builder.getIndexType();

  AffineMapAttr *mapAttr;
  unsigned numDims;
  if (parser->parseAttribute(mapAttr, "map", result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  auto map = mapAttr->getValue();

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
  auto *affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return emitOpError("requires an affine map");

  // Check input and output dimensions match.
  auto map = affineMapAttr->getValue();

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
// is a CFG value or if it is an MLValue, and all the operands are valid
// dimension ids.
bool AffineApplyOp::isValidDim() const {
  for (auto *op : getOperands()) {
    if (auto *v = dyn_cast<MLValue>(op))
      if (!v->isValidDim())
        return false;
  }
  return true;
}

// The result of the affine apply operation can be used as a symbol if it is
// a CFG value or if it is an MLValue, and all the operands are symbols.
bool AffineApplyOp::isValidSymbol() const {
  for (auto *op : getOperands()) {
    if (auto *v = dyn_cast<MLValue>(op))
      if (!v->isValidSymbol())
        return false;
  }
  return true;
}

bool AffineApplyOp::constantFold(ArrayRef<Attribute *> operandConstants,
                                 SmallVectorImpl<Attribute *> &results,
                                 MLIRContext *context) const {
  auto map = getAffineMap();
  if (map.constantFold(operandConstants, results))
    return true;
  // Return false on success.
  return false;
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

/// Builds a constant op with the specified attribute value and result type.
void ConstantOp::build(Builder *builder, OperationState *result,
                       Attribute *value, Type *type) {
  result->addAttribute("value", value);
  result->types.push_back(type);
}

void ConstantOp::print(OpAsmPrinter *p) const {
  *p << "constant " << *getValue();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"value");

  if (!isa<FunctionAttr>(getValue()))
    *p << " : " << *getType();
}

bool ConstantOp::parse(OpAsmParser *parser, OperationState *result) {
  Attribute *valueAttr;
  Type *type;

  if (parser->parseAttribute(valueAttr, "value", result->attributes) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;

  // 'constant' taking a function reference doesn't get a redundant type
  // specifier.  The attribute itself carries it.
  if (auto *fnAttr = dyn_cast<FunctionAttr>(valueAttr))
    return parser->addTypeToList(fnAttr->getValue()->getType(), result->types);

  return parser->parseColonType(type) ||
         parser->addTypeToList(type, result->types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
bool ConstantOp::verify() const {
  auto *value = getValue();
  if (!value)
    return emitOpError("requires a 'value' attribute");

  auto *type = this->getType();
  if (isa<IntegerType>(type) || type->isIndex()) {
    if (!isa<IntegerAttr>(value))
      return emitOpError(
          "requires 'value' to be an integer for an integer result type");
    return false;
  }

  if (isa<FloatType>(type)) {
    if (!isa<FloatAttr>(value))
      return emitOpError("requires 'value' to be a floating point constant");
    return false;
  }

  if (type->isTFString()) {
    if (!isa<StringAttr>(value))
      return emitOpError("requires 'value' to be a string constant");
    return false;
  }

  if (isa<FunctionType>(type)) {
    if (!isa<FunctionAttr>(value))
      return emitOpError("requires 'value' to be a function reference");
    return false;
  }

  return emitOpError(
      "requires a result type that aligns with the 'value' attribute");
}

Attribute *ConstantOp::constantFold(ArrayRef<Attribute *> operands,
                                    MLIRContext *context) const {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantFloatOp::build(Builder *builder, OperationState *result,
                            const APFloat &value, FloatType *type) {
  ConstantOp::build(builder, result, builder->getFloatAttr(value), type);
}

bool ConstantFloatOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<FloatType>(op->getResult(0)->getType());
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) &&
         isa<IntegerType>(op->getResult(0)->getType());
}

void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, unsigned width) {
  ConstantOp::build(builder, result, builder->getIntegerAttr(value),
                    builder->getIntegerType(width));
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, Type *type) {
  assert(isa<IntegerType>(type) && "ConstantIntOp can only have integer type");
  ConstantOp::build(builder, result, builder->getIntegerAttr(value), type);
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::isClassFor(const Operation *op) {
  return ConstantOp::isClassFor(op) && op->getResult(0)->getType()->isIndex();
}

void ConstantIndexOp::build(Builder *builder, OperationState *result,
                            int64_t value) {
  ConstantOp::build(builder, result, builder->getIntegerAttr(value),
                    builder->getIndexType());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::build(Builder *builder, OperationState *result,
                     ArrayRef<SSAValue *> results) {
  result->addOperands(results);
}

bool ReturnOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type *, 2> types;
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
    interleave(operand_begin(), operand_end(),
               [&](const SSAValue *e) { p->printType(e->getType()); },
               [&]() { *p << ", "; });
  }
}

bool ReturnOp::verify() const {
  // ReturnOp must be part of an ML function.
  if (auto *stmt = dyn_cast<OperationStmt>(getOperation())) {
    StmtBlock *block = stmt->getBlock();
    if (!block || !isa<MLFunction>(block) || &block->back() != stmt)
      return emitOpError("must be the last statement in the ML function");

    // Return success. Checking that operand types match those in the function
    // signature is performed in the ML function verifier.
    return false;
  }
  return emitOpError("cannot occur in a CFG function");
}

//===----------------------------------------------------------------------===//
// Register operations.
//===----------------------------------------------------------------------===//

/// Install the builtin operations in the specified MLIRContext..
void mlir::registerBuiltinOperations(MLIRContext *ctx) {
  auto &opSet = OperationSet::get(ctx);
  opSet.addOperations<AffineApplyOp, ConstantOp, ReturnOp>(
      /*prefix=*/"");
}
