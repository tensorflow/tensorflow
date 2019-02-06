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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

BuiltinDialect::BuiltinDialect(MLIRContext *context)
    : Dialect(/*namePrefix=*/"", context) {
  addOperations<BranchOp, CondBranchOp, ConstantOp, ReturnOp>();
  addTypes<FunctionType, IndexType, UnknownType, FloatType, IntegerType,
           VectorType, RankedTensorType, UnrankedTensorType, MemRefType>();
}

void mlir::printDimAndSymbolList(Instruction::const_operand_iterator begin,
                                 Instruction::const_operand_iterator end,
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
// BranchOp
//===----------------------------------------------------------------------===//

void BranchOp::build(Builder *builder, OperationState *result, Block *dest,
                     ArrayRef<Value *> operands) {
  result->addSuccessor(dest, operands);
}

bool BranchOp::parse(OpAsmParser *parser, OperationState *result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser->parseSuccessorAndUseList(dest, destOperands))
    return true;
  result->addSuccessor(dest, destOperands);
  return false;
}

void BranchOp::print(OpAsmPrinter *p) const {
  *p << "br ";
  p->printSuccessorAndUseList(getInstruction(), 0);
}

Block *BranchOp::getDest() { return getInstruction()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getInstruction()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getInstruction()->eraseSuccessorOperand(0, index);
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

namespace {
/// cond_br true, ^bb1, ^bb2 -> br ^bb1
/// cond_br false, ^bb1, ^bb2 -> br ^bb2
///
struct SimplifyConstCondBranchPred : public RewritePattern {
  SimplifyConstCondBranchPred(MLIRContext *context)
      : RewritePattern(CondBranchOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Instruction *op) const override {
    auto condbr = op->cast<CondBranchOp>();
    if (matchPattern(condbr->getCondition(), m_Op<ConstantOp>()))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(Instruction *op, PatternRewriter &rewriter) const override {
    auto condbr = op->cast<CondBranchOp>();
    Block *foldedDest;
    SmallVector<Value *, 4> branchArgs;

    // If the condition is known to evaluate to false we fold to a branch to the
    // false destination. Otherwise, we fold to a branch to the true
    // destination.
    if (matchPattern(condbr->getCondition(), m_Zero())) {
      foldedDest = condbr->getFalseDest();
      branchArgs.assign(condbr->false_operand_begin(),
                        condbr->false_operand_end());
    } else {
      foldedDest = condbr->getTrueDest();
      branchArgs.assign(condbr->true_operand_begin(),
                        condbr->true_operand_end());
    }

    rewriter.replaceOpWithNewOp<BranchOp>(op, foldedDest, branchArgs);
  }
};
} // end anonymous namespace.

void CondBranchOp::build(Builder *builder, OperationState *result,
                         Value *condition, Block *trueDest,
                         ArrayRef<Value *> trueOperands, Block *falseDest,
                         ArrayRef<Value *> falseOperands) {
  result->addOperands(condition);
  result->addSuccessor(trueDest, trueOperands);
  result->addSuccessor(falseDest, falseOperands);
}

bool CondBranchOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<Value *, 4> destOperands;
  Block *dest;
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
  p->printSuccessorAndUseList(getInstruction(), trueIndex);
  *p << ", ";
  p->printSuccessorAndUseList(getInstruction(), falseIndex);
}

bool CondBranchOp::verify() const {
  if (!getCondition()->getType().isInteger(1))
    return emitOpError("expected condition type was boolean (i1)");
  return false;
}

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyConstCondBranchPred>(context));
}

Block *CondBranchOp::getTrueDest() {
  return getInstruction()->getSuccessor(trueIndex);
}

Block *CondBranchOp::getFalseDest() {
  return getInstruction()->getSuccessor(falseIndex);
}

unsigned CondBranchOp::getNumTrueOperands() const {
  return getInstruction()->getNumSuccessorOperands(trueIndex);
}

void CondBranchOp::eraseTrueOperand(unsigned index) {
  getInstruction()->eraseSuccessorOperand(trueIndex, index);
}

unsigned CondBranchOp::getNumFalseOperands() const {
  return getInstruction()->getNumSuccessorOperands(falseIndex);
}

void CondBranchOp::eraseFalseOperand(unsigned index) {
  getInstruction()->eraseSuccessorOperand(falseIndex, index);
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

/// Builds a constant op with the specified attribute value and result type.
void ConstantOp::build(Builder *builder, OperationState *result, Type type,
                       Attribute value) {
  auto attr = value.dyn_cast<NumericAttr>();
  assert(attr && "expected numeric value");
  assert(attr.getType() == type && "value should be of the given type");
  (void)attr;

  result->addAttribute("value", value);
  result->types.push_back(type);
}

void ConstantOp::build(Builder *builder, OperationState *result,
                       NumericAttr value) {
  result->addAttribute("value", value);
  result->types.push_back(value.getType());
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
  ConstantOp::build(builder, result, type, builder->getFloatAttr(type, value));
}

bool ConstantFloatOp::isClassFor(const Instruction *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType().isa<FloatType>();
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::isClassFor(const Instruction *op) {
  return ConstantOp::isClassFor(op) &&
         op->getResult(0)->getType().isa<IntegerType>();
}

void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, unsigned width) {
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(Builder *builder, OperationState *result,
                          int64_t value, Type type) {
  assert(type.isa<IntegerType>() && "ConstantIntOp can only have integer type");
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::isClassFor(const Instruction *op) {
  return ConstantOp::isClassFor(op) && op->getResult(0)->getType().isIndex();
}

void ConstantIndexOp::build(Builder *builder, OperationState *result,
                            int64_t value) {
  Type type = builder->getIndexType();
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
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
  auto *function = getInstruction()->getFunction();

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
