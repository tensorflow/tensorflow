//===- Operation.cpp - Operation support code -----------------------------===//
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

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Statements.h"
using namespace mlir;

/// Form the OperationName for an op with the specified string.  This either is
/// a reference to an AbstractOperation if one is known, or a uniqued Identifier
/// if not.
OperationName::OperationName(StringRef name, MLIRContext *context) {
  if (auto *op = AbstractOperation::lookup(name, context))
    representation = op;
  else
    representation = Identifier::get(name, context);
}

/// Return the name of this operation.  This always succeeds.
StringRef OperationName::getStringRef() const {
  if (auto *op = representation.dyn_cast<const AbstractOperation *>())
    return op->name;
  return representation.get<Identifier>().strref();
}

const AbstractOperation *OperationName::getAbstractOperation() const {
  return representation.dyn_cast<const AbstractOperation *>();
}

OperationName OperationName::getFromOpaquePointer(void *pointer) {
  return OperationName(RepresentationUnion::getFromOpaqueValue(pointer));
}

OpAsmParser::~OpAsmParser() {}

//===----------------------------------------------------------------------===//
// OpState trait class.
//===----------------------------------------------------------------------===//

// The fallback for the parser is to reject the short form.
bool OpState::parse(OpAsmParser *parser, OperationState *result) {
  return parser->emitError(parser->getNameLoc(), "has no concise form");
}

// The fallback for the printer is to print it the longhand form.
void OpState::print(OpAsmPrinter *p) const {
  p->printDefaultOp(getOperation());
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  NOTE: This may terminate
/// the containing application, only use when the IR is in an inconsistent
/// state.
bool OpState::emitError(const Twine &message) const {
  return getOperation()->emitError(message);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
bool OpState::emitOpError(const Twine &message) const {
  return getOperation()->emitOpError(message);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void OpState::emitWarning(const Twine &message) const {
  getOperation()->emitWarning(message);
}

/// Emit a note about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void OpState::emitNote(const Twine &message) const {
  getOperation()->emitNote(message);
}

//===----------------------------------------------------------------------===//
// Op Trait implementations
//===----------------------------------------------------------------------===//

bool OpTrait::impl::verifyZeroOperands(const OperationInst *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError("requires zero operands");
  return false;
}

bool OpTrait::impl::verifyOneOperand(const OperationInst *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("requires a single operand");
  return false;
}

bool OpTrait::impl::verifyNOperands(const OperationInst *op,
                                    unsigned numOperands) {
  if (op->getNumOperands() != numOperands) {
    return op->emitOpError("expected " + Twine(numOperands) +
                           " operands, but found " +
                           Twine(op->getNumOperands()));
  }
  return false;
}

bool OpTrait::impl::verifyAtLeastNOperands(const OperationInst *op,
                                           unsigned numOperands) {
  if (op->getNumOperands() < numOperands)
    return op->emitOpError("expected " + Twine(numOperands) +
                           " or more operands");
  return false;
}

/// If this is a vector type, or a tensor type, return the scalar element type
/// that it is built around, otherwise return the type unmodified.
static Type getTensorOrVectorElementType(Type type) {
  if (auto vec = type.dyn_cast<VectorType>())
    return vec.getElementType();

  // Look through tensor<vector<...>> to find the underlying element type.
  if (auto tensor = type.dyn_cast<TensorType>())
    return getTensorOrVectorElementType(tensor.getElementType());
  return type;
}

bool OpTrait::impl::verifyOperandsAreIntegerLike(const OperationInst *op) {
  for (auto *operand : op->getOperands()) {
    auto type = getTensorOrVectorElementType(operand->getType());
    if (!type.isIntOrIndex())
      return op->emitOpError("requires an integer or index type");
  }
  return false;
}

bool OpTrait::impl::verifySameTypeOperands(const OperationInst *op) {
  // Zero or one operand always have the "same" type.
  unsigned nOperands = op->getNumOperands();
  if (nOperands < 2)
    return false;

  auto type = op->getOperand(0)->getType();
  for (unsigned i = 1; i < nOperands; ++i) {
    if (op->getOperand(i)->getType() != type)
      return op->emitOpError("requires all operands to have the same type");
  }
  return false;
}

bool OpTrait::impl::verifyZeroResult(const OperationInst *op) {
  if (op->getNumResults() != 0)
    return op->emitOpError("requires zero results");
  return false;
}

bool OpTrait::impl::verifyOneResult(const OperationInst *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError("requires one result");
  return false;
}

bool OpTrait::impl::verifyNResults(const OperationInst *op,
                                   unsigned numOperands) {
  if (op->getNumResults() != numOperands)
    return op->emitOpError("expected " + Twine(numOperands) + " results");
  return false;
}

bool OpTrait::impl::verifyAtLeastNResults(const OperationInst *op,
                                          unsigned numOperands) {
  if (op->getNumResults() < numOperands)
    return op->emitOpError("expected " + Twine(numOperands) +
                           " or more results");
  return false;
}

/// Returns false if the given two types have the same shape. That is,
/// they are both scalars, or they are both vectors / ranked tensors with
/// the same dimension specifications. The element type does not matter.
static bool verifyShapeMatch(Type type1, Type type2) {
  // Check scalar cases
  if (type1.isIntOrIndexOrFloat())
    return !type2.isIntOrIndexOrFloat();

  // Check unranked tensor cases
  if (type1.isa<UnrankedTensorType>() || type2.isa<UnrankedTensorType>())
    return true;

  // Check normal vector/tensor cases
  if (auto vtType1 = type1.dyn_cast<VectorOrTensorType>()) {
    auto vtType2 = type2.dyn_cast<VectorOrTensorType>();
    return !(vtType2 && vtType1.getShape() == vtType2.getShape());
  }

  return false;
}

bool OpTrait::impl::verifySameOperandsAndResultShape(const OperationInst *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return true;

  auto type = op->getOperand(0)->getType();
  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    if (verifyShapeMatch(op->getResult(i)->getType(), type))
      return op->emitOpError(
          "requires the same shape for all operands and results");
  }
  for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i) {
    if (verifyShapeMatch(op->getOperand(i)->getType(), type))
      return op->emitOpError(
          "requires the same shape for all operands and results");
  }
  return false;
}

bool OpTrait::impl::verifySameOperandsAndResultType(const OperationInst *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return true;

  auto type = op->getResult(0)->getType();
  for (unsigned i = 1, e = op->getNumResults(); i < e; ++i) {
    if (op->getResult(i)->getType() != type)
      return op->emitOpError(
          "requires the same type for all operands and results");
  }
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i)->getType() != type)
      return op->emitOpError(
          "requires the same type for all operands and results");
  }
  return false;
}

static bool verifyBBArguments(
    llvm::iterator_range<OperationInst::const_operand_iterator> operands,
    const BasicBlock *destBB, const OperationInst *op) {
  unsigned operandCount = std::distance(operands.begin(), operands.end());
  if (operandCount != destBB->getNumArguments())
    return op->emitError("branch has " + Twine(operandCount) +
                         " operands, but target block has " +
                         Twine(destBB->getNumArguments()));

  auto operandIt = operands.begin();
  for (unsigned i = 0, e = operandCount; i != e; ++i, ++operandIt) {
    if ((*operandIt)->getType() != destBB->getArgument(i)->getType())
      return op->emitError("type mismatch in bb argument #" + Twine(i));
  }

  return false;
}

static bool verifyTerminatorSuccessors(const OperationInst *op) {
  // Verify that the operands lines up with the BB arguments in the successor.
  const Function *fn = op->getFunction();
  for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
    auto *succ = op->getSuccessor(i);
    if (succ->getFunction() != fn)
      return op->emitError("reference to block defined in another function");
    if (verifyBBArguments(op->getSuccessorOperands(i), succ, op))
      return true;
  }
  return false;
}

bool OpTrait::impl::verifyIsTerminator(const OperationInst *op) {
  // Verify that the operation is at the end of the respective parent block.
  if (op->getFunction()->isML()) {
    StmtBlock *block = op->getBlock();
    if (!block || block->getContainingStmt() || &block->back() != op)
      return op->emitOpError("must be the last statement in the ML function");
  } else {
    const BasicBlock *block = op->getBlock();
    if (!block || &block->back() != op)
      return op->emitOpError(
          "must be the last instruction in the parent basic block.");
  }

  // Verify the state of the successor blocks.
  if (op->getNumSuccessors() != 0 && verifyTerminatorSuccessors(op))
    return true;
  return false;
}

bool OpTrait::impl::verifyResultsAreBoolLike(const OperationInst *op) {
  for (auto *result : op->getResults()) {
    auto elementType = getTensorOrVectorElementType(result->getType());
    auto intType = elementType.dyn_cast<IntegerType>();
    bool isBoolType = intType && intType.getWidth() == 1;
    if (!isBoolType)
      return op->emitOpError("requires a bool result type");
  }

  return false;
}

bool OpTrait::impl::verifyResultsAreFloatLike(const OperationInst *op) {
  for (auto *result : op->getResults()) {
    if (!getTensorOrVectorElementType(result->getType()).isa<FloatType>())
      return op->emitOpError("requires a floating point type");
  }

  return false;
}

bool OpTrait::impl::verifyResultsAreIntegerLike(const OperationInst *op) {
  for (auto *result : op->getResults()) {
    auto type = getTensorOrVectorElementType(result->getType());
    if (!type.isIntOrIndex())
      return op->emitOpError("requires an integer or index type");
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BinaryOp implementation
//===----------------------------------------------------------------------===//

// These functions are out-of-line implementations of the methods in BinaryOp,
// which avoids them being template instantiated/duplicated.

void impl::buildBinaryOp(Builder *builder, OperationState *result, Value *lhs,
                         Value *rhs) {
  assert(lhs->getType() == rhs->getType());
  result->addOperands({lhs, rhs});
  result->types.push_back(lhs->getType());
}

bool impl::parseBinaryOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  return parser->parseOperandList(ops, 2) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperands(ops, type, result->operands) ||
         parser->addTypeToList(type, result->types);
}

void impl::printBinaryOp(const OperationInst *op, OpAsmPrinter *p) {
  *p << op->getName() << ' ' << *op->getOperand(0) << ", "
     << *op->getOperand(1);
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// CastOp implementation
//===----------------------------------------------------------------------===//

void impl::buildCastOp(Builder *builder, OperationState *result, Value *source,
                       Type destType) {
  result->addOperands(source);
  result->addTypes(destType);
}

bool impl::parseCastOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType srcInfo;
  Type srcType, dstType;
  return parser->parseOperand(srcInfo) || parser->parseColonType(srcType) ||
         parser->resolveOperand(srcInfo, srcType, result->operands) ||
         parser->parseKeywordType("to", dstType) ||
         parser->addTypeToList(dstType, result->types);
}

void impl::printCastOp(const OperationInst *op, OpAsmPrinter *p) {
  *p << op->getName() << ' ' << *op->getOperand(0) << " : "
     << op->getOperand(0)->getType() << " to " << op->getResult(0)->getType();
}
