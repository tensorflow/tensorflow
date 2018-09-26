//===- Operation.cpp - MLIR Operation Class -------------------------------===//
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

#include "mlir/IR/Operation.h"
#include "AttributeListStorage.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Instructions.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/Statements.h"
using namespace mlir;

Operation::Operation(bool isInstruction, Identifier name,
                     ArrayRef<NamedAttribute> attrs, MLIRContext *context)
    : nameAndIsInstruction(name, isInstruction) {
  this->attrs = AttributeListStorage::get(attrs, context);

#ifndef NDEBUG
  for (auto elt : attrs)
    assert(elt.second != nullptr && "Attributes cannot have null entries");
#endif
}

Operation::~Operation() {}

/// Return the context this operation is associated with.
MLIRContext *Operation::getContext() const {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getContext();
  return cast<OperationStmt>(this)->getContext();
}

/// The source location the operation was defined or derived from.  Note that
/// it is possible for this pointer to be null.
Location *Operation::getLoc() const {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getLoc();
  return cast<OperationStmt>(this)->getLoc();
}

/// Return the function this operation is defined in.
Function *Operation::getOperationFunction() {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getFunction();
  return cast<OperationStmt>(this)->findFunction();
}

/// Return the number of operands this operation has.
unsigned Operation::getNumOperands() const {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getNumOperands();

  return cast<OperationStmt>(this)->getNumOperands();
}

SSAValue *Operation::getOperand(unsigned idx) {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getOperand(idx);

  return cast<OperationStmt>(this)->getOperand(idx);
}

void Operation::setOperand(unsigned idx, SSAValue *value) {
  if (auto *inst = dyn_cast<OperationInst>(this)) {
    inst->setOperand(idx, cast<CFGValue>(value));
  } else {
    auto *stmt = cast<OperationStmt>(this);
    stmt->setOperand(idx, cast<MLValue>(value));
  }
}

/// Return the number of results this operation has.
unsigned Operation::getNumResults() const {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getNumResults();

  return cast<OperationStmt>(this)->getNumResults();
}

/// Return the indicated result.
SSAValue *Operation::getResult(unsigned idx) {
  if (auto *inst = dyn_cast<OperationInst>(this))
    return inst->getResult(idx);

  return cast<OperationStmt>(this)->getResult(idx);
}

ArrayRef<NamedAttribute> Operation::getAttrs() const {
  if (!attrs)
    return {};
  return attrs->getElements();
}

/// If an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
void Operation::setAttr(Identifier name, Attribute *value) {
  assert(value && "attributes may never be null");
  auto origAttrs = getAttrs();

  SmallVector<NamedAttribute, 8> newAttrs(origAttrs.begin(), origAttrs.end());
  auto *context = getContext();

  // If we already have this attribute, replace it.
  for (auto &elt : newAttrs)
    if (elt.first == name) {
      elt.second = value;
      attrs = AttributeListStorage::get(newAttrs, context);
      return;
    }

  // Otherwise, add it.
  newAttrs.push_back({name, value});
  attrs = AttributeListStorage::get(newAttrs, context);
}

/// Remove the attribute with the specified name if it exists.  The return
/// value indicates whether the attribute was present or not.
auto Operation::removeAttr(Identifier name) -> RemoveResult {
  auto origAttrs = getAttrs();
  for (unsigned i = 0, e = origAttrs.size(); i != e; ++i) {
    if (origAttrs[i].first == name) {
      SmallVector<NamedAttribute, 8> newAttrs;
      newAttrs.reserve(origAttrs.size() - 1);
      newAttrs.append(origAttrs.begin(), origAttrs.begin() + i);
      newAttrs.append(origAttrs.begin() + i + 1, origAttrs.end());
      attrs = AttributeListStorage::get(newAttrs, getContext());
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}

/// Emit a note about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void Operation::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void Operation::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  NOTE: This may terminate
/// the containing application, only use when the IR is in an inconsistent
/// state.
void Operation::emitError(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Error);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
bool Operation::emitOpError(const Twine &message) const {
  emitError(Twine('\'') + getName().str() + "' op " + message);
  return true;
}

/// Attempt to constant fold this operation with the specified constant
/// operand values.  If successful, this returns false and fills in the
/// results vector.  If not, this returns true and results is unspecified.
bool Operation::constantFold(ArrayRef<Attribute *> operands,
                             SmallVectorImpl<Attribute *> &results) const {
  // If we have a registered operation definition matching this one, use it to
  // try to constant fold the operation.
  if (auto *abstractOp = getAbstractOperation())
    if (!abstractOp->constantFoldHook(this, operands, results))
      return false;

  // TODO: Otherwise, fall back on the dialect hook to handle it.
  return true;
}

//===----------------------------------------------------------------------===//
// OpBaseState trait class.
//===----------------------------------------------------------------------===//

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  NOTE: This may terminate
/// the containing application, only use when the IR is in an inconsistent
/// state.
void OpBaseState::emitError(const Twine &message) const {
  getOperation()->emitError(message);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
bool OpBaseState::emitOpError(const Twine &message) const {
  return getOperation()->emitOpError(message);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void OpBaseState::emitWarning(const Twine &message) const {
  getOperation()->emitWarning(message);
}

/// Emit a note about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void OpBaseState::emitNote(const Twine &message) const {
  getOperation()->emitNote(message);
}

//===----------------------------------------------------------------------===//
// Op Trait implementations
//===----------------------------------------------------------------------===//

bool OpTrait::impl::verifySameOperandsAndResult(const Operation *op) {
  auto *type = op->getResult(0)->getType();
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

/// If this is a vector type, or a tensor type, return the scalar element type
/// that it is built around, otherwise return the type unmodified.
static Type *getTensorOrVectorElementType(Type *type) {
  if (auto *vec = dyn_cast<VectorType>(type))
    return vec->getElementType();

  // Look through tensor<vector<...>> to find the underlying element type.
  if (auto *tensor = dyn_cast<TensorType>(type))
    return getTensorOrVectorElementType(tensor->getElementType());
  return type;
}

bool OpTrait::impl::verifyResultsAreFloatLike(const Operation *op) {
  for (auto *result : op->getResults()) {
    if (!isa<FloatType>(getTensorOrVectorElementType(result->getType())))
      return op->emitOpError("requires a floating point type");
  }

  return false;
}

bool OpTrait::impl::verifyResultsAreIntegerLike(const Operation *op) {
  for (auto *result : op->getResults()) {
    if (!isa<IntegerType>(getTensorOrVectorElementType(result->getType())))
      return op->emitOpError("requires an integer type");
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BinaryOp implementation
//===----------------------------------------------------------------------===//

// These functions are out-of-line implementations of the methods in BinaryOp,
// which avoids them being template instantiated/duplicated.

void impl::buildBinaryOp(Builder *builder, OperationState *result,
                         SSAValue *lhs, SSAValue *rhs) {
  assert(lhs->getType() == rhs->getType());
  result->addOperands({lhs, rhs});
  result->types.push_back(lhs->getType());
}

bool impl::parseBinaryOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type *type;
  return parser->parseOperandList(ops, 2) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperands(ops, type, result->operands) ||
         parser->addTypeToList(type, result->types);
}

void impl::printBinaryOp(const Operation *op, OpAsmPrinter *p) {
  *p << op->getName() << " " << *op->getOperand(0) << ", "
     << *op->getOperand(1);
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : " << *op->getResult(0)->getType();
}
