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
// Operation class
//===----------------------------------------------------------------------===//

Operation::Operation(bool isInstruction, OperationName name,
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
  return llvm::cast<OperationStmt>(this)->getContext();
}

/// The source location the operation was defined or derived from.  Note that
/// it is possible for this pointer to be null.
Location Operation::getLoc() const {
  return llvm::cast<OperationStmt>(this)->getLoc();
}

/// Set the source location the operation was defined or derived from.
void Operation::setLoc(Location loc) {
  llvm::cast<OperationStmt>(this)->setLoc(loc);
}

/// Return the function this operation is defined in.
Function *Operation::getOperationFunction() {
  return llvm::cast<OperationStmt>(this)->getFunction();
}

/// Return the number of operands this operation has.
unsigned Operation::getNumOperands() const {
  return llvm::cast<OperationStmt>(this)->getNumOperands();
}

SSAValue *Operation::getOperand(unsigned idx) {
  return llvm::cast<OperationStmt>(this)->getOperand(idx);
}

void Operation::setOperand(unsigned idx, SSAValue *value) {
  auto *stmt = llvm::cast<OperationStmt>(this);
  stmt->setOperand(idx, llvm::cast<MLValue>(value));
}

/// Return the number of results this operation has.
unsigned Operation::getNumResults() const {
  return llvm::cast<OperationStmt>(this)->getNumResults();
}

/// Return the indicated result.
SSAValue *Operation::getResult(unsigned idx) {
  return llvm::cast<OperationStmt>(this)->getResult(idx);
}

unsigned Operation::getNumSuccessors() const {
  assert(isTerminator() && "Only terminators have successors.");
  return llvm::cast<OperationStmt>(this)->getNumSuccessors();
}

unsigned Operation::getNumSuccessorOperands(unsigned index) const {
  assert(isTerminator() && "Only terminators have successors.");
  return llvm::cast<OperationStmt>(this)->getNumSuccessorOperands(index);
}
BasicBlock *Operation::getSuccessor(unsigned index) {
  assert(isTerminator() && "Only terminators have successors");
  return llvm::cast<OperationStmt>(this)->getSuccessor(index);
}
void Operation::setSuccessor(BasicBlock *block, unsigned index) {
  assert(isTerminator() && "Only terminators have successors");
  llvm::cast<OperationStmt>(this)->setSuccessor(block, index);
}

void Operation::eraseSuccessorOperand(unsigned succIndex, unsigned opIndex) {
  assert(isTerminator() && "Only terminators have successors");
  return llvm::cast<OperationStmt>(this)->eraseSuccessorOperand(succIndex,
                                                                opIndex);
}
auto Operation::getSuccessorOperands(unsigned index) const
    -> llvm::iterator_range<const_operand_iterator> {
  assert(isTerminator() && "Only terminators have successors.");
  unsigned succOperandIndex =
      llvm::cast<OperationStmt>(this)->getSuccessorOperandIndex(index);
  return {const_operand_iterator(this, succOperandIndex),
          const_operand_iterator(this, succOperandIndex +
                                           getNumSuccessorOperands(index))};
}
auto Operation::getSuccessorOperands(unsigned index)
    -> llvm::iterator_range<operand_iterator> {
  assert(isTerminator() && "Only terminators have successors.");
  unsigned succOperandIndex =
      llvm::cast<OperationStmt>(this)->getSuccessorOperandIndex(index);
  return {operand_iterator(this, succOperandIndex),
          operand_iterator(this,
                           succOperandIndex + getNumSuccessorOperands(index))};
}

/// Return true if there are no users of any results of this operation.
bool Operation::use_empty() const {
  for (auto *result : getResults())
    if (!result->use_empty())
      return false;
  return true;
}

void Operation::moveBefore(Operation *existingOp) {
  return llvm::cast<OperationStmt>(this)->moveBefore(
      llvm::cast<OperationStmt>(existingOp));
}

ArrayRef<NamedAttribute> Operation::getAttrs() const {
  if (!attrs)
    return {};
  return attrs->getElements();
}

/// If an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
void Operation::setAttr(Identifier name, Attribute value) {
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
/// any diagnostic handlers that may be listening.  This function always returns
/// true.  NOTE: This may terminate the containing application, only use when
/// the IR is in an inconsistent state.
bool Operation::emitError(const Twine &message) const {
  return getContext()->emitError(getLoc(), message);
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
bool Operation::emitOpError(const Twine &message) const {
  return emitError(Twine('\'') + getName().getStringRef() + "' op " + message);
}

/// Remove this operation from its parent block and delete it.
void Operation::erase() {
  return llvm::cast<OperationStmt>(this)->erase();
}

/// Attempt to constant fold this operation with the specified constant
/// operand values.  If successful, this returns false and fills in the
/// results vector.  If not, this returns true and results is unspecified.
bool Operation::constantFold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<Attribute> &results) const {
  if (auto *abstractOp = getAbstractOperation()) {
    // If we have a registered operation definition matching this one, use it to
    // try to constant fold the operation.
    if (!abstractOp->constantFoldHook(this, operands, results))
      return false;

    // Otherwise, fall back on the dialect hook to handle it.
    return abstractOp->dialect.constantFoldHook(this, operands, results);
  }

  // If this operation hasn't been registered or doesn't have abstract
  // operation, fall back to a dialect which matches the prefix.
  auto opName = getName().getStringRef();
  if (auto *dialect = getContext()->getRegisteredDialect(opName)) {
    return dialect->constantFoldHook(this, operands, results);
  }

  return true;
}

void Operation::print(raw_ostream &os) const {
  return llvm::cast<OperationStmt>(this)->print(os);
}

void Operation::dump() const {
  return llvm::cast<OperationStmt>(this)->dump();
}

/// Methods for support type inquiry through isa, cast, and dyn_cast.
bool Operation::classof(const Statement *stmt) {
  return stmt->getKind() == Statement::Kind::Operation;
}
bool Operation::classof(const IROperandOwner *ptr) {
  return ptr->getKind() == IROperandOwner::Kind::Instruction ||
         ptr->getKind() == IROperandOwner::Kind::OperationStmt;
}

/// We need to teach the LLVM cast/dyn_cast etc logic how to cast from an
/// IROperandOwner* to Operation*.  This can't be done with a simple pointer to
/// pointer cast because the pointer adjustment depends on whether the Owner is
/// dynamically an Instruction or Statement, because of multiple inheritance.
Operation *
llvm::cast_convert_val<mlir::Operation, mlir::IROperandOwner *,
                       mlir::IROperandOwner *>::doit(const mlir::IROperandOwner
                                                         *value) {
  // TODO(clattner): obsolete this.
  const Operation *op;
  auto *ptr = cast<OperationStmt>(value);
  op = ptr;
  return const_cast<Operation *>(op);
}

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

bool OpTrait::impl::verifyZeroOperands(const Operation *op) {
  if (op->getNumOperands() != 0)
    return op->emitOpError("requires zero operands");
  return false;
}

bool OpTrait::impl::verifyOneOperand(const Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("requires a single operand");
  return false;
}

bool OpTrait::impl::verifyNOperands(const Operation *op, unsigned numOperands) {
  if (op->getNumOperands() != numOperands) {
    return op->emitOpError("expected " + Twine(numOperands) +
                           " operands, but found " +
                           Twine(op->getNumOperands()));
  }
  return false;
}

bool OpTrait::impl::verifyAtLeastNOperands(const Operation *op,
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

bool OpTrait::impl::verifyOperandsAreIntegerLike(const Operation *op) {
  for (auto *operand : op->getOperands()) {
    auto type = getTensorOrVectorElementType(operand->getType());
    if (!type.isIntOrIndex())
      return op->emitOpError("requires an integer or index type");
  }
  return false;
}

bool OpTrait::impl::verifySameTypeOperands(const Operation *op) {
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

bool OpTrait::impl::verifyZeroResult(const Operation *op) {
  if (op->getNumResults() != 0)
    return op->emitOpError("requires zero results");
  return false;
}

bool OpTrait::impl::verifyOneResult(const Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError("requires one result");
  return false;
}

bool OpTrait::impl::verifyNResults(const Operation *op, unsigned numOperands) {
  if (op->getNumResults() != numOperands)
    return op->emitOpError("expected " + Twine(numOperands) + " results");
  return false;
}

bool OpTrait::impl::verifyAtLeastNResults(const Operation *op,
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

bool OpTrait::impl::verifySameOperandsAndResultShape(const Operation *op) {
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

bool OpTrait::impl::verifySameOperandsAndResultType(const Operation *op) {
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
    llvm::iterator_range<Operation::const_operand_iterator> operands,
    const BasicBlock *destBB, const Operation *op) {
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

static bool verifyTerminatorSuccessors(const Operation *op) {
  // Verify that the operands lines up with the BB arguments in the successor.
  const Function *fn = op->getOperationFunction();
  for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
    auto *succ = op->getSuccessor(i);
    if (succ->getFunction() != fn)
      return op->emitError("reference to block defined in another function");
    if (verifyBBArguments(op->getSuccessorOperands(i), succ, op))
      return true;
  }
  return false;
}

bool OpTrait::impl::verifyIsTerminator(const Operation *op) {
  // Verify that the operation is at the end of the respective parent block.
  if (op->getOperationFunction()->isML()) {
    auto *stmt = cast<OperationStmt>(op);
    StmtBlock *block = stmt->getBlock();
    if (!block || block->getContainingStmt() || &block->back() != stmt)
      return op->emitOpError("must be the last statement in the ML function");
  } else {
    auto *inst = cast<OperationInst>(op);
    const BasicBlock *block = inst->getBlock();
    if (!block || &block->back() != inst)
      return op->emitOpError(
          "must be the last instruction in the parent basic block.");
  }

  // Verify the state of the successor blocks.
  if (op->getNumSuccessors() != 0 && verifyTerminatorSuccessors(op))
    return true;
  return false;
}

bool OpTrait::impl::verifyResultsAreBoolLike(const Operation *op) {
  for (auto *result : op->getResults()) {
    auto elementType = getTensorOrVectorElementType(result->getType());
    auto intType = elementType.dyn_cast<IntegerType>();
    bool isBoolType = intType && intType.getWidth() == 1;
    if (!isBoolType)
      return op->emitOpError("requires a bool result type");
  }

  return false;
}

bool OpTrait::impl::verifyResultsAreFloatLike(const Operation *op) {
  for (auto *result : op->getResults()) {
    if (!getTensorOrVectorElementType(result->getType()).isa<FloatType>())
      return op->emitOpError("requires a floating point type");
  }

  return false;
}

bool OpTrait::impl::verifyResultsAreIntegerLike(const Operation *op) {
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

void impl::buildBinaryOp(Builder *builder, OperationState *result,
                         SSAValue *lhs, SSAValue *rhs) {
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

void impl::printBinaryOp(const Operation *op, OpAsmPrinter *p) {
  *p << op->getName() << ' ' << *op->getOperand(0) << ", "
     << *op->getOperand(1);
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// CastOp implementation
//===----------------------------------------------------------------------===//

void impl::buildCastOp(Builder *builder, OperationState *result,
                       SSAValue *source, Type destType) {
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

void impl::printCastOp(const Operation *op, OpAsmPrinter *p) {
  *p << op->getName() << ' ' << *op->getOperand(0) << " : "
     << op->getOperand(0)->getType() << " to " << op->getResult(0)->getType();
}
