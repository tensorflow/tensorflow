//===- Statement.cpp - MLIR Statement Classes ----------------------------===//
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

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// StmtResult
//===------------------------------------------------------------------===//

/// Return the result number of this result.
unsigned StmtResult::getResultNumber() const {
  // Results are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getOwner()->getStmtResults()[0];
}

//===----------------------------------------------------------------------===//
// StmtOperand
//===------------------------------------------------------------------===//

/// Return which operand this is in the operand list.
template <> unsigned StmtOperand::getOperandNumber() const {
  return this - &getOwner()->getStmtOperands()[0];
}

//===----------------------------------------------------------------------===//
// Statement
//===------------------------------------------------------------------===//

// Statements are deleted through the destroy() member because we don't have
// a virtual destructor.
Statement::~Statement() {
  assert(block == nullptr && "statement destroyed but still in a block");
}

/// Destroy this statement or one of its subclasses.
void Statement::destroy() {
  switch (this->getKind()) {
  case Kind::Operation:
    cast<OperationStmt>(this)->destroy();
    break;
  case Kind::For:
    delete cast<ForStmt>(this);
    break;
  case Kind::If:
    delete cast<IfStmt>(this);
    break;
  }
}

Statement *Statement::getParentStmt() const {
  return block ? block->getContainingStmt() : nullptr;
}

MLFunction *Statement::findFunction() const {
  return block ? block->findFunction() : nullptr;
}

MLValue *Statement::getOperand(unsigned idx) {
  return getStmtOperand(idx).get();
}

const MLValue *Statement::getOperand(unsigned idx) const {
  return getStmtOperand(idx).get();
}

// MLValue can be used as a dimension id if it is valid as a symbol, or
// it is an induction variable, or it is a result of affine apply operation
// with dimension id arguments.
bool MLValue::isValidDim() const {
  if (auto *stmt = getDefiningStmt()) {
    // Top level statement or constant operation is ok.
    if (stmt->getParentStmt() == nullptr || stmt->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = stmt->dyn_cast<AffineApplyOp>())
      return op->isValidDim();
    return false;
  }
  // This value is either a function argument or an induction variable. Both
  // are ok.
  return true;
}

// MLValue can be used as a symbol if it is a constant, or it is defined at
// the top level, or it is a result of affine apply operation with symbol
// arguments.
bool MLValue::isValidSymbol() const {
  if (auto *stmt = getDefiningStmt()) {
    // Top level statement or constant operation is ok.
    if (stmt->getParentStmt() == nullptr || stmt->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = stmt->dyn_cast<AffineApplyOp>())
      return op->isValidSymbol();
    return false;
  }
  // This value is either a function argument or an induction variable.
  // Function argument is ok, induction variable is not.
  return isa<MLFuncArgument>(this);
}

void Statement::setOperand(unsigned idx, MLValue *value) {
  getStmtOperand(idx).set(value);
}

unsigned Statement::getNumOperands() const {
  switch (getKind()) {
  case Kind::Operation:
    return cast<OperationStmt>(this)->getNumOperands();
  case Kind::For:
    return cast<ForStmt>(this)->getNumOperands();
  case Kind::If:
    return cast<IfStmt>(this)->getNumOperands();
  }
}

MutableArrayRef<StmtOperand> Statement::getStmtOperands() {
  switch (getKind()) {
  case Kind::Operation:
    return cast<OperationStmt>(this)->getStmtOperands();
  case Kind::For:
    return cast<ForStmt>(this)->getStmtOperands();
  case Kind::If:
    return cast<IfStmt>(this)->getStmtOperands();
  }
}

/// Emit a note about this statement, reporting up to any diagnostic
/// handlers that may be listening.
void Statement::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this statement, reporting up to any diagnostic
/// handlers that may be listening.
void Statement::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns true.  NOTE: This may terminate the containing application, only
/// use when the IR is in an inconsistent state.
bool Statement::emitError(const Twine &message) const {
  return getContext()->emitError(getLoc(), message);
}
//===----------------------------------------------------------------------===//
// ilist_traits for Statement
//===----------------------------------------------------------------------===//

void llvm::ilist_traits<::mlir::Statement>::deleteNode(Statement *stmt) {
  stmt->destroy();
}

StmtBlock *llvm::ilist_traits<::mlir::Statement>::getContainingBlock() {
  size_t Offset(
      size_t(&((StmtBlock *)nullptr->*StmtBlock::getSublistAccess(nullptr))));
  iplist<Statement> *Anchor(static_cast<iplist<Statement> *>(this));
  return reinterpret_cast<StmtBlock *>(reinterpret_cast<char *>(Anchor) -
                                       Offset);
}

/// This is a trait method invoked when a statement is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::addNodeToList(Statement *stmt) {
  assert(!stmt->getBlock() && "already in a statement block!");
  stmt->block = getContainingBlock();
}

/// This is a trait method invoked when a statement is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::removeNodeFromList(
    Statement *stmt) {
  assert(stmt->block && "not already in a statement block!");
  stmt->block = nullptr;
}

/// This is a trait method invoked when a statement is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Statement>::transferNodesFromList(
    ilist_traits<Statement> &otherList, stmt_iterator first,
    stmt_iterator last) {
  // If we are transferring statements within the same block, the block
  // pointer doesn't need to be updated.
  StmtBlock *curParent = getContainingBlock();
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each statement.
  for (; first != last; ++first)
    first->block = curParent;
}

/// Remove this statement (and its descendants) from its StmtBlock and delete
/// all of them.
void Statement::erase() {
  assert(getBlock() && "Statement has no block");
  getBlock()->getStatements().erase(this);
}

/// Unlink this statement from its current block and insert it right before
/// `existingStmt` which may be in the same or another block in the same
/// function.
void Statement::moveBefore(Statement *existingStmt) {
  moveBefore(existingStmt->getBlock(), existingStmt->getIterator());
}

/// Unlink this operation instruction from its current basic block and insert
/// it right before `iterator` in the specified basic block.
void Statement::moveBefore(StmtBlock *block,
                           llvm::iplist<Statement>::iterator iterator) {
  block->getStatements().splice(iterator, getBlock()->getStatements(),
                                getIterator());
}

//===----------------------------------------------------------------------===//
// OperationStmt
//===----------------------------------------------------------------------===//

/// Create a new OperationStmt with the specific fields.
OperationStmt *OperationStmt::create(Location location, OperationName name,
                                     ArrayRef<MLValue *> operands,
                                     ArrayRef<Type> resultTypes,
                                     ArrayRef<NamedAttribute> attributes,
                                     MLIRContext *context) {
  auto byteSize = totalSizeToAlloc<StmtOperand, StmtResult>(operands.size(),
                                                            resultTypes.size());
  void *rawMem = malloc(byteSize);

  // Initialize the OperationStmt part of the statement.
  auto stmt = ::new (rawMem) OperationStmt(
      location, name, operands.size(), resultTypes.size(), attributes, context);

  // Initialize the operands and results.
  auto stmtOperands = stmt->getStmtOperands();
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    new (&stmtOperands[i]) StmtOperand(stmt, operands[i]);

  auto stmtResults = stmt->getStmtResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&stmtResults[i]) StmtResult(resultTypes[i], stmt);
  return stmt;
}

OperationStmt::OperationStmt(Location location, OperationName name,
                             unsigned numOperands, unsigned numResults,
                             ArrayRef<NamedAttribute> attributes,
                             MLIRContext *context)
    : Operation(/*isInstruction=*/false, name, attributes, context),
      Statement(Kind::Operation, location), numOperands(numOperands),
      numResults(numResults) {}

OperationStmt::~OperationStmt() {
  // Explicitly run the destructors for the operands and results.
  for (auto &operand : getStmtOperands())
    operand.~StmtOperand();

  for (auto &result : getStmtResults())
    result.~StmtResult();
}

void OperationStmt::destroy() {
  this->~OperationStmt();
  free(this);
}

/// Return the context this operation is associated with.
MLIRContext *OperationStmt::getContext() const {
  // If we have a result or operand type, that is a constant time way to get
  // to the context.
  if (getNumResults())
    return getResult(0)->getType().getContext();
  if (getNumOperands())
    return getOperand(0)->getType().getContext();

  // In the very odd case where we have no operands or results, fall back to
  // doing a find.
  return findFunction()->getContext();
}

bool OperationStmt::isReturn() const { return isa<ReturnOp>(); }

//===----------------------------------------------------------------------===//
// ForStmt
//===----------------------------------------------------------------------===//

ForStmt *ForStmt::create(Location location, ArrayRef<MLValue *> lbOperands,
                         AffineMap lbMap, ArrayRef<MLValue *> ubOperands,
                         AffineMap ubMap, int64_t step) {
  assert(lbOperands.size() == lbMap.getNumInputs() &&
         "lower bound operand count does not match the affine map");
  assert(ubOperands.size() == ubMap.getNumInputs() &&
         "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  unsigned numOperands = lbOperands.size() + ubOperands.size();
  ForStmt *stmt = new ForStmt(location, numOperands, lbMap, ubMap, step);

  unsigned i = 0;
  for (unsigned e = lbOperands.size(); i != e; ++i)
    stmt->operands.emplace_back(StmtOperand(stmt, lbOperands[i]));

  for (unsigned j = 0, e = ubOperands.size(); j != e; ++i, ++j)
    stmt->operands.emplace_back(StmtOperand(stmt, ubOperands[j]));

  return stmt;
}

ForStmt::ForStmt(Location location, unsigned numOperands, AffineMap lbMap,
                 AffineMap ubMap, int64_t step)
    : Statement(Kind::For, location),
      MLValue(MLValueKind::ForStmt,
              Type::getIndex(lbMap.getResult(0).getContext())),
      body(this), lbMap(lbMap), ubMap(ubMap), step(step) {
  operands.reserve(numOperands);
}

const AffineBound ForStmt::getLowerBound() const {
  return AffineBound(*this, 0, lbMap.getNumInputs(), lbMap);
}

const AffineBound ForStmt::getUpperBound() const {
  return AffineBound(*this, lbMap.getNumInputs(), getNumOperands(), ubMap);
}

void ForStmt::setLowerBound(ArrayRef<MLValue *> lbOperands, AffineMap map) {
  assert(lbOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<MLValue *, 4> ubOperands(getUpperBoundOperands());

  operands.clear();
  operands.reserve(lbOperands.size() + ubMap.getNumInputs());
  for (auto *operand : lbOperands) {
    operands.emplace_back(StmtOperand(this, operand));
  }
  for (auto *operand : ubOperands) {
    operands.emplace_back(StmtOperand(this, operand));
  }
  this->lbMap = map;
}

void ForStmt::setUpperBound(ArrayRef<MLValue *> ubOperands, AffineMap map) {
  assert(ubOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<MLValue *, 4> lbOperands(getLowerBoundOperands());

  operands.clear();
  operands.reserve(lbOperands.size() + ubOperands.size());
  for (auto *operand : lbOperands) {
    operands.emplace_back(StmtOperand(this, operand));
  }
  for (auto *operand : ubOperands) {
    operands.emplace_back(StmtOperand(this, operand));
  }
  this->ubMap = map;
}

void ForStmt::setLowerBoundMap(AffineMap map) {
  assert(lbMap.getNumDims() == map.getNumDims() &&
         lbMap.getNumSymbols() == map.getNumSymbols());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");
  this->lbMap = map;
}

void ForStmt::setUpperBoundMap(AffineMap map) {
  assert(ubMap.getNumDims() == map.getNumDims() &&
         ubMap.getNumSymbols() == map.getNumSymbols());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");
  this->ubMap = map;
}

bool ForStmt::hasConstantLowerBound() const { return lbMap.isSingleConstant(); }

bool ForStmt::hasConstantUpperBound() const { return ubMap.isSingleConstant(); }

int64_t ForStmt::getConstantLowerBound() const {
  return lbMap.getSingleConstantResult();
}

int64_t ForStmt::getConstantUpperBound() const {
  return ubMap.getSingleConstantResult();
}

void ForStmt::setConstantLowerBound(int64_t value) {
  setLowerBound({}, AffineMap::getConstantMap(value, getContext()));
}

void ForStmt::setConstantUpperBound(int64_t value) {
  setUpperBound({}, AffineMap::getConstantMap(value, getContext()));
}

ForStmt::operand_range ForStmt::getLowerBoundOperands() {
  return {operand_begin(), operand_begin() + getLowerBoundMap().getNumInputs()};
}

ForStmt::const_operand_range ForStmt::getLowerBoundOperands() const {
  return {operand_begin(), operand_begin() + getLowerBoundMap().getNumInputs()};
}

ForStmt::operand_range ForStmt::getUpperBoundOperands() {
  return {operand_begin() + getLowerBoundMap().getNumInputs(), operand_end()};
}

ForStmt::const_operand_range ForStmt::getUpperBoundOperands() const {
  return {operand_begin() + getLowerBoundMap().getNumInputs(), operand_end()};
}

bool ForStmt::matchingBoundOperandList() const {
  if (lbMap.getNumDims() != ubMap.getNumDims() ||
      lbMap.getNumSymbols() != ubMap.getNumSymbols())
    return false;

  unsigned numOperands = lbMap.getNumInputs();
  for (unsigned i = 0, e = lbMap.getNumInputs(); i < e; i++) {
    // Compare MLValue *'s.
    if (getOperand(i) != getOperand(numOperands + i))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// IfStmt
//===----------------------------------------------------------------------===//

IfStmt::IfStmt(Location location, unsigned numOperands, IntegerSet set)
    : Statement(Kind::If, location), thenClause(this), elseClause(nullptr),
      set(set) {
  operands.reserve(numOperands);
}

IfStmt::~IfStmt() {
  if (elseClause)
    delete elseClause;

  // An IfStmt's IntegerSet 'set' should not be deleted since it is
  // allocated through MLIRContext's bump pointer allocator.
}

IfStmt *IfStmt::create(Location location, ArrayRef<MLValue *> operands,
                       IntegerSet set) {
  unsigned numOperands = operands.size();
  assert(numOperands == set.getNumOperands() &&
         "operand cound does not match the integer set operand count");

  IfStmt *stmt = new IfStmt(location, numOperands, set);

  for (auto *op : operands)
    stmt->operands.emplace_back(StmtOperand(stmt, op));

  return stmt;
}

const AffineCondition IfStmt::getCondition() const {
  return AffineCondition(*this, set);
}

MLIRContext *IfStmt::getContext() const {
  // Check for degenerate case of if statement with no operands.
  // This is unlikely, but legal.
  if (operands.empty())
    return findFunction()->getContext();

  return getOperand(0)->getType().getContext();
}

//===----------------------------------------------------------------------===//
// Statement Cloning
//===----------------------------------------------------------------------===//

/// Create a deep copy of this statement, remapping any operands that use
/// values outside of the statement using the map that is provided (leaving
/// them alone if no entry is present).  Replaces references to cloned
/// sub-statements to the corresponding statement that is copied, and adds
/// those mappings to the map.
Statement *Statement::clone(DenseMap<const MLValue *, MLValue *> &operandMap,
                            MLIRContext *context) const {
  // If the specified value is in operandMap, return the remapped value.
  // Otherwise return the value itself.
  auto remapOperand = [&](const MLValue *value) -> MLValue * {
    auto it = operandMap.find(value);
    return it != operandMap.end() ? it->second : const_cast<MLValue *>(value);
  };

  SmallVector<MLValue *, 8> operands;
  operands.reserve(getNumOperands());
  for (auto *opValue : getOperands())
    operands.push_back(remapOperand(opValue));

  if (auto *opStmt = dyn_cast<OperationStmt>(this)) {
    SmallVector<Type, 8> resultTypes;
    resultTypes.reserve(opStmt->getNumResults());
    for (auto *result : opStmt->getResults())
      resultTypes.push_back(result->getType());
    auto *newOp =
        OperationStmt::create(getLoc(), opStmt->getName(), operands,
                              resultTypes, opStmt->getAttrs(), context);
    // Remember the mapping of any results.
    for (unsigned i = 0, e = opStmt->getNumResults(); i != e; ++i)
      operandMap[opStmt->getResult(i)] = newOp->getResult(i);
    return newOp;
  }

  if (auto *forStmt = dyn_cast<ForStmt>(this)) {
    auto lbMap = forStmt->getLowerBoundMap();
    auto ubMap = forStmt->getUpperBoundMap();

    auto *newFor = ForStmt::create(
        getLoc(),
        ArrayRef<MLValue *>(operands).take_front(lbMap.getNumInputs()), lbMap,
        ArrayRef<MLValue *>(operands).take_back(ubMap.getNumInputs()), ubMap,
        forStmt->getStep());

    // Remember the induction variable mapping.
    operandMap[forStmt] = newFor;

    // Recursively clone the body of the for loop.
    for (auto &subStmt : *forStmt->getBody())
      newFor->getBody()->push_back(subStmt.clone(operandMap, context));

    return newFor;
  }

  // Otherwise, we must have an If statement.
  auto *ifStmt = cast<IfStmt>(this);
  auto *newIf = IfStmt::create(getLoc(), operands, ifStmt->getIntegerSet());

  auto *resultThen = newIf->getThen();
  for (auto &childStmt : *ifStmt->getThen())
    resultThen->push_back(childStmt.clone(operandMap, context));

  if (ifStmt->hasElse()) {
    auto *resultElse = newIf->createElse();
    for (auto &childStmt : *ifStmt->getElse())
      resultElse->push_back(childStmt.clone(operandMap, context));
  }

  return newIf;
}
