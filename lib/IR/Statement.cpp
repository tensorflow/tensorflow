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

#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/IR/Types.h"
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
  return block ? block->getParentStmt() : nullptr;
}

MLFunction *Statement::findFunction() const {
  return block ? block->findFunction() : nullptr;
}

bool Statement::isInnermost() const {
  struct NestedLoopCounter : public StmtWalker<NestedLoopCounter> {
    unsigned numNestedLoops;
    NestedLoopCounter() : numNestedLoops(0) {}
    void walkForStmt(const ForStmt *fs) { numNestedLoops++; }
  };

  NestedLoopCounter nlc;
  nlc.walk(const_cast<Statement *>(this));
  return nlc.numNestedLoops == 1;
}

Statement *Statement::clone() const {
  switch (kind) {
  case Kind::Operation:
    return cast<OperationStmt>(this)->clone();
  case Kind::If:
    llvm_unreachable("cloning for if's not implemented yet");
    return cast<IfStmt>(this)->clone();
  case Kind::For:
    return cast<ForStmt>(this)->clone();
  }
}

/// Replaces all uses of oldVal with newVal.
// TODO(bondhugula,clattner): do this more efficiently by walking those uses of
// oldVal that fall within this statement.
void Statement::replaceUses(MLValue *oldVal, MLValue *newVal) {
  struct ReplaceUseWalker : public StmtWalker<ReplaceUseWalker> {
    // Value to be replaced.
    MLValue *oldVal;
    // Value to be replaced with.
    MLValue *newVal;

    ReplaceUseWalker(MLValue *oldVal, MLValue *newVal)
        : oldVal(oldVal), newVal(newVal){};

    void visitOperationStmt(OperationStmt *os) {
      for (auto &operand : os->getStmtOperands()) {
        if (operand.get() == oldVal)
          operand.set(newVal);
      }
    }
  };

  ReplaceUseWalker ri(oldVal, newVal);
  ri.walk(this);
}

//===----------------------------------------------------------------------===//
// ilist_traits for Statement
//===----------------------------------------------------------------------===//

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
void Statement::eraseFromBlock() {
  assert(getBlock() && "Statement has no block");
  getBlock()->getStatements().erase(this);
}

//===----------------------------------------------------------------------===//
// OperationStmt
//===----------------------------------------------------------------------===//

/// Create a new OperationStmt with the specific fields.
OperationStmt *OperationStmt::create(Identifier name,
                                     ArrayRef<MLValue *> operands,
                                     ArrayRef<Type *> resultTypes,
                                     ArrayRef<NamedAttribute> attributes,
                                     MLIRContext *context) {
  auto byteSize = totalSizeToAlloc<StmtOperand, StmtResult>(operands.size(),
                                                            resultTypes.size());
  void *rawMem = malloc(byteSize);

  // Initialize the OperationStmt part of the statement.
  auto stmt = ::new (rawMem) OperationStmt(
      name, operands.size(), resultTypes.size(), attributes, context);

  // Initialize the operands and results.
  auto stmtOperands = stmt->getStmtOperands();
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    new (&stmtOperands[i]) StmtOperand(stmt, operands[i]);

  auto stmtResults = stmt->getStmtResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&stmtResults[i]) StmtResult(resultTypes[i], stmt);
  return stmt;
}

/// Clone an existing OperationStmt.
OperationStmt *OperationStmt::clone() const {
  SmallVector<MLValue *, 8> operands;
  SmallVector<Type *, 8> resultTypes;

  // TODO(clattner): switch this to iterator logic.
  // Put together operands and results.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    operands.push_back(getStmtOperand(i).get());

  for (unsigned i = 0, e = getNumResults(); i != e; ++i)
    resultTypes.push_back(getStmtResult(i).getType());

  return create(getName(), operands, resultTypes, getAttrs(), getContext());
}

OperationStmt::OperationStmt(Identifier name, unsigned numOperands,
                             unsigned numResults,
                             ArrayRef<NamedAttribute> attributes,
                             MLIRContext *context)
    : Operation(name, /*isInstruction=*/false, attributes, context),
      Statement(Kind::Operation), numOperands(numOperands),
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
    return getResult(0)->getType()->getContext();
  if (getNumOperands())
    return getOperand(0)->getType()->getContext();

  // In the very odd case where we have no operands or results, fall back to
  // doing a find.
  return findFunction()->getContext();
}

//===----------------------------------------------------------------------===//
// ForStmt
//===----------------------------------------------------------------------===//

ForStmt::ForStmt(AffineConstantExpr *lowerBound, AffineConstantExpr *upperBound,
                 AffineConstantExpr *step, MLIRContext *context)
    : Statement(Kind::For),
      MLValue(MLValueKind::ForStmt, Type::getAffineInt(context)),
      StmtBlock(StmtBlockKind::For), lowerBound(lowerBound),
      upperBound(upperBound), step(step) {}

ForStmt *ForStmt::clone() const {
  auto *forStmt = new ForStmt(getLowerBound(), getUpperBound(), getStep(),
                              Statement::findFunction()->getContext());

  // Pairs of <old op stmt result whose uses need to be replaced,
  // new result generated by the corresponding cloned op stmt>.
  SmallVector<std::pair<MLValue *, MLValue *>, 8> oldNewResultPairs;
  for (auto &s : getStatements()) {
    auto *cloneStmt = s.clone();
    forStmt->getStatements().push_back(cloneStmt);
    if (auto *opStmt = dyn_cast<OperationStmt>(&s)) {
      auto *cloneOpStmt = cast<OperationStmt>(cloneStmt);
      for (unsigned i = 0, e = opStmt->getNumResults(); i < e; i++) {
        oldNewResultPairs.push_back(
            std::make_pair(const_cast<StmtResult *>(&opStmt->getStmtResult(i)),
                           &cloneOpStmt->getStmtResult(i)));
      }
    }
  }
  // Replace uses of old op results' with the newly created ones.
  for (unsigned i = 0, e = oldNewResultPairs.size(); i < e; i++) {
    for (auto &stmt : *forStmt) {
      stmt.replaceUses(oldNewResultPairs[i].first, oldNewResultPairs[i].second);
    }
  }

  // Replace uses of old loop IV with the new one.
  forStmt->Statement::replaceUses(const_cast<ForStmt *>(this), forStmt);
  return forStmt;
}

//===----------------------------------------------------------------------===//
// IfStmt
//===----------------------------------------------------------------------===//

IfStmt::~IfStmt() {
  delete thenClause;
  if (elseClause)
    delete elseClause;
}

IfStmt *IfStmt::clone() const {
  llvm_unreachable("cloning for if's not implemented yet");
  return nullptr;
}
