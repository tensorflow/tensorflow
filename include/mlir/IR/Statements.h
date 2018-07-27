//===- Statements.h - MLIR ML Statement Classes -----------------*- C++ -*-===//
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
//
// This file defines classes for special kinds of ML Function statements.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STATEMENTS_H
#define MLIR_IR_STATEMENTS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Statement.h"
#include "mlir/IR/StmtBlock.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

/// Operation statements represent operations inside ML functions.
class OperationStmt final
    : public Operation,
      public Statement,
      private llvm::TrailingObjects<OperationStmt, StmtOperand, StmtResult> {
public:
  /// Create a new OperationStmt with the specific fields.
  static OperationStmt *create(Identifier name, ArrayRef<MLValue *> operands,
                               ArrayRef<Type *> resultTypes,
                               ArrayRef<NamedAttribute> attributes,
                               MLIRContext *context);

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return numOperands; }

  MLValue *getOperand(unsigned idx) { return getStmtOperand(idx).get(); }
  const MLValue *getOperand(unsigned idx) const {
    return getStmtOperand(idx).get();
  }
  void setOperand(unsigned idx, MLValue *value) {
    return getStmtOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<OperationStmt, MLValue>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const OperationStmt, const MLValue>;

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }

  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

  ArrayRef<StmtOperand> getStmtOperands() const {
    return {getTrailingObjects<StmtOperand>(), numOperands};
  }
  MutableArrayRef<StmtOperand> getStmtOperands() {
    return {getTrailingObjects<StmtOperand>(), numOperands};
  }

  StmtOperand &getStmtOperand(unsigned idx) { return getStmtOperands()[idx]; }
  const StmtOperand &getStmtOperand(unsigned idx) const {
    return getStmtOperands()[idx];
  }

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  unsigned getNumResults() const { return numResults; }

  MLValue *getResult(unsigned idx) { return &getStmtResult(idx); }
  const MLValue *getResult(unsigned idx) const { return &getStmtResult(idx); }

  // Support non-const result iteration.
  typedef ResultIterator<OperationStmt, MLValue> result_iterator;
  result_iterator result_begin() { return result_iterator(this, 0); }
  result_iterator result_end() {
    return result_iterator(this, getNumResults());
  }
  llvm::iterator_range<result_iterator> getResults() {
    return {result_begin(), result_end()};
  }

  // Support const operand iteration.
  typedef ResultIterator<const OperationStmt, const MLValue>
      const_result_iterator;
  const_result_iterator result_begin() const {
    return const_result_iterator(this, 0);
  }

  const_result_iterator result_end() const {
    return const_result_iterator(this, getNumResults());
  }

  llvm::iterator_range<const_result_iterator> getResults() const {
    return {result_begin(), result_end()};
  }

  ArrayRef<StmtResult> getStmtResults() const {
    return {getTrailingObjects<StmtResult>(), numResults};
  }

  MutableArrayRef<StmtResult> getStmtResults() {
    return {getTrailingObjects<StmtResult>(), numResults};
  }

  StmtResult &getStmtResult(unsigned idx) { return getStmtResults()[idx]; }

  const StmtResult &getStmtResult(unsigned idx) const {
    return getStmtResults()[idx];
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Unlink this statement from its StmtBlock and delete it.
  void eraseFromBlock();

  void destroy();

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::Operation;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Statement;
  }

private:
  const unsigned numOperands, numResults;

  OperationStmt(Identifier name, unsigned numOperands, unsigned numResults,
                ArrayRef<NamedAttribute> attributes, MLIRContext *context);
  ~OperationStmt();

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperationStmt, StmtOperand, StmtResult>;
  size_t numTrailingObjects(OverloadToken<StmtOperand>) const {
    return numOperands;
  }
  size_t numTrailingObjects(OverloadToken<StmtResult>) const {
    return numResults;
  }
};

/// For statement represents an affine loop nest.
class ForStmt : public Statement, public StmtBlock {
public:
  // TODO: lower and upper bounds should be affine maps with
  // dimension and symbol use lists.
  explicit ForStmt(AffineConstantExpr *lowerBound,
                   AffineConstantExpr *upperBound, AffineConstantExpr *step)
      : Statement(Kind::For), StmtBlock(StmtBlockKind::For),
        lowerBound(lowerBound), upperBound(upperBound), step(step) {}

  // Loop bounds and step are immortal objects and don't need to be deleted.
  ~ForStmt() {}

  // TODO: represent induction variable
  AffineConstantExpr *getLowerBound() const { return lowerBound; }
  AffineConstantExpr *getUpperBound() const { return upperBound; }
  AffineConstantExpr *getStep() const { return step; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::For;
  }

  static bool classof(const StmtBlock *block) {
    return block->getStmtBlockKind() == StmtBlockKind::For;
  }

  /// Returns true if there are no more for stmt's nested under this for stmt.
  bool isInnermost() const { return 1 == getNumNestedLoops(); }

private:
  AffineConstantExpr *lowerBound;
  AffineConstantExpr *upperBound;
  AffineConstantExpr *step;
};

/// An if clause represents statements contained within a then or an else clause
/// of an if statement.
class IfClause : public StmtBlock {
public:
  explicit IfClause(IfStmt *stmt)
      : StmtBlock(StmtBlockKind::IfClause), ifStmt(stmt) {
    assert(stmt != nullptr && "If clause must have non-null parent");
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast
  static bool classof(const StmtBlock *block) {
    return block->getStmtBlockKind() == StmtBlockKind::IfClause;
  }

  ~IfClause() {}

  /// Returns the if statement that contains this clause.
  IfStmt *getIf() const { return ifStmt; }

private:
  IfStmt *ifStmt;
};

/// If statement restricts execution to a subset of the loop iteration space.
class IfStmt : public Statement {
public:
  explicit IfStmt()
    : Statement(Kind::If), thenClause(new IfClause(this)),
      elseClause(nullptr) {}

  ~IfStmt();

  IfClause *getThenClause() const { return thenClause; }
  IfClause *getElseClause() const { return elseClause; }
  bool hasElseClause() const { return elseClause != nullptr; }
  IfClause *createElseClause() { return (elseClause = new IfClause(this)); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::If;
  }
private:
  IfClause *thenClause;
  IfClause *elseClause;
  // TODO: Represent IntegerSet condition
};
} //end namespace mlir

#endif  // MLIR_IR_STATEMENTS_H
