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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/StmtBlock.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class AffineBound;
class IntegerSet;
class AffineCondition;
class OperationStmt;

/// Operation statements represent operations inside ML functions.
class OperationStmt final
    : public Operation,
      public Statement,
      private llvm::TrailingObjects<OperationStmt, StmtOperand, StmtResult,
                                    StmtBlockOperand, unsigned> {
public:
  /// Create a new OperationStmt with the specific fields.
  static OperationStmt *
  create(Location location, OperationName name, ArrayRef<MLValue *> operands,
         ArrayRef<Type> resultTypes, ArrayRef<NamedAttribute> attributes,
         ArrayRef<StmtBlock *> successors, MLIRContext *context);

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const;

  using Statement::dump;
  using Statement::emitError;
  using Statement::emitNote;
  using Statement::emitWarning;
  using Statement::getLoc;
  using Statement::moveBefore;
  using Statement::print;
  using Statement::setLoc;

  /// Check if this statement is a return statement.
  bool isReturn() const;

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

  /// Returns an iterator on the underlying MLValue's (MLValue *).
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

  /// Returns a const iterator on the underlying MLValue's (MLValue *).
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

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  unsigned getNumResults() const { return numResults; }

  MLValue *getResult(unsigned idx) { return &getStmtResult(idx); }
  const MLValue *getResult(unsigned idx) const { return &getStmtResult(idx); }

  // Support non-const result iteration.
  using result_iterator = ResultIterator<OperationStmt, MLValue>;
  result_iterator result_begin() { return result_iterator(this, 0); }
  result_iterator result_end() {
    return result_iterator(this, getNumResults());
  }
  llvm::iterator_range<result_iterator> getResults() {
    return {result_begin(), result_end()};
  }

  // Support const result iteration.
  using const_result_iterator =
      ResultIterator<const OperationStmt, const MLValue>;
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

  // Support result type iteration.
  using result_type_iterator =
      ResultTypeIterator<const OperationStmt, const MLValue>;
  result_type_iterator result_type_begin() const {
    return result_type_iterator(this, 0);
  }

  result_type_iterator result_type_end() const {
    return result_type_iterator(this, getNumResults());
  }

  llvm::iterator_range<result_type_iterator> getResultTypes() const {
    return {result_type_begin(), result_type_end()};
  }

  //===--------------------------------------------------------------------===//
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<StmtBlockOperand> getBlockOperands() {
    assert(isTerminator() && "Only terminators have a block operands list");
    return {getTrailingObjects<StmtBlockOperand>(), numSuccs};
  }
  ArrayRef<StmtBlockOperand> getBlockOperands() const {
    return const_cast<OperationStmt *>(this)->getBlockOperands();
  }

  MutableArrayRef<StmtOperand> getSuccessorOperands(unsigned index) {
    assert(isTerminator() && "Only terminators have successors");
    assert(index < getNumSuccessors());
    unsigned succOpIndex = getSuccessorOperandIndex(index);
    auto *operandBegin = getStmtOperands().data() + succOpIndex;
    return {operandBegin, getNumSuccessorOperands(index)};
  }
  ArrayRef<StmtOperand> getSuccessorOperands(unsigned index) const {
    return const_cast<OperationStmt *>(this)->getSuccessorOperands(index);
  }

  unsigned getNumSuccessors() const { return numSuccs; }
  unsigned getNumSuccessorOperands(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors");
    assert(index < getNumSuccessors());
    return getTrailingObjects<unsigned>()[index];
  }

  StmtBlock *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  const StmtBlock *getSuccessor(unsigned index) const {
    return const_cast<OperationStmt *>(this)->getSuccessor(index);
  }
  void setSuccessor(BasicBlock *block, unsigned index);

  /// Get the index of the first operand of the successor at the provided
  /// index.
  unsigned getSuccessorOperandIndex(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());

    // Count the number of operands for each of the successors after, and
    // including, the one at 'index'. This is based upon the assumption that all
    // non successor operands are placed at the beginning of the operand list.
    auto *successorOpCountBegin = getTrailingObjects<unsigned>();
    unsigned postSuccessorOpCount =
        std::accumulate(successorOpCountBegin + index,
                        successorOpCountBegin + getNumSuccessors(), 0);
    return getNumOperands() - postSuccessorOpCount;
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  void destroy();
  using Statement::erase;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::OperationStmt;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Statement;
  }

private:
  const unsigned numOperands, numResults, numSuccs;

  OperationStmt(Location location, OperationName name, unsigned numOperands,
                unsigned numResults, unsigned numSuccessors,
                ArrayRef<NamedAttribute> attributes, MLIRContext *context);
  ~OperationStmt();

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperationStmt, StmtOperand, StmtResult,
                               StmtBlockOperand, unsigned>;
  size_t numTrailingObjects(OverloadToken<StmtOperand>) const {
    return numOperands;
  }
  size_t numTrailingObjects(OverloadToken<StmtResult>) const {
    return numResults;
  }
  size_t numTrailingObjects(OverloadToken<StmtBlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const { return numSuccs; }
};

/// A ForStmtBody represents statements contained within a ForStmt.
class ForStmtBody : public StmtBlock {
public:
  explicit ForStmtBody(ForStmt *stmt)
      : StmtBlock(StmtBlockKind::ForBody), forStmt(stmt) {
    assert(stmt != nullptr && "ForStmtBody must have non-null parent");
  }

  ~ForStmtBody() {}

  /// Methods for support type inquiry through isa, cast, and dyn_cast
  static bool classof(const StmtBlock *block) {
    return block->getStmtBlockKind() == StmtBlockKind::ForBody;
  }

  /// Returns the 'for' statement that contains this body.
  ForStmt *getFor() { return forStmt; }
  const ForStmt *getFor() const { return forStmt; }

private:
  ForStmt *forStmt;
};

/// For statement represents an affine loop nest.
class ForStmt : public Statement, public MLValue {
public:
  static ForStmt *create(Location location, ArrayRef<MLValue *> lbOperands,
                         AffineMap lbMap, ArrayRef<MLValue *> ubOperands,
                         AffineMap ubMap, int64_t step);

  ~ForStmt() {
    // Explicitly erase statements instead of relying of 'StmtBlock' destructor
    // since child statements need to be destroyed before the MLValue that this
    // for stmt represents is destroyed. Affine maps are immortal objects and
    // don't need to be deleted.
    getBody()->clear();
  }

  /// Resolve base class ambiguity.
  using Statement::findFunction;

  /// Operand iterators.
  using operand_iterator = OperandIterator<ForStmt, MLValue>;
  using const_operand_iterator = OperandIterator<const ForStmt, const MLValue>;

  /// Operand iterator range.
  using operand_range = llvm::iterator_range<operand_iterator>;
  using const_operand_range = llvm::iterator_range<const_operand_iterator>;

  /// Get the body of the ForStmt.
  ForStmtBody *getBody() { return &body; }

  /// Get the body of the ForStmt.
  const ForStmtBody *getBody() const { return &body; }

  //===--------------------------------------------------------------------===//
  // Bounds and step
  //===--------------------------------------------------------------------===//

  /// Returns information about the lower bound as a single object.
  const AffineBound getLowerBound() const;

  /// Returns information about the upper bound as a single object.
  const AffineBound getUpperBound() const;

  /// Returns loop step.
  int64_t getStep() const { return step; }

  /// Returns affine map for the lower bound.
  AffineMap getLowerBoundMap() const { return lbMap; }
  /// Returns affine map for the upper bound. The upper bound is exclusive.
  AffineMap getUpperBoundMap() const { return ubMap; }

  /// Set lower bound.
  void setLowerBound(ArrayRef<MLValue *> operands, AffineMap map);
  /// Set upper bound.
  void setUpperBound(ArrayRef<MLValue *> operands, AffineMap map);

  /// Set the lower bound map without changing operands.
  void setLowerBoundMap(AffineMap map);

  /// Set the upper bound map without changing operands.
  void setUpperBoundMap(AffineMap map);

  /// Set loop step.
  void setStep(int64_t step) {
    assert(step > 0 && "step has to be a positive integer constant");
    this->step = step;
  }

  /// Returns true if the lower bound is constant.
  bool hasConstantLowerBound() const;
  /// Returns true if the upper bound is constant.
  bool hasConstantUpperBound() const;
  /// Returns true if both bounds are constant.
  bool hasConstantBounds() const {
    return hasConstantLowerBound() && hasConstantUpperBound();
  }
  /// Returns the value of the constant lower bound.
  /// Fails assertion if the bound is non-constant.
  int64_t getConstantLowerBound() const;
  /// Returns the value of the constant upper bound. The upper bound is
  /// exclusive. Fails assertion if the bound is non-constant.
  int64_t getConstantUpperBound() const;
  /// Sets the lower bound to the given constant value.
  void setConstantLowerBound(int64_t value);
  /// Sets the upper bound to the given constant value.
  void setConstantUpperBound(int64_t value);

  /// Returns true if both the lower and upper bound have the same operand lists
  /// (same operands in the same order).
  bool matchingBoundOperandList() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return operands.size(); }

  MLValue *getOperand(unsigned idx) { return getStmtOperand(idx).get(); }
  const MLValue *getOperand(unsigned idx) const {
    return getStmtOperand(idx).get();
  }
  void setOperand(unsigned idx, MLValue *value) {
    getStmtOperand(idx).set(value);
  }

  operand_iterator operand_begin() { return operand_iterator(this, 0); }
  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }
  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  ArrayRef<StmtOperand> getStmtOperands() const { return operands; }
  MutableArrayRef<StmtOperand> getStmtOperands() { return operands; }
  StmtOperand &getStmtOperand(unsigned idx) { return getStmtOperands()[idx]; }
  const StmtOperand &getStmtOperand(unsigned idx) const {
    return getStmtOperands()[idx];
  }

  // TODO: provide iterators for the lower and upper bound operands
  // if the current access via getLowerBound(), getUpperBound() is too slow.

  /// Returns operands for the lower bound map.
  operand_range getLowerBoundOperands();
  const_operand_range getLowerBoundOperands() const;

  /// Returns operands for the upper bound map.
  operand_range getUpperBoundOperands();
  const_operand_range getUpperBoundOperands() const;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const { return getType().getContext(); }

  using Statement::dump;
  using Statement::print;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::ForStmt;
  }

  // For statement represents implicitly represents induction variable by
  // inheriting from MLValue class. Whenever you need to refer to the loop
  // induction variable, just use the for statement itself.
  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::ForStmt;
  }

private:
  // The StmtBlock for the body.
  ForStmtBody body;

  // Affine map for the lower bound.
  AffineMap lbMap;
  // Affine map for the upper bound. The upper bound is exclusive.
  AffineMap ubMap;
  // Positive constant step. Since index is stored as an int64_t, we restrict
  // step to the set of positive integers that int64_t can represent.
  int64_t step;
  // Operands for the lower and upper bounds, with the former followed by the
  // latter. Dimensional operands are followed by symbolic operands for each
  // bound.
  std::vector<StmtOperand> operands;

  explicit ForStmt(Location location, unsigned numOperands, AffineMap lbMap,
                   AffineMap ubMap, int64_t step);
};

/// AffineBound represents a lower or upper bound in the for statement.
/// This class does not own the underlying operands. Instead, it refers
/// to the operands stored in the ForStmt. Its life span should not exceed
/// that of the for statement it refers to.
class AffineBound {
public:
  const ForStmt *getForStmt() const { return &stmt; }
  AffineMap getMap() const { return map; }

  unsigned getNumOperands() const { return opEnd - opStart; }
  const MLValue *getOperand(unsigned idx) const {
    return stmt.getOperand(opStart + idx);
  }
  const StmtOperand &getStmtOperand(unsigned idx) const {
    return stmt.getStmtOperand(opStart + idx);
  }

  using operand_iterator = ForStmt::operand_iterator;
  using operand_range = ForStmt::operand_range;

  operand_iterator operand_begin() const {
    // These are iterators over MLValue *. Not casting away const'ness would
    // require the caller to use const MLValue *.
    return operand_iterator(const_cast<ForStmt *>(&stmt), opStart);
  }
  operand_iterator operand_end() const {
    return operand_iterator(const_cast<ForStmt *>(&stmt), opEnd);
  }

  /// Returns an iterator on the underlying MLValue's (MLValue *).
  operand_range getOperands() const { return {operand_begin(), operand_end()}; }
  ArrayRef<StmtOperand> getStmtOperands() const {
    auto ops = stmt.getStmtOperands();
    return ArrayRef<StmtOperand>(ops.begin() + opStart, ops.begin() + opEnd);
  }

private:
  // 'for' statement that contains this bound.
  const ForStmt &stmt;
  // Start and end positions of this affine bound operands in the list of
  // the containing 'for' statement operands.
  unsigned opStart, opEnd;
  // Affine map for this bound.
  AffineMap map;

  AffineBound(const ForStmt &stmt, unsigned opStart, unsigned opEnd,
              AffineMap map)
      : stmt(stmt), opStart(opStart), opEnd(opEnd), map(map) {}

  friend class ForStmt;
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
  const IfStmt *getIf() const { return ifStmt; }

  IfStmt *getIf() { return ifStmt; }

private:
  IfStmt *ifStmt;
};

/// If statement restricts execution to a subset of the loop iteration space.
class IfStmt : public Statement {
public:
  static IfStmt *create(Location location, ArrayRef<MLValue *> operands,
                        IntegerSet set);
  ~IfStmt();

  //===--------------------------------------------------------------------===//
  // Then, else, condition.
  //===--------------------------------------------------------------------===//

  IfClause *getThen() { return &thenClause; }
  const IfClause *getThen() const { return &thenClause; }
  IfClause *getElse() { return elseClause; }
  const IfClause *getElse() const { return elseClause; }
  bool hasElse() const { return elseClause != nullptr; }

  IfClause *createElse() {
    assert(elseClause == nullptr && "already has an else clause!");
    return (elseClause = new IfClause(this));
  }

  const AffineCondition getCondition() const;

  IntegerSet getIntegerSet() const { return set; }
  void setIntegerSet(IntegerSet newSet) {
    assert(newSet.getNumOperands() == operands.size());
    set = newSet;
  }

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Operand iterators.
  using operand_iterator = OperandIterator<IfStmt, MLValue>;
  using const_operand_iterator = OperandIterator<const IfStmt, const MLValue>;

  /// Operand iterator range.
  using operand_range = llvm::iterator_range<operand_iterator>;
  using const_operand_range = llvm::iterator_range<const_operand_iterator>;

  unsigned getNumOperands() const { return operands.size(); }

  MLValue *getOperand(unsigned idx) { return getStmtOperand(idx).get(); }
  const MLValue *getOperand(unsigned idx) const {
    return getStmtOperand(idx).get();
  }
  void setOperand(unsigned idx, MLValue *value) {
    getStmtOperand(idx).set(value);
  }

  operand_iterator operand_begin() { return operand_iterator(this, 0); }
  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }
  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  ArrayRef<StmtOperand> getStmtOperands() const { return operands; }
  MutableArrayRef<StmtOperand> getStmtOperands() { return operands; }
  StmtOperand &getStmtOperand(unsigned idx) { return getStmtOperands()[idx]; }
  const StmtOperand &getStmtOperand(unsigned idx) const {
    return getStmtOperands()[idx];
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  MLIRContext *getContext() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::IfStmt;
  }

private:
  // it is always present.
  IfClause thenClause;
  // 'else' clause of the if statement. 'nullptr' if there is no else clause.
  IfClause *elseClause;

  // The integer set capturing the conditional guard.
  IntegerSet set;

  // Condition operands.
  std::vector<StmtOperand> operands;

  explicit IfStmt(Location location, unsigned numOperands, IntegerSet set);
};

/// AffineCondition represents a condition of the 'if' statement.
/// Its life span should not exceed that of the objects it refers to.
/// AffineCondition does not provide its own methods for iterating over
/// the operands since the iterators of the if statement accomplish
/// the same purpose.
///
/// AffineCondition is trivially copyable, so it should be passed by value.
class AffineCondition {
public:
  const IfStmt *getIfStmt() const { return &stmt; }
  IntegerSet getIntegerSet() const { return set; }

private:
  // 'if' statement that contains this affine condition.
  const IfStmt &stmt;
  // Integer set for this affine condition.
  IntegerSet set;

  AffineCondition(const IfStmt &stmt, IntegerSet set) : stmt(stmt), set(set) {}

  friend class IfStmt;
};
} // end namespace mlir

#endif  // MLIR_IR_STATEMENTS_H
