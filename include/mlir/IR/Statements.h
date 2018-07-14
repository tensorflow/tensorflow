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

#include "mlir/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Statement.h"
#include "mlir/IR/StmtBlock.h"

namespace mlir {

/// Operation statements represent operations inside ML functions.
class OperationStmt : public Operation, public Statement {
public:
  explicit OperationStmt(Identifier name, ArrayRef<NamedAttribute> attrs,
                         MLIRContext *context)
      : Operation(name, attrs, context), Statement(Kind::Operation) {}
  ~OperationStmt() {}

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::Operation;
  }
};

/// For statement represents an affine loop nest.
class ForStmt : public Statement, public StmtBlock {
public:
  explicit ForStmt() : Statement(Kind::For), StmtBlock(StmtBlockKind::For) {}
  //TODO: delete nested statements or assert that they are gone.
  ~ForStmt() {}

  // TODO: represent loop variable, bounds and step

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::For;
  }

  static bool classof(const StmtBlock *block) {
    return block->getStmtBlockKind() == StmtBlockKind::For;
  }
};

/// If clause represents statements contained within then or else clause
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

  //TODO: delete nested statements or assert that they are gone.
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
