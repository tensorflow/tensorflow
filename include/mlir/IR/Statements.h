//===- Statements.h - MLIR ML Statement Classes ------------*- C++ -*-===//
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
// This file defines the classes for MLFunction statements.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STATEMENTS_H
#define MLIR_IR_STATEMENTS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include <vector>

namespace mlir {
  class MLFunction;
  class NodeStmt;
  class ElseClause;

  typedef PointerUnion<MLFunction *, NodeStmt *> ParentType;

/// Statement is a basic unit of execution within an ML function.
/// Statements can be nested within each other, effectively forming a tree.
class Statement {
public:
  enum class Kind {
    Operation,
    For,
    If,
    Else
  };

  Kind getKind() const { return kind; }

  /// Returns the parent of this statement. The parent of a nested statement
  /// is the closest surrounding for or if statement. The parent of
  /// a top-level statement is the function that contains the statement.
  ParentType getParent() const { return parent; }

  /// Returns the function that this statement is part of.
  MLFunction *getFunction() const;

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  Statement(Kind kind, ParentType parent) : kind(kind), parent(parent) {}
private:
  Kind kind;
  ParentType parent;
};

/// Node statement represents a statement that may contain other statements.
class NodeStmt : public Statement {
public:
  // FIXME: wrong representation and API, leaks memory etc
  std::vector<Statement*> children;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() != Kind::Operation;
  }

protected:
  NodeStmt(Kind kind, ParentType parent) : Statement(kind, parent) {}
};

/// For statement represents an affine loop nest.
class ForStmt : public NodeStmt {
public:
  explicit ForStmt(ParentType parent) : NodeStmt(Kind::For, parent) {}

  // TODO: represent loop variable, bounds and step

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::For;
  }
};

/// If statement restricts execution to a subset of the loop iteration space.
class IfStmt : public NodeStmt {
public:
  explicit IfStmt(ParentType parent) : NodeStmt(Kind::If, parent) {}

  // TODO: Represent condition

  // FIXME: most likely wrong representation since it's wrong everywhere else
  std::vector<ElseClause *> elseClauses;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::If;
  }
};

/// Else clause reprsents else or else-if clause of an if statement
class ElseClause : public NodeStmt {
public:
  explicit ElseClause(IfStmt *ifStmt, int clauseNum);

  // TODO: Represent optional condition

  // Returns ordinal number of this clause in the list of clauses.
  int getClauseNumber() const { return clauseNum;}

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Statement *stmt) {
    return stmt->getKind() == Kind::Else;
  }
private:
  int clauseNum;
};

} //end namespace mlir
#endif  // MLIR_IR_STATEMENTS_H
