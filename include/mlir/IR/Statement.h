//===- Statement.h - MLIR ML Statement Class --------------------*- C++ -*-===//
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
// This file defines the Statement class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STATEMENT_H
#define MLIR_IR_STATEMENT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class MLFunction;
class StmtBlock;
class ForStmt;
class MLIRContext;
class MLValue;

/// Statement is a basic unit of execution within an ML function.
/// Statements can be nested within for and if statements effectively
/// forming a tree. Child statements are organized into statement blocks
/// represented by a 'StmtBlock' class.
class Statement : public llvm::ilist_node_with_parent<Statement, StmtBlock> {
public:
  enum class Kind {
    Operation,
    For,
    If
  };

  Kind getKind() const { return kind; }
  /// Remove this statement from its block and delete it.
  void eraseFromBlock();

  /// Clone this statement, the cloning is deep.
  Statement *clone() const;

  /// Returns the statement block that contains this statement.
  StmtBlock *getBlock() const { return block; }

  /// Returns the closest surrounding statement that contains this statement
  /// or nullptr if this is a top-level statement.
  Statement *getParentStmt() const;

  /// Returns the function that this statement is part of.
  /// The function is determined by traversing the chain of parent statements.
  /// Returns nullptr if the statement is unlinked.
  MLFunction *findFunction() const;

  /// Returns true if there are no more loops nested under this stmt.
  bool isInnermost() const;

  /// Destroys this statement and its subclass data.
  void destroy();

  void print(raw_ostream &os) const;
  void dump() const;

  /// Replace all uses of 'oldVal' with 'newVal' in 'stmt'.
  void replaceUses(MLValue *oldVal, MLValue *newVal);

protected:
  Statement(Kind kind) : kind(kind) {}
  // Statements are deleted through the destroy() member because this class
  // does not have a virtual destructor.
  ~Statement();

private:
  Kind kind;
  /// The statement block that containts this statement.
  StmtBlock *block = nullptr;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Statement>;
};

inline raw_ostream &operator<<(raw_ostream &os, const Statement &stmt) {
  stmt.print(os);
  return os;
}
} //end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Statement
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::Statement> {
  using Statement = ::mlir::Statement;
  using stmt_iterator = simple_ilist<Statement>::iterator;

  static void deleteNode(Statement *stmt) { stmt->destroy(); }

  void addNodeToList(Statement *stmt);
  void removeNodeFromList(Statement *stmt);
  void transferNodesFromList(ilist_traits<Statement> &otherList,
                             stmt_iterator first, stmt_iterator last);
private:
  mlir::StmtBlock *getContainingBlock();
};

} // end namespace llvm

#endif  // MLIR_IR_STATEMENT_H
