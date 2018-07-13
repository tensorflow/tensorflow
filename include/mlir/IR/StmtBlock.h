//===- StmtBlock.h ----------------------------------------------*- C++ -*-===//
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
// This file defines StmtBlock and *Stmt classes that extend Statement.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STMTBLOCK_H
#define MLIR_IR_STMTBLOCK_H

#include "mlir/Support/LLVM.h"
#include "mlir/IR/Statement.h"

namespace mlir {
  class MLFunction;
  class IfStmt;

/// Statement block represents an ordered list of statements.
class StmtBlock {
public:
  /// Returns the closest surrounding statement that contains this block or
  /// nullptr if this is a top-level statement block.
  Statement *getParent() const { return parent; }

  /// Returns the function that this statement block is part of.
  MLFunction *getFunction() const;

  //===--------------------------------------------------------------------===//
  // Statement list management
  //===--------------------------------------------------------------------===//

  /// This is the list of statements in the block.
  typedef llvm::iplist<Statement> StmtListType;
  StmtListType       &getStatements() { return statements; }
  const StmtListType &getStatements() const { return statements; }

  // Iteration over the statements in the block.
  using iterator = StmtListType::iterator;
  using const_iterator = StmtListType::const_iterator;
  using reverse_iterator = StmtListType::reverse_iterator;
  using const_reverse_iterator = StmtListType::const_reverse_iterator;

  iterator               begin() { return statements.begin(); }
  iterator               end() { return statements.end(); }
  const_iterator         begin() const { return statements.begin(); }
  const_iterator         end() const { return statements.end(); }
  reverse_iterator       rbegin() { return statements.rbegin(); }
  reverse_iterator       rend() { return statements.rend(); }
  const_reverse_iterator rbegin() const { return statements.rbegin(); }
  const_reverse_iterator rend() const { return statements.rend(); }

  bool empty() const { return statements.empty(); }
  void push_back(Statement *stmt) { statements.push_back(stmt); }
  void push_front(Statement *stmt) { statements.push_front(stmt); }

  Statement       &back() { return statements.back(); }
  const Statement &back() const {
    return const_cast<StmtBlock *>(this)->back();
  }
  Statement       &front() { return statements.front(); }
  const Statement &front() const {
    return const_cast<StmtBlock*>(this)->front();
  }

  void print(raw_ostream &os) const;
  void dump() const;

  /// getSublistAccess() - Returns pointer to member of statement list
  static StmtListType StmtBlock::*getSublistAccess(Statement*) {
    return &StmtBlock::statements;
  }

protected:
  Statement *parent;

  StmtBlock(Statement *parent=nullptr) : parent(parent) {}
private:
  /// This is the list of statements in the block.
  StmtListType statements;

  StmtBlock(const StmtBlock&) = delete;
  void operator=(const StmtBlock&) = delete;

};

} //end namespace mlir
#endif  // MLIR_IR_STMTBLOCK_H
