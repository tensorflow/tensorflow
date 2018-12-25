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

#include "mlir/IR/MLValue.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class Location;
class MLFunction;
class StmtBlock;
class ForStmt;
class MLIRContext;

/// The operand of a Terminator contains a StmtBlock.
using StmtBlockOperand = IROperandImpl<StmtBlock, OperationStmt>;

} // namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Statement
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct ilist_traits<::mlir::Statement> {
  using Statement = ::mlir::Statement;
  using stmt_iterator = simple_ilist<Statement>::iterator;

  static void deleteNode(Statement *stmt);
  void addNodeToList(Statement *stmt);
  void removeNodeFromList(Statement *stmt);
  void transferNodesFromList(ilist_traits<Statement> &otherList,
                             stmt_iterator first, stmt_iterator last);

private:
  mlir::StmtBlock *getContainingBlock();
};

} // end namespace llvm

namespace mlir {

/// Statement is a basic unit of execution within an ML function.
/// Statements can be nested within for and if statements effectively
/// forming a tree. Child statements are organized into statement blocks
/// represented by a 'StmtBlock' class.
class Statement : public IROperandOwner,
                  public llvm::ilist_node_with_parent<Statement, StmtBlock> {
public:
  enum class Kind {
    Operation = (int)IROperandOwner::Kind::OperationStmt,
    For = (int)IROperandOwner::Kind::ForStmt,
    If = (int)IROperandOwner::Kind::IfStmt,
  };

  Kind getKind() const { return (Kind)IROperandOwner::getKind(); }

  /// Remove this statement from its parent block and delete it.
  void erase();

  // This is a verbose type used by the clone method below.
  using OperandMapTy =
      DenseMap<const MLValue *, MLValue *, llvm::DenseMapInfo<const MLValue *>,
               llvm::detail::DenseMapPair<const MLValue *, MLValue *>>;

  /// Create a deep copy of this statement, remapping any operands that use
  /// values outside of the statement using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-statements to the corresponding statement that is copied, and adds
  /// those mappings to the map.
  Statement *clone(OperandMapTy &operandMap, MLIRContext *context) const;

  /// Returns the statement block that contains this statement.
  StmtBlock *getBlock() const { return block; }

  /// Returns the closest surrounding statement that contains this statement
  /// or nullptr if this is a top-level statement.
  Statement *getParentStmt() const;

  /// Returns the function that this statement is part of.
  /// The function is determined by traversing the chain of parent statements.
  /// Returns nullptr if the statement is unlinked.
  MLFunction *findFunction() const;

  /// Destroys this statement and its subclass data.
  void destroy();

  /// Unlink this statement from its current block and insert it right before
  /// `existingStmt` which may be in the same or another block in the same
  /// function.
  void moveBefore(Statement *existingStmt);

  /// Unlink this operation instruction from its current basic block and insert
  /// it right before `iterator` in the specified basic block.
  void moveBefore(StmtBlock *block, llvm::iplist<Statement>::iterator iterator);

  void print(raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const;

  MLValue *getOperand(unsigned idx);
  const MLValue *getOperand(unsigned idx) const;
  void setOperand(unsigned idx, MLValue *value);

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<Statement, MLValue>;

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
      OperandIterator<const Statement, const MLValue>;

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

  MutableArrayRef<StmtOperand> getStmtOperands();
  ArrayRef<StmtOperand> getStmtOperands() const {
    return const_cast<Statement *>(this)->getStmtOperands();
  }

  StmtOperand &getStmtOperand(unsigned idx) { return getStmtOperands()[idx]; }
  const StmtOperand &getStmtOperand(unsigned idx) const {
    return getStmtOperands()[idx];
  }

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() <= IROperandOwner::Kind::STMT_LAST;
  }

protected:
  Statement(Kind kind, Location location)
      : IROperandOwner((IROperandOwner::Kind)kind, location) {}

  // Statements are deleted through the destroy() member because this class
  // does not have a virtual destructor.
  ~Statement();

private:
  /// The statement block that containts this statement.
  StmtBlock *block = nullptr;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Statement>;
};

inline raw_ostream &operator<<(raw_ostream &os, const Statement &stmt) {
  stmt.print(os);
  return os;
}
} // end namespace mlir

#endif  // MLIR_IR_STATEMENT_H
