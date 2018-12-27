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

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class Location;
using MLFunction = Function;
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
template <typename ObjectType, typename ElementType> class OperandIterator;

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
      DenseMap<const Value *, Value *, llvm::DenseMapInfo<const Value *>,
               llvm::detail::DenseMapPair<const Value *, Value *>>;

  /// Create a deep copy of this statement, remapping any operands that use
  /// values outside of the statement using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-statements to the corresponding statement that is copied, and adds
  /// those mappings to the map.
  Statement *clone(OperandMapTy &operandMap, MLIRContext *context) const;
  Statement *clone(MLIRContext *context) const;

  /// Returns the statement block that contains this statement.
  StmtBlock *getBlock() const { return block; }

  /// Returns the closest surrounding statement that contains this statement
  /// or nullptr if this is a top-level statement.
  Statement *getParentStmt() const;

  /// Returns the function that this statement is part of.
  /// The function is determined by traversing the chain of parent statements.
  /// Returns nullptr if the statement is unlinked.
  MLFunction *getFunction() const;

  /// Destroys this statement and its subclass data.
  void destroy();

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Unlink this statement from its current block and insert it right before
  /// `existingStmt` which may be in the same or another block in the same
  /// function.
  void moveBefore(Statement *existingStmt);

  /// Unlink this operation instruction from its current basic block and insert
  /// it right before `iterator` in the specified basic block.
  void moveBefore(StmtBlock *block, llvm::iplist<Statement>::iterator iterator);

  // Returns whether the Statement is a terminator.
  bool isTerminator() const;

  void print(raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const;

  Value *getOperand(unsigned idx);
  const Value *getOperand(unsigned idx) const;
  void setOperand(unsigned idx, Value *value);

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<Statement, Value>;

  operand_iterator operand_begin();

  operand_iterator operand_end();

  /// Returns an iterator on the underlying Values.
  llvm::iterator_range<operand_iterator> getOperands();

  // Support const operand iteration.
  using const_operand_iterator = OperandIterator<const Statement, const Value>;

  const_operand_iterator operand_begin() const;

  const_operand_iterator operand_end() const;

  /// Returns a const iterator on the underlying Values.
  llvm::iterator_range<const_operand_iterator> getOperands() const;

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

/// This is a helper template used to implement an iterator that contains a
/// pointer to some object and an index into it.  The iterator moves the
/// index but keeps the object constant.
template <typename ConcreteType, typename ObjectType, typename ElementType>
class IndexedAccessorIterator
    : public llvm::iterator_facade_base<
          ConcreteType, std::random_access_iterator_tag, ElementType *,
          std::ptrdiff_t, ElementType *, ElementType *> {
public:
  ptrdiff_t operator-(const IndexedAccessorIterator &rhs) const {
    assert(object == rhs.object && "incompatible iterators");
    return index - rhs.index;
  }
  bool operator==(const IndexedAccessorIterator &rhs) const {
    return object == rhs.object && index == rhs.index;
  }
  bool operator<(const IndexedAccessorIterator &rhs) const {
    assert(object == rhs.object && "incompatible iterators");
    return index < rhs.index;
  }

  ConcreteType &operator+=(ptrdiff_t offset) {
    this->index += offset;
    return static_cast<ConcreteType &>(*this);
  }
  ConcreteType &operator-=(ptrdiff_t offset) {
    this->index -= offset;
    return static_cast<ConcreteType &>(*this);
  }

protected:
  IndexedAccessorIterator(ObjectType *object, unsigned index)
      : object(object), index(index) {}
  ObjectType *object;
  unsigned index;
};

/// This template implements the const/non-const operand iterators for the
/// Instruction class in terms of getOperand(idx).
template <typename ObjectType, typename ElementType>
class OperandIterator final
    : public IndexedAccessorIterator<OperandIterator<ObjectType, ElementType>,
                                     ObjectType, ElementType> {
public:
  /// Initializes the operand iterator to the specified operand index.
  OperandIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<OperandIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator OperandIterator<const ObjectType, const ElementType>() const {
    return OperandIterator<const ObjectType, const ElementType>(this->object,
                                                                this->index);
  }

  ElementType *operator*() const {
    return this->object->getOperand(this->index);
  }
};

// Implement the inline operand iterator methods.
inline auto Statement::operand_begin() -> operand_iterator {
  return operand_iterator(this, 0);
}

inline auto Statement::operand_end() -> operand_iterator {
  return operand_iterator(this, getNumOperands());
}

inline auto Statement::getOperands() -> llvm::iterator_range<operand_iterator> {
  return {operand_begin(), operand_end()};
}

inline auto Statement::operand_begin() const -> const_operand_iterator {
  return const_operand_iterator(this, 0);
}

inline auto Statement::operand_end() const -> const_operand_iterator {
  return const_operand_iterator(this, getNumOperands());
}

inline auto Statement::getOperands() const
    -> llvm::iterator_range<const_operand_iterator> {
  return {operand_begin(), operand_end()};
}

} // end namespace mlir

#endif  // MLIR_IR_STATEMENT_H
