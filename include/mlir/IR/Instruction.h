//===- Instruction.h - MLIR ML Instruction Class --------------------*- C++
//-*-===//
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
// This file defines the Instruction class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTION_H
#define MLIR_IR_INSTRUCTION_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class Block;
class BlockAndValueMapping;
class Location;
class ForInst;
class MLIRContext;

/// Terminator operations can have Block operands to represent successors.
using BlockOperand = IROperandImpl<Block, OperationInst>;

} // namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Instruction
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct ilist_traits<::mlir::Instruction> {
  using Instruction = ::mlir::Instruction;
  using inst_iterator = simple_ilist<Instruction>::iterator;

  static void deleteNode(Instruction *inst);
  void addNodeToList(Instruction *inst);
  void removeNodeFromList(Instruction *inst);
  void transferNodesFromList(ilist_traits<Instruction> &otherList,
                             inst_iterator first, inst_iterator last);

private:
  mlir::Block *getContainingBlock();
};

} // end namespace llvm

namespace mlir {
template <typename ObjectType, typename ElementType> class OperandIterator;

/// Instruction is a basic unit of execution within an ML function.
/// Instructions can be nested within for and if instructions effectively
/// forming a tree. Child instructions are organized into instruction blocks
/// represented by a 'Block' class.
class Instruction : public IROperandOwner,
                    public llvm::ilist_node_with_parent<Instruction, Block> {
public:
  enum class Kind {
    OperationInst = (int)IROperandOwner::Kind::OperationInst,
    For = (int)IROperandOwner::Kind::ForInst,
  };

  Kind getKind() const { return (Kind)IROperandOwner::getKind(); }

  /// Remove this instruction from its parent block and delete it.
  void erase();

  /// Create a deep copy of this instruction, remapping any operands that use
  /// values outside of the instruction using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-instructions to the corresponding instruction that is copied, and adds
  /// those mappings to the map.
  Instruction *clone(BlockAndValueMapping &mapper, MLIRContext *context) const;
  Instruction *clone(MLIRContext *context) const;

  /// Returns the instruction block that contains this instruction.
  const Block *getBlock() const { return block; }
  Block *getBlock() { return block; }

  /// Returns the closest surrounding instruction that contains this instruction
  /// or nullptr if this is a top-level instruction.
  Instruction *getParentInst() const;

  /// Returns the function that this instruction is part of.
  /// The function is determined by traversing the chain of parent instructions.
  /// Returns nullptr if the instruction is unlinked.
  Function *getFunction() const;

  /// Destroys this instruction and its subclass data.
  void destroy();

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Unlink this instruction from its current block and insert it right before
  /// `existingInst` which may be in the same or another block in the same
  /// function.
  void moveBefore(Instruction *existingInst);

  /// Unlink this operation instruction from its current block and insert it
  /// right before `iterator` in the specified block.
  void moveBefore(Block *block, llvm::iplist<Instruction>::iterator iterator);

  /// Given an instruction 'other' that is within the same parent block, return
  /// whether the current instruction is before 'other' in the instruction list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of instructions within the parent block.
  bool isBeforeInBlock(const Instruction *other) const;

  /// Returns whether the instruction is a terminator.
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
  using operand_iterator = OperandIterator<Instruction, Value>;

  operand_iterator operand_begin();

  operand_iterator operand_end();

  /// Returns an iterator on the underlying Values.
  llvm::iterator_range<operand_iterator> getOperands();

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const Instruction, const Value>;

  const_operand_iterator operand_begin() const;

  const_operand_iterator operand_end() const;

  /// Returns a const iterator on the underlying Values.
  llvm::iterator_range<const_operand_iterator> getOperands() const;

  MutableArrayRef<InstOperand> getInstOperands();
  ArrayRef<InstOperand> getInstOperands() const {
    return const_cast<Instruction *>(this)->getInstOperands();
  }

  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return getInstOperands()[idx];
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
    return ptr->getKind() <= IROperandOwner::Kind::INST_LAST;
  }

protected:
  Instruction(Kind kind, Location location)
      : IROperandOwner((IROperandOwner::Kind)kind, location) {}

  // Instructions are deleted through the destroy() member because this class
  // does not have a virtual destructor.
  ~Instruction();

private:
  /// The instruction block that containts this instruction.
  Block *block = nullptr;

  /// Relative order of this instruction in its parent block. Used for
  /// O(1) local dominance checks between instructions.
  mutable unsigned orderIndex = 0;

  // Provide a 'getParent' method for ilist_node_with_parent methods.
  const Block *getParent() const { return getBlock(); }

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Instruction>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Instruction, Block>;
};

inline raw_ostream &operator<<(raw_ostream &os, const Instruction &inst) {
  inst.print(os);
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
inline auto Instruction::operand_begin() -> operand_iterator {
  return operand_iterator(this, 0);
}

inline auto Instruction::operand_end() -> operand_iterator {
  return operand_iterator(this, getNumOperands());
}

inline auto Instruction::getOperands()
    -> llvm::iterator_range<operand_iterator> {
  return {operand_begin(), operand_end()};
}

inline auto Instruction::operand_begin() const -> const_operand_iterator {
  return const_operand_iterator(this, 0);
}

inline auto Instruction::operand_end() const -> const_operand_iterator {
  return const_operand_iterator(this, getNumOperands());
}

inline auto Instruction::getOperands() const
    -> llvm::iterator_range<const_operand_iterator> {
  return {operand_begin(), operand_end()};
}

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTION_H
