//===- BasicBlock.h - MLIR BasicBlock Class ---------------------*- C++ -*-===//
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

#ifndef MLIR_IR_BASICBLOCK_H
#define MLIR_IR_BASICBLOCK_H

#include "mlir/IR/Instructions.h"

namespace mlir {
class BBArgument;
class CFGFunction;
template <typename BlockType> class PredecessorIterator;
template <typename BlockType> class SuccessorIterator;

/// Each basic block in a CFG function contains a list of basic block arguments,
/// normal instructions, and a terminator instruction.
///
/// Basic blocks form a graph (the CFG) which can be traversed through
/// predecessor and successor edges.
class BasicBlock
    : public IRObjectWithUseList,
      public llvm::ilist_node_with_parent<BasicBlock, CFGFunction> {
public:
  explicit BasicBlock();
  ~BasicBlock();

  /// Return the function that a BasicBlock is part of.
  CFGFunction *getFunction() { return function; }
  const CFGFunction *getFunction() const { return function; }

  /// Return the function that a BasicBlock is part of.
  const CFGFunction *getParent() const { return function; }
  CFGFunction *getParent() { return function; }

  //===--------------------------------------------------------------------===//
  // Block arguments management
  //===--------------------------------------------------------------------===//

  // This is the list of arguments to the block.
  using BBArgListType = ArrayRef<BBArgument *>;
  BBArgListType getArguments() const { return arguments; }

  using args_iterator = BBArgListType::iterator;
  using reverse_args_iterator = BBArgListType::reverse_iterator;
  args_iterator args_begin() const { return getArguments().begin(); }
  args_iterator args_end() const { return getArguments().end(); }
  reverse_args_iterator args_rbegin() const { return getArguments().rbegin(); }
  reverse_args_iterator args_rend() const { return getArguments().rend(); }

  bool args_empty() const { return arguments.empty(); }

  /// Add one value to the argument list.
  BBArgument *addArgument(Type type);

  /// Add one argument to the argument list for each type specified in the list.
  llvm::iterator_range<args_iterator> addArguments(ArrayRef<Type> types);

  /// Erase the argument at 'index' and remove it from the argument list.
  void eraseArgument(unsigned index);

  unsigned getNumArguments() const { return arguments.size(); }
  BBArgument *getArgument(unsigned i) { return arguments[i]; }
  const BBArgument *getArgument(unsigned i) const { return arguments[i]; }

  //===--------------------------------------------------------------------===//
  // Operation list management
  //===--------------------------------------------------------------------===//

  /// This is the list of operations in the block.
  using OperationListType = llvm::iplist<Instruction>;
  OperationListType &getOperations() { return operations; }
  const OperationListType &getOperations() const { return operations; }

  // Iteration over the operations in the block.
  using iterator = OperationListType::iterator;
  using const_iterator = OperationListType::const_iterator;
  using reverse_iterator = OperationListType::reverse_iterator;
  using const_reverse_iterator = OperationListType::const_reverse_iterator;

  iterator begin() { return operations.begin(); }
  iterator end() { return operations.end(); }
  const_iterator begin() const { return operations.begin(); }
  const_iterator end() const { return operations.end(); }
  reverse_iterator rbegin() { return operations.rbegin(); }
  reverse_iterator rend() { return operations.rend(); }
  const_reverse_iterator rbegin() const { return operations.rbegin(); }
  const_reverse_iterator rend() const { return operations.rend(); }

  bool empty() const { return operations.empty(); }
  void push_back(Instruction *inst) { operations.push_back(inst); }
  void push_front(Instruction *inst) { operations.push_front(inst); }

  Instruction &back() { return operations.back(); }
  const Instruction &back() const {
    return const_cast<BasicBlock *>(this)->back();
  }

  Instruction &front() { return operations.front(); }
  const Instruction &front() const {
    return const_cast<BasicBlock*>(this)->front();
  }

  //===--------------------------------------------------------------------===//
  // Terminator management
  //===--------------------------------------------------------------------===//

  /// Get the terminator instruction of this block, or null if the block is
  /// malformed.
  Instruction *getTerminator() const;

  //===--------------------------------------------------------------------===//
  // Predecessors and successors.
  //===--------------------------------------------------------------------===//

  // Predecessor iteration.
  using const_pred_iterator = PredecessorIterator<const BasicBlock>;
  const_pred_iterator pred_begin() const;
  const_pred_iterator pred_end() const;
  llvm::iterator_range<const_pred_iterator> getPredecessors() const;

  using pred_iterator = PredecessorIterator<BasicBlock>;
  pred_iterator pred_begin();
  pred_iterator pred_end();
  llvm::iterator_range<pred_iterator> getPredecessors();

  /// Return true if this block has no predecessors.
  bool hasNoPredecessors() const;

  /// If this basic block has exactly one predecessor, return it.  Otherwise,
  /// return null.
  ///
  /// Note that if a block has duplicate predecessors from a single block (e.g.
  /// if you have a conditional branch with the same block as the true/false
  /// destinations) is not considered to be a single predecessor.
  BasicBlock *getSinglePredecessor();

  const BasicBlock *getSinglePredecessor() const {
    return const_cast<BasicBlock *>(this)->getSinglePredecessor();
  }

  // Indexed successor access.
  unsigned getNumSuccessors() const {
    return getTerminator()->getNumSuccessors();
  }
  const BasicBlock *getSuccessor(unsigned i) const {
    return const_cast<BasicBlock *>(this)->getSuccessor(i);
  }
  BasicBlock *getSuccessor(unsigned i) {
    return getTerminator()->getSuccessor(i);
  }

  // Successor iteration.
  using const_succ_iterator = SuccessorIterator<const BasicBlock>;
  const_succ_iterator succ_begin() const;
  const_succ_iterator succ_end() const;
  llvm::iterator_range<const_succ_iterator> getSuccessors() const;

  using succ_iterator = SuccessorIterator<BasicBlock>;
  succ_iterator succ_begin();
  succ_iterator succ_end();
  llvm::iterator_range<succ_iterator> getSuccessors();

  //===--------------------------------------------------------------------===//
  // Manipulators
  //===--------------------------------------------------------------------===//

  /// Unlink this BasicBlock from its CFGFunction and delete it.
  void eraseFromFunction();

  /// Split the basic block into two basic blocks before the specified
  /// instruction or iterator.
  ///
  /// Note that all instructions BEFORE the specified iterator stay as part of
  /// the original basic block, an unconditional branch is added to the original
  /// block (going to the new block), and the rest of the instructions in the
  /// original block are moved to the new BB, including the old terminator.  The
  /// newly formed BasicBlock is returned.
  ///
  /// This function invalidates the specified iterator.
  BasicBlock *splitBasicBlock(iterator splitBefore);
  BasicBlock *splitBasicBlock(Instruction *splitBeforeInst) {
    return splitBasicBlock(iterator(splitBeforeInst));
  }

  void print(raw_ostream &os) const;
  void dump() const;

  /// Print out the name of the basic block without printing its body.
  /// NOTE: The printType argument is ignored.  We keep it for compatibility
  /// with LLVM dominator machinery that expects it to exist.
  void printAsOperand(raw_ostream &os, bool printType = true);

  /// getSublistAccess() - Returns pointer to member of operation list
  static OperationListType BasicBlock::*getSublistAccess(Instruction *) {
    return &BasicBlock::operations;
  }

private:
  CFGFunction *function = nullptr;

  /// This is the list of operations in the block.
  OperationListType operations;

  /// This is the list of arguments to the block.
  std::vector<BBArgument *> arguments;

  BasicBlock(const BasicBlock&) = delete;
  void operator=(const BasicBlock&) = delete;

  friend struct llvm::ilist_traits<BasicBlock>;
};

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

/// Implement a predecessor iterator as a forward iterator.  This works by
/// walking the use lists of basic blocks.  The entries on this list are the
/// BasicBlockOperands that are embedded into terminator instructions.  From the
/// operand, we can get the terminator that contains it, and it's parent block
/// is the predecessor.
template <typename BlockType>
class PredecessorIterator
    : public llvm::iterator_facade_base<PredecessorIterator<BlockType>,
                                        std::forward_iterator_tag,
                                        BlockType *> {
public:
  PredecessorIterator(BasicBlockOperand *firstOperand)
      : bbUseIterator(firstOperand) {}

  PredecessorIterator &operator=(const PredecessorIterator &rhs) {
    bbUseIterator = rhs.bbUseIterator;
  }

  bool operator==(const PredecessorIterator &rhs) const {
    return bbUseIterator == rhs.bbUseIterator;
  }

  BlockType *operator*() const {
    // The use iterator points to an operand of a terminator.  The predecessor
    // we return is the basic block that that terminator is embedded into.
    return bbUseIterator.getUser()->getBlock();
  }

  PredecessorIterator &operator++() {
    ++bbUseIterator;
    return *this;
  }

  /// Get the successor number in the predecessor terminator.
  unsigned getSuccessorIndex() const {
    return bbUseIterator->getOperandNumber();
  }

private:
  using BBUseIterator = SSAValueUseIterator<BasicBlockOperand, Instruction>;
  BBUseIterator bbUseIterator;
};

inline auto BasicBlock::pred_begin() const -> const_pred_iterator {
  return const_pred_iterator((BasicBlockOperand *)getFirstUse());
}

inline auto BasicBlock::pred_end() const -> const_pred_iterator {
  return const_pred_iterator(nullptr);
}

inline auto BasicBlock::getPredecessors() const
    -> llvm::iterator_range<const_pred_iterator> {
  return {pred_begin(), pred_end()};
}

inline auto BasicBlock::pred_begin() -> pred_iterator {
  return pred_iterator((BasicBlockOperand *)getFirstUse());
}

inline auto BasicBlock::pred_end() -> pred_iterator {
  return pred_iterator(nullptr);
}

inline auto BasicBlock::getPredecessors()
    -> llvm::iterator_range<pred_iterator> {
  return {pred_begin(), pred_end()};
}

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//

/// This template implments the successor iterators for basic block.
template <typename BlockType>
class SuccessorIterator final
    : public IndexedAccessorIterator<SuccessorIterator<BlockType>, BlockType,
                                     BlockType> {
public:
  /// Initializes the result iterator to the specified index.
  SuccessorIterator(BlockType *object, unsigned index)
      : IndexedAccessorIterator<SuccessorIterator<BlockType>, BlockType,
                                BlockType>(object, index) {}

  SuccessorIterator(const SuccessorIterator &other)
      : SuccessorIterator(other.object, other.index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator SuccessorIterator<const BlockType>() const {
    return SuccessorIterator<const BlockType>(this->object, this->index);
  }

  BlockType *operator*() const {
    return this->object->getSuccessor(this->index);
  }

  /// Get the successor number in the terminator.
  unsigned getSuccessorIndex() const { return this->index; }
};

inline auto BasicBlock::succ_begin() const -> const_succ_iterator {
  return const_succ_iterator(this, 0);
}

inline auto BasicBlock::succ_end() const -> const_succ_iterator {
  return const_succ_iterator(this, getNumSuccessors());
}

inline auto BasicBlock::getSuccessors() const
    -> llvm::iterator_range<const_succ_iterator> {
  return {succ_begin(), succ_end()};
}

inline auto BasicBlock::succ_begin() -> succ_iterator {
  return succ_iterator(this, 0);
}

inline auto BasicBlock::succ_end() -> succ_iterator {
  return succ_iterator(this, getNumSuccessors());
}

inline auto BasicBlock::getSuccessors() -> llvm::iterator_range<succ_iterator> {
  return {succ_begin(), succ_end()};
}

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for BasicBlock
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::BasicBlock>
  : public ilist_alloc_traits<::mlir::BasicBlock> {
  using BasicBlock = ::mlir::BasicBlock;
  using block_iterator = simple_ilist<BasicBlock>::iterator;

  void addNodeToList(BasicBlock *block);
  void removeNodeFromList(BasicBlock *block);
  void transferNodesFromList(ilist_traits<BasicBlock> &otherList,
                             block_iterator first, block_iterator last);
private:
  mlir::CFGFunction *getContainingFunction();
};
} // end namespace llvm


#endif  // MLIR_IR_BASICBLOCK_H
