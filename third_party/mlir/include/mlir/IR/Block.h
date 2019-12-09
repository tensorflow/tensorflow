//===- Block.h - MLIR Block Class -------------------------------*- C++ -*-===//
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
// This file defines the Block class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCK_H
#define MLIR_IR_BLOCK_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

//===----------------------------------------------------------------------===//
// ilist_traits for Operation
//===----------------------------------------------------------------------===//

namespace llvm {
namespace ilist_detail {
// Explicitly define the node access for the operation list so that we can
// break the dependence on the Operation class in this header. This allows for
// operations to have trailing Regions without a circular include
// dependence.
template <>
struct SpecificNodeAccess<
    typename compute_node_options<::mlir::Operation>::type> : NodeAccess {
protected:
  using OptionsT = typename compute_node_options<mlir::Operation>::type;
  using pointer = typename OptionsT::pointer;
  using const_pointer = typename OptionsT::const_pointer;
  using node_type = ilist_node_impl<OptionsT>;

  static node_type *getNodePtr(pointer N);
  static const node_type *getNodePtr(const_pointer N);

  static pointer getValuePtr(node_type *N);
  static const_pointer getValuePtr(const node_type *N);
};
} // end namespace ilist_detail

template <> struct ilist_traits<::mlir::Operation> {
  using Operation = ::mlir::Operation;
  using op_iterator = simple_ilist<Operation>::iterator;

  static void deleteNode(Operation *op);
  void addNodeToList(Operation *op);
  void removeNodeFromList(Operation *op);
  void transferNodesFromList(ilist_traits<Operation> &otherList,
                             op_iterator first, op_iterator last);

private:
  mlir::Block *getContainingBlock();
};
} // end namespace llvm

namespace mlir {
using BlockOperand = IROperandImpl<Block>;

class PredecessorIterator;
class SuccessorIterator;

/// `Block` represents an ordered list of `Operation`s.
class Block : public IRObjectWithUseList,
              public llvm::ilist_node_with_parent<Block, Region> {
public:
  explicit Block() {}
  ~Block();

  void clear() {
    // Drop all references from within this block.
    dropAllReferences();

    // Clear operations in the reverse order so that uses are destroyed
    // before their defs.
    while (!empty())
      operations.pop_back();
  }

  /// Provide a 'getParent' method for ilist_node_with_parent methods.
  /// We mark it as a const function because ilist_node_with_parent specifically
  /// requires a 'getParent() const' method. Once ilist_node removes this
  /// constraint, we should drop the const to fit the rest of the MLIR const
  /// model.
  Region *getParent() const;

  /// Returns the closest surrounding operation that contains this block.
  Operation *getParentOp();

  /// Return if this block is the entry block in the parent region.
  bool isEntryBlock();

  /// Insert this block (which must not already be in a function) right before
  /// the specified block.
  void insertBefore(Block *block);

  /// Unlink this Block from its parent region and delete it.
  void erase();

  //===--------------------------------------------------------------------===//
  // Block argument management
  //===--------------------------------------------------------------------===//

  // This is the list of arguments to the block.
  using BlockArgListType = ArrayRef<BlockArgument *>;

  BlockArgListType getArguments() { return arguments; }

  using args_iterator = BlockArgListType::iterator;
  using reverse_args_iterator = BlockArgListType::reverse_iterator;
  args_iterator args_begin() { return getArguments().begin(); }
  args_iterator args_end() { return getArguments().end(); }
  reverse_args_iterator args_rbegin() { return getArguments().rbegin(); }
  reverse_args_iterator args_rend() { return getArguments().rend(); }

  bool args_empty() { return arguments.empty(); }

  /// Add one value to the argument list.
  BlockArgument *addArgument(Type type);

  /// Add one argument to the argument list for each type specified in the list.
  llvm::iterator_range<args_iterator> addArguments(ArrayRef<Type> types);

  /// Erase the argument at 'index' and remove it from the argument list. If
  /// 'updatePredTerms' is set to true, this argument is also removed from the
  /// terminators of each predecessor to this block.
  void eraseArgument(unsigned index, bool updatePredTerms = true);

  unsigned getNumArguments() { return arguments.size(); }
  BlockArgument *getArgument(unsigned i) { return arguments[i]; }

  //===--------------------------------------------------------------------===//
  // Operation list management
  //===--------------------------------------------------------------------===//

  /// This is the list of operations in the block.
  using OpListType = llvm::iplist<Operation>;
  OpListType &getOperations() { return operations; }

  // Iteration over the operations in the block.
  using iterator = OpListType::iterator;
  using reverse_iterator = OpListType::reverse_iterator;

  iterator begin() { return operations.begin(); }
  iterator end() { return operations.end(); }
  reverse_iterator rbegin() { return operations.rbegin(); }
  reverse_iterator rend() { return operations.rend(); }

  bool empty() { return operations.empty(); }
  void push_back(Operation *op) { operations.push_back(op); }
  void push_front(Operation *op) { operations.push_front(op); }

  Operation &back() { return operations.back(); }
  Operation &front() { return operations.front(); }

  /// Returns 'op' if 'op' lies in this block, or otherwise finds the
  /// ancestor operation of 'op' that lies in this block. Returns nullptr if
  /// the latter fails.
  /// TODO: This is very specific functionality that should live somewhere else,
  /// probably in Dominance.cpp.
  Operation *findAncestorOpInBlock(Operation &op);

  /// This drops all operand uses from operations within this block, which is
  /// an essential step in breaking cyclic dependences between references when
  /// they are to be deleted.
  void dropAllReferences();

  /// This drops all uses of values defined in this block or in the blocks of
  /// nested regions wherever the uses are located.
  void dropAllDefinedValueUses();

  /// Returns true if the ordering of the child operations is valid, false
  /// otherwise.
  bool isOpOrderValid();

  /// Invalidates the current ordering of operations.
  void invalidateOpOrder();

  /// Verifies the current ordering of child operations matches the
  /// validOpOrder flag. Returns false if the order is valid, true otherwise.
  bool verifyOpOrder();

  /// Recomputes the ordering of child operations within the block.
  void recomputeOpOrder();

private:
  /// A utility iterator that filters out operations that are not 'OpT'.
  template <typename OpT>
  class op_filter_iterator
      : public llvm::filter_iterator<Block::iterator, bool (*)(Operation &)> {
    static bool filter(Operation &op) { return llvm::isa<OpT>(op); }

  public:
    op_filter_iterator(Block::iterator it, Block::iterator end)
        : llvm::filter_iterator<Block::iterator, bool (*)(Operation &)>(
              it, end, &filter) {}

    /// Allow implicit conversion to the underlying block iterator.
    operator Block::iterator() const { return this->wrapped(); }
  };

public:
  /// This class provides iteration over the held operations of a block for a
  /// specific operation type.
  template <typename OpT>
  class op_iterator : public llvm::mapped_iterator<op_filter_iterator<OpT>,
                                                   OpT (*)(Operation &)> {
    static OpT unwrap(Operation &op) { return llvm::cast<OpT>(op); }

  public:
    using reference = OpT;

    /// Initializes the iterator to the specified filter iterator.
    op_iterator(op_filter_iterator<OpT> it)
        : llvm::mapped_iterator<op_filter_iterator<OpT>, OpT (*)(Operation &)>(
              it, &unwrap) {}

    /// Allow implicit conversion to the underlying block iterator.
    operator Block::iterator() const { return this->wrapped(); }
  };

  /// Return an iterator range over the operations within this block that are of
  /// 'OpT'.
  template <typename OpT> llvm::iterator_range<op_iterator<OpT>> getOps() {
    auto endIt = end();
    return {op_filter_iterator<OpT>(begin(), endIt),
            op_filter_iterator<OpT>(endIt, endIt)};
  }
  template <typename OpT> op_iterator<OpT> op_begin() {
    return op_filter_iterator<OpT>(begin(), end());
  }
  template <typename OpT> op_iterator<OpT> op_end() {
    return op_filter_iterator<OpT>(end(), end());
  }

  /// Return an iterator range over the operation within this block excluding
  /// the terminator operation at the end.
  llvm::iterator_range<iterator> without_terminator() {
    if (begin() == end())
      return {begin(), end()};
    auto endIt = --end();
    return {begin(), endIt};
  }

  //===--------------------------------------------------------------------===//
  // Terminator management
  //===--------------------------------------------------------------------===//

  /// Get the terminator operation of this block. This function asserts that
  /// the block has a valid terminator operation.
  Operation *getTerminator();

  //===--------------------------------------------------------------------===//
  // Predecessors and successors.
  //===--------------------------------------------------------------------===//

  // Predecessor iteration.
  using pred_iterator = PredecessorIterator;
  pred_iterator pred_begin();
  pred_iterator pred_end();
  llvm::iterator_range<pred_iterator> getPredecessors();

  /// Return true if this block has no predecessors.
  bool hasNoPredecessors();

  /// If this block has exactly one predecessor, return it.  Otherwise, return
  /// null.
  ///
  /// Note that if a block has duplicate predecessors from a single block (e.g.
  /// if you have a conditional branch with the same block as the true/false
  /// destinations) is not considered to be a single predecessor.
  Block *getSinglePredecessor();

  // Indexed successor access.
  unsigned getNumSuccessors();
  Block *getSuccessor(unsigned i);

  // Successor iteration.
  using succ_iterator = SuccessorIterator;
  succ_iterator succ_begin();
  succ_iterator succ_end();
  llvm::iterator_range<succ_iterator> getSuccessors();

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operations in this block in postorder, calling the callback for
  /// each operation.
  /// See Operation::walk for more details.
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  RetT walk(FnT &&callback) {
    return walk(begin(), end(), std::forward<FnT>(callback));
  }

  /// Walk the operations in the specified [begin, end) range of this block in
  /// postorder, calling the callback for each operation. This method is invoked
  /// for void return callbacks.
  /// See Operation::walk for more details.
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<std::is_same<RetT, void>::value, RetT>::type
  walk(Block::iterator begin, Block::iterator end, FnT &&callback) {
    for (auto &op : llvm::make_early_inc_range(llvm::make_range(begin, end)))
      detail::walkOperations(&op, callback);
  }

  /// Walk the operations in the specified [begin, end) range of this block in
  /// postorder, calling the callback for each operation. This method is invoked
  /// for interruptible callbacks.
  /// See Operation::walk for more details.
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<std::is_same<RetT, WalkResult>::value, RetT>::type
  walk(Block::iterator begin, Block::iterator end, FnT &&callback) {
    for (auto &op : llvm::make_early_inc_range(llvm::make_range(begin, end)))
      if (detail::walkOperations(&op, callback).wasInterrupted())
        return WalkResult::interrupt();
    return WalkResult::advance();
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Split the block into two blocks before the specified operation or
  /// iterator.
  ///
  /// Note that all operations BEFORE the specified iterator stay as part of
  /// the original basic block, and the rest of the operations in the original
  /// block are moved to the new block, including the old terminator.  The
  /// original block is left without a terminator.
  ///
  /// The newly formed Block is returned, and the specified iterator is
  /// invalidated.
  Block *splitBlock(iterator splitBefore);
  Block *splitBlock(Operation *splitBeforeOp) {
    return splitBlock(iterator(splitBeforeOp));
  }

  /// Returns pointer to member of operation list.
  static OpListType Block::*getSublistAccess(Operation *) {
    return &Block::operations;
  }

  void print(raw_ostream &os);
  void dump();

  /// Print out the name of the block without printing its body.
  /// NOTE: The printType argument is ignored.  We keep it for compatibility
  /// with LLVM dominator machinery that expects it to exist.
  void printAsOperand(raw_ostream &os, bool printType = true);

private:
  /// Pair of the parent object that owns this block and a bit that signifies if
  /// the operations within this block have a valid ordering.
  llvm::PointerIntPair<Region *, /*IntBits=*/1, bool> parentValidOpOrderPair;

  /// This is the list of operations in the block.
  OpListType operations;

  /// This is the list of arguments to the block.
  std::vector<BlockArgument *> arguments;

  Block(Block &) = delete;
  void operator=(Block &) = delete;

  friend struct llvm::ilist_traits<Block>;
};

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Block
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::Block> : public ilist_alloc_traits<::mlir::Block> {
  using Block = ::mlir::Block;
  using block_iterator = simple_ilist<::mlir::Block>::iterator;

  void addNodeToList(Block *block);
  void removeNodeFromList(Block *block);
  void transferNodesFromList(ilist_traits<Block> &otherList,
                             block_iterator first, block_iterator last);

private:
  mlir::Region *getParentRegion();
};
} // end namespace llvm

namespace mlir {
//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

/// Implement a predecessor iterator for blocks. This works by walking the use
/// lists of the blocks. The entries on this list are the BlockOperands that
/// are embedded into terminator operations. From the operand, we can get the
/// terminator that contains it, and its parent block is the predecessor.
class PredecessorIterator final
    : public llvm::mapped_iterator<ValueUseIterator<BlockOperand>,
                                   Block *(*)(BlockOperand &)> {
  static Block *unwrap(BlockOperand &value);

public:
  using reference = Block *;

  /// Initializes the operand type iterator to the specified operand iterator.
  PredecessorIterator(ValueUseIterator<BlockOperand> it)
      : llvm::mapped_iterator<ValueUseIterator<BlockOperand>,
                              Block *(*)(BlockOperand &)>(it, &unwrap) {}
  explicit PredecessorIterator(BlockOperand *operand)
      : PredecessorIterator(ValueUseIterator<BlockOperand>(operand)) {}

  /// Get the successor number in the predecessor terminator.
  unsigned getSuccessorIndex() const;
};

inline auto Block::pred_begin() -> pred_iterator {
  return pred_iterator((BlockOperand *)getFirstUse());
}

inline auto Block::pred_end() -> pred_iterator {
  return pred_iterator(nullptr);
}

inline auto Block::getPredecessors() -> llvm::iterator_range<pred_iterator> {
  return {pred_begin(), pred_end()};
}

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//

/// This template implements the successor iterators for Block.
class SuccessorIterator final
    : public indexed_accessor_iterator<SuccessorIterator, Block *, Block *,
                                       Block *, Block *> {
public:
  /// Initializes the result iterator to the specified index.
  SuccessorIterator(Block *object, unsigned index)
      : indexed_accessor_iterator<SuccessorIterator, Block *, Block *, Block *,
                                  Block *>(object, index) {}

  SuccessorIterator(const SuccessorIterator &other)
      : SuccessorIterator(other.base, other.index) {}

  Block *operator*() const { return this->base->getSuccessor(this->index); }

  /// Get the successor number in the terminator.
  unsigned getSuccessorIndex() const { return this->index; }
};

inline auto Block::succ_begin() -> succ_iterator {
  return succ_iterator(this, 0);
}

inline auto Block::succ_end() -> succ_iterator {
  return succ_iterator(this, getNumSuccessors());
}

inline auto Block::getSuccessors() -> llvm::iterator_range<succ_iterator> {
  return {succ_begin(), succ_end()};
}

} // end namespace mlir

#endif // MLIR_IR_BLOCK_H
