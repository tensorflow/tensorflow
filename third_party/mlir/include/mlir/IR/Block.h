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

#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
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
  pred_iterator pred_begin() {
    return pred_iterator((BlockOperand *)getFirstUse());
  }
  pred_iterator pred_end() { return pred_iterator(nullptr); }
  llvm::iterator_range<pred_iterator> getPredecessors() {
    return {pred_begin(), pred_end()};
  }

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
  using succ_iterator = SuccessorRange::iterator;
  succ_iterator succ_begin() { return getSuccessors().begin(); }
  succ_iterator succ_end() { return getSuccessors().end(); }
  SuccessorRange getSuccessors() { return SuccessorRange(this); }

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

#endif // MLIR_IR_BLOCK_H
