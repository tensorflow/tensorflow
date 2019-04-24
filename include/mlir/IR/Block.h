//===- Block.h - MLIR Block and Region Classes ------------------*- C++ -*-===//
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
// This file defines Block and Region classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCK_H
#define MLIR_IR_BLOCK_H

#include "mlir/IR/Value.h"
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
class BlockAndValueMapping;
class Region;
class Function;

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

  /// Blocks are maintained in a Region.
  Region *getParent() { return parentValidInstOrderPair.getPointer(); }

  /// Returns the closest surrounding operation that contains this block or
  /// nullptr if this is a top-level block.
  Operation *getContainingOp();

  /// Returns the function that this block is part of, even if the block is
  /// nested under an operation region.
  Function *getFunction();

  /// Insert this block (which must not already be in a function) right before
  /// the specified block.
  void insertBefore(Block *block);

  /// Unlink this Block from its Function and delete it.
  void eraseFromFunction();

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

  /// Erase the argument at 'index' and remove it from the argument list.
  void eraseArgument(unsigned index);

  unsigned getNumArguments() { return arguments.size(); }
  BlockArgument *getArgument(unsigned i) { return arguments[i]; }

  //===--------------------------------------------------------------------===//
  // Operation list management
  //===--------------------------------------------------------------------===//

  /// This is the list of operations in the block.
  using InstListType = llvm::iplist<Operation>;
  InstListType &getOperations() { return operations; }

  // Iteration over the operations in the block.
  using iterator = InstListType::iterator;
  using reverse_iterator = InstListType::reverse_iterator;

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
  Operation *findAncestorInstInBlock(Operation &op);

  /// This drops all operand uses from operations within this block, which is
  /// an essential step in breaking cyclic dependences between references when
  /// they are to be deleted.
  void dropAllReferences();

  /// This drops all uses of values defined in this block or in the blocks of
  /// nested regions wherever the uses are located.
  void dropAllDefinedValueUses();

  /// Returns true if the ordering of the child operations is valid, false
  /// otherwise.
  bool isInstOrderValid() { return parentValidInstOrderPair.getInt(); }

  /// Invalidates the current ordering of operations.
  void invalidateInstOrder() {
    // Validate the current ordering.
    assert(!verifyInstOrder());
    parentValidInstOrderPair.setInt(false);
  }

  /// Verifies the current ordering of child operations matches the
  /// validInstOrder flag. Returns false if the order is valid, true otherwise.
  bool verifyInstOrder();

  /// Recomputes the ordering of child operations within the block.
  void recomputeInstOrder();

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
  void walk(const std::function<void(Operation *)> &callback);

  /// Walk the operations in the specified [begin, end) range of this block in
  /// postorder, calling the callback for each operation.
  void walk(Block::iterator begin, Block::iterator end,
            const std::function<void(Operation *)> &callback);

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
  Block *splitBlock(Operation *splitBeforeInst) {
    return splitBlock(iterator(splitBeforeInst));
  }

  /// Returns pointer to member of operation list.
  static InstListType Block::*getSublistAccess(Operation *) {
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
  llvm::PointerIntPair<Region *, /*IntBits=*/1, bool> parentValidInstOrderPair;

  /// This is the list of operations in the block.
  InstListType operations;

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
  mlir::Region *getContainingRegion();
};
} // end namespace llvm

namespace mlir {

/// This class contains a list of basic blocks and has a notion of the object it
/// is part of - a Function or an Operation.
class Region {
public:
  explicit Region(Function *container = nullptr);
  explicit Region(Operation *container);
  ~Region();

  using RegionType = llvm::iplist<Block>;
  RegionType &getBlocks() { return blocks; }

  // Iteration over the block in the function.
  using iterator = RegionType::iterator;
  using reverse_iterator = RegionType::reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }

  bool empty() { return blocks.empty(); }
  void push_back(Block *block) { blocks.push_back(block); }
  void push_front(Block *block) { blocks.push_front(block); }

  Block &back() { return blocks.back(); }
  Block &front() { return blocks.front(); }

  /// getSublistAccess() - Returns pointer to member of region.
  static RegionType Region::*getSublistAccess(Block *) {
    return &Region::blocks;
  }

  /// Return the region containing this region or nullptr if it is a top-level
  /// region, i.e. a function body region.
  Region *getContainingRegion();

  /// A Region is either a function body or a part of an operation.  If it is
  /// part of an operation, then return the operation, otherwise return null.
  Operation *getContainingOp();

  /// A Region is either a function body or a part of an operation.  If it is
  /// a Function body, then return this function, otherwise return null.
  Function *getContainingFunction();

  /// Return true if this region is a proper ancestor of the `other` region.
  bool isProperAncestor(Region *other);

  /// Return true if this region is ancestor of the `other` region.  A region
  /// is considered as its own ancestor, use `isProperAncestor` to avoid this.
  bool isAncestor(Region *other) {
    return this == other || isProperAncestor(other);
  }

  /// Clone the internal blocks from this region into dest. Any
  /// cloned blocks are appended to the back of dest. If the mapper
  /// contains entries for block arguments, these arguments are not included
  /// in the respective cloned block.
  void cloneInto(Region *dest, BlockAndValueMapping &mapper,
                 MLIRContext *context);

  /// Takes body of another region (that region will have no body after this
  /// operation completes).  The current body of this region is cleared.
  void takeBody(Region &other) {
    blocks.clear();
    blocks.splice(blocks.end(), other.getBlocks());
  }

private:
  RegionType blocks;

  /// This is the object we are part of.
  llvm::PointerUnion<Function *, Operation *> container;
};

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

/// Implement a predecessor iterator as a forward iterator.  This works by
/// walking the use lists of the blocks.  The entries on this list are the
/// BlockOperands that are embedded into terminator operations.  From the
/// operand, we can get the terminator that contains it, and it's parent block
/// is the predecessor.
class PredecessorIterator
    : public llvm::iterator_facade_base<PredecessorIterator,
                                        std::forward_iterator_tag, Block *> {
public:
  PredecessorIterator(BlockOperand *firstOperand)
      : bbUseIterator(firstOperand) {}

  PredecessorIterator &operator=(const PredecessorIterator &rhs) {
    bbUseIterator = rhs.bbUseIterator;
    return *this;
  }

  bool operator==(const PredecessorIterator &rhs) const {
    return bbUseIterator == rhs.bbUseIterator;
  }

  Block *operator*() const;

  PredecessorIterator &operator++() {
    ++bbUseIterator;
    return *this;
  }

  /// Get the successor number in the predecessor terminator.
  unsigned getSuccessorIndex() const;

private:
  using BBUseIterator = ValueUseIterator<BlockOperand>;
  BBUseIterator bbUseIterator;
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
    : public IndexedAccessorIterator<SuccessorIterator, Block, Block> {
public:
  /// Initializes the result iterator to the specified index.
  SuccessorIterator(Block *object, unsigned index)
      : IndexedAccessorIterator<SuccessorIterator, Block, Block>(object,
                                                                 index) {}

  SuccessorIterator(const SuccessorIterator &other)
      : SuccessorIterator(other.object, other.index) {}

  Block *operator*() const { return this->object->getSuccessor(this->index); }

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
