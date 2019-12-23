//===- BlockSupport.h -------------------------------------------*- C++ -*-===//
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
// This file defines a number of support types for the Block class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCK_SUPPORT_H
#define MLIR_IR_BLOCK_SUPPORT_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class Block;

using BlockOperand = IROperandImpl<Block>;

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

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//

/// This class implements the successor iterators for Block.
class SuccessorRange final
    : public detail::indexed_accessor_range_base<SuccessorRange, BlockOperand *,
                                                 Block *, Block *, Block *> {
public:
  using RangeBaseT::RangeBaseT;
  SuccessorRange(Block *block);
  SuccessorRange(Operation *term);

private:
  /// See `detail::indexed_accessor_range_base` for details.
  static BlockOperand *offset_base(BlockOperand *object, ptrdiff_t index) {
    return object + index;
  }
  /// See `detail::indexed_accessor_range_base` for details.
  static Block *dereference_iterator(BlockOperand *object, ptrdiff_t index) {
    return object[index].get();
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

} // end namespace mlir

namespace llvm {

//===----------------------------------------------------------------------===//
// ilist_traits for Operation
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ilist_traits for Block
//===----------------------------------------------------------------------===//

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

#endif // MLIR_IR_BLOCK_SUPPORT_H
