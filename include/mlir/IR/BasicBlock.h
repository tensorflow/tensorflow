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
#include <memory>

namespace mlir {
class BBArgument;

/// Each basic block in a CFG function contains a list of basic block arguments,
/// normal instructions, and a terminator instruction.
///
/// Basic blocks form a graph (the CFG) which can be traversed through
/// predecessor and successor edges.
class BasicBlock
  : public llvm::ilist_node_with_parent<BasicBlock, CFGFunction> {
public:
  explicit BasicBlock();
  ~BasicBlock();

  /// Return the function that a BasicBlock is part of.
  CFGFunction *getFunction() const {
    return function;
  }

  /// Unlink this BasicBlock from its CFGFunction and delete it.
  void eraseFromFunction();

  //===--------------------------------------------------------------------===//
  // Block arguments management
  //===--------------------------------------------------------------------===//

  // This is the list of arguments to the block.
  typedef ArrayRef<BBArgument *> BBArgListType;
  BBArgListType getArguments() const { return arguments; }

  using args_iterator = BBArgListType::iterator;
  using reverse_args_iterator = BBArgListType::reverse_iterator;
  args_iterator args_begin() const { return getArguments().begin(); }
  args_iterator args_end() const { return getArguments().end(); }
  reverse_args_iterator args_rbegin() const { return getArguments().rbegin(); }
  reverse_args_iterator args_rend() const { return getArguments().rend(); }

  bool args_empty() const { return arguments.empty(); }

  /// Add one value to the operand list.
  BBArgument *addArgument(Type *type);

  /// Add one argument to the argument list for each type specified in the list.
  llvm::iterator_range<args_iterator> addArguments(ArrayRef<Type *> types);

  unsigned getNumArguments() const { return arguments.size(); }
  BBArgument *getArgument(unsigned i) { return arguments[i]; }
  const BBArgument *getArgument(unsigned i) const { return arguments[i]; }

  //===--------------------------------------------------------------------===//
  // Operation list management
  //===--------------------------------------------------------------------===//

  /// This is the list of operations in the block.
  typedef llvm::iplist<OperationInst> OperationListType;
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
  void push_back(OperationInst *inst) { operations.push_back(inst); }
  void push_front(OperationInst *inst) { operations.push_front(inst); }

  OperationInst &back() { return operations.back(); }
  const OperationInst &back() const {
    return const_cast<BasicBlock *>(this)->back();
  }

  OperationInst &front() { return operations.front(); }
  const OperationInst &front() const {
    return const_cast<BasicBlock*>(this)->front();
  }

  //===--------------------------------------------------------------------===//
  // Terminator management
  //===--------------------------------------------------------------------===//

  /// Change the terminator of this block to the specified instruction.
  void setTerminator(TerminatorInst *inst);

  TerminatorInst *getTerminator() const { return terminator; }

  void print(raw_ostream &os) const;
  void dump() const;

  /// getSublistAccess() - Returns pointer to member of operation list
  static OperationListType BasicBlock::*getSublistAccess(OperationInst*) {
    return &BasicBlock::operations;
  }

private:
  CFGFunction *function = nullptr;

  /// This is the list of operations in the block.
  OperationListType operations;

  /// This is the list of arguments to the block.
  std::vector<BBArgument *> arguments;

  /// This is the owning reference to the terminator of the block.
  TerminatorInst *terminator = nullptr;

  BasicBlock(const BasicBlock&) = delete;
  void operator=(const BasicBlock&) = delete;

  friend struct llvm::ilist_traits<BasicBlock>;
};

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for OperationInst
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
