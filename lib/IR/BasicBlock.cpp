//===- BasicBlock.cpp - MLIR BasicBlock Class -----------------------------===//
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

#include "mlir/IR/BasicBlock.h"
#include "mlir/IR/CFGFunction.h"
using namespace mlir;

BasicBlock::BasicBlock() {}

BasicBlock::~BasicBlock() {
  if (terminator)
    terminator->eraseFromBlock();
  for (BBArgument *arg : arguments)
    delete arg;
  arguments.clear();
}

/// Unlink this BasicBlock from its CFGFunction and delete it.
void BasicBlock::eraseFromFunction() {
  assert(getFunction() && "BasicBlock has no parent");
  getFunction()->getBlocks().erase(this);
}

void BasicBlock::setTerminator(TerminatorInst *inst) {
  // If we already had a terminator, abandon it.
  if (terminator)
    terminator->block = nullptr;

  // Reset our terminator to the new instruction.
  terminator = inst;
  if (inst)
    inst->block = this;
}

mlir::CFGFunction *
llvm::ilist_traits<::mlir::BasicBlock>::getContainingFunction() {
  size_t Offset(
    size_t(&((CFGFunction *)nullptr->*CFGFunction::getSublistAccess(nullptr))));
  iplist<BasicBlock> *Anchor(static_cast<iplist<BasicBlock> *>(this));
  return reinterpret_cast<CFGFunction *>(reinterpret_cast<char *>(Anchor) -
                                           Offset);
}

/// This is a trait method invoked when a basic block is added to a function.
/// We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
addNodeToList(BasicBlock *block) {
  assert(!block->function && "already in a function!");
  block->function = getContainingFunction();
}

/// This is a trait method invoked when an instruction is removed from a
/// function.  We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
removeNodeFromList(BasicBlock *block) {
  assert(block->function && "not already in a function!");
  block->function = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
transferNodesFromList(ilist_traits<BasicBlock> &otherList,
                      block_iterator first, block_iterator last) {
  // If we are transferring instructions within the same function, the parent
  // pointer doesn't need to be updated.
  CFGFunction *curParent = getContainingFunction();
  if (curParent == otherList.getContainingFunction())
    return;

  // Update the 'function' member of each BasicBlock.
  for (; first != last; ++first)
    first->function = curParent;
}

BBArgument *BasicBlock::addArgument(Type *type) {
  arguments.push_back(new BBArgument(type, this));
  return arguments.back();
}

llvm::iterator_range<BasicBlock::BBArgListType::iterator>
BasicBlock::addArguments(ArrayRef<Type *> types) {
  auto initial_size = arguments.size();
  for (auto *type : types) {
    addArgument(type);
  }
  return {arguments.data() + initial_size, arguments.data() + arguments.size()};
}
