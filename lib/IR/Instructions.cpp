//===- Instructions.cpp - MLIR CFGFunction Instruction Classes ------------===//
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

#include "mlir/IR/Instructions.h"
#include "mlir/IR/BasicBlock.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

// Instructions are deleted through the destroy() member because we don't have
// a virtual destructor.
Instruction::~Instruction() {
  assert(block == nullptr && "instruction destroyed but still in a block");
}

/// Destroy this instruction or one of its subclasses.
void Instruction::destroy(Instruction *inst) {
  switch (inst->getKind()) {
  case Kind::Operation:
    delete cast<OperationInst>(inst);
    break;
  case Kind::Branch:
    delete cast<BranchInst>(inst);
    break;
  case Kind::Return:
    delete cast<ReturnInst>(inst);
    break;
  }
}

CFGFunction *Instruction::getFunction() const {
  return getBlock()->getFunction();
}

//===----------------------------------------------------------------------===//
// OperationInst
//===----------------------------------------------------------------------===//

/// Create a new OperationInst with the specific fields.
OperationInst *OperationInst::create(Identifier name,
                                     ArrayRef<CFGValue *> operands,
                                     ArrayRef<Type *> resultTypes,
                                     ArrayRef<NamedAttribute> attributes,
                                     MLIRContext *context) {
  auto byteSize = totalSizeToAlloc<InstOperand, InstResult>(operands.size(),
                                                            resultTypes.size());
  void *rawMem = ::operator new(byteSize);

  // Initialize the OperationInst part of the instruction.
  auto inst = ::new (rawMem) OperationInst(
      name, operands.size(), resultTypes.size(), attributes, context);

  // Initialize the operands and results.
  auto instOperands = inst->getInstOperands();
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    new (&instOperands[i]) InstOperand(inst, operands[i]);

  auto instResults = inst->getInstResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&instResults[i]) InstResult(resultTypes[i], inst);
  return inst;
}

OperationInst::OperationInst(Identifier name, unsigned numOperands,
                             unsigned numResults,
                             ArrayRef<NamedAttribute> attributes,
                             MLIRContext *context)
    : Operation(name, /*isInstruction=*/ true, attributes, context),
      Instruction(Kind::Operation), numOperands(numOperands),
      numResults(numResults) {}

OperationInst::~OperationInst() {
  // Explicitly run the destructors for the operands and results.
  for (auto &operand : getInstOperands())
    operand.~InstOperand();

  for (auto &result : getInstResults())
    result.~InstResult();
}

mlir::BasicBlock *
llvm::ilist_traits<::mlir::OperationInst>::getContainingBlock() {
  size_t Offset(
      size_t(&((BasicBlock *)nullptr->*BasicBlock::getSublistAccess(nullptr))));
  iplist<OperationInst> *Anchor(static_cast<iplist<OperationInst> *>(this));
  return reinterpret_cast<BasicBlock *>(reinterpret_cast<char *>(Anchor) -
                                           Offset);
}

/// This is a trait method invoked when an instruction is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::OperationInst>::
addNodeToList(OperationInst *inst) {
  assert(!inst->getBlock() && "already in a basic block!");
  inst->block = getContainingBlock();
}

/// This is a trait method invoked when an instruction is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::OperationInst>::
removeNodeFromList(OperationInst *inst) {
  assert(inst->block && "not already in a basic block!");
  inst->block = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::OperationInst>::
transferNodesFromList(ilist_traits<OperationInst> &otherList,
                      instr_iterator first, instr_iterator last) {
  // If we are transferring instructions within the same basic block, the block
  // pointer doesn't need to be updated.
  BasicBlock *curParent = getContainingBlock();
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each instruction.
  for (; first != last; ++first)
    first->block = curParent;
}

/// Unlink this instruction from its BasicBlock and delete it.
void OperationInst::eraseFromBlock() {
  assert(getBlock() && "Instruction has no parent");
  getBlock()->getOperations().erase(this);
}

//===----------------------------------------------------------------------===//
// Terminators
//===----------------------------------------------------------------------===//

/// Remove this terminator from its BasicBlock and delete it.
void TerminatorInst::eraseFromBlock() {
  assert(getBlock() && "Instruction has no parent");
  getBlock()->setTerminator(nullptr);
  TerminatorInst::destroy(this);
}


