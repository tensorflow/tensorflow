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

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead.  When this returns
/// there are zero uses of 'this'.
void IRObjectWithUseList::replaceAllUsesWith(IRObjectWithUseList *newValue) {
  assert(this != newValue && "cannot RAUW a value with itself");
  while (!use_empty()) {
    use_begin()->set(newValue);
  }
}

/// Return the result number of this result.
unsigned InstResult::getResultNumber() const {
  // Results are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getOwner()->getInstResults()[0];
}

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

// Instructions are deleted through the destroy() member because we don't have
// a virtual destructor.
Instruction::~Instruction() {
  assert(block == nullptr && "instruction destroyed but still in a block");
}

/// Destroy this instruction or one of its subclasses.
void Instruction::destroy() {
  switch (getKind()) {
  case Kind::Operation:
    cast<OperationInst>(this)->destroy();
    break;
  case Kind::Branch:
    delete cast<BranchInst>(this);
    break;
  case Kind::CondBranch:
    delete cast<CondBranchInst>(this);
    break;
  case Kind::Return:
    cast<ReturnInst>(this)->destroy();
    break;
  }
}

void OperationInst::destroy() {
  this->~OperationInst();
  free(this);
}

CFGFunction *Instruction::getFunction() const {
  return getBlock()->getFunction();
}

unsigned Instruction::getNumOperands() const {
  switch (getKind()) {
  case Kind::Operation:
    return cast<OperationInst>(this)->getNumOperands();
  case Kind::Branch:
    return cast<BranchInst>(this)->getNumOperands();
  case Kind::CondBranch:
    return cast<CondBranchInst>(this)->getNumOperands();
  case Kind::Return:
    return cast<ReturnInst>(this)->getNumOperands();
  }
}

MutableArrayRef<InstOperand> Instruction::getInstOperands() {
  switch (getKind()) {
  case Kind::Operation:
    return cast<OperationInst>(this)->getInstOperands();
  case Kind::Branch:
    return cast<BranchInst>(this)->getInstOperands();
  case Kind::CondBranch:
    return cast<CondBranchInst>(this)->getInstOperands();
  case Kind::Return:
    return cast<ReturnInst>(this)->getInstOperands();
  }
}

/// This drops all operand uses from this instruction, which is an essential
/// step in breaking cyclic dependences between references when they are to
/// be deleted.
void Instruction::dropAllReferences() {
  for (auto &op : getInstOperands())
    op.drop();

  if (auto *term = dyn_cast<TerminatorInst>(this))
    for (auto &dest : term->getDestinations())
      dest.drop();
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
  void *rawMem = malloc(byteSize);

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
    : Operation(name, /*isInstruction=*/true, attributes, context),
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
void llvm::ilist_traits<::mlir::OperationInst>::addNodeToList(
    OperationInst *inst) {
  assert(!inst->getBlock() && "already in a basic block!");
  inst->block = getContainingBlock();
}

/// This is a trait method invoked when an instruction is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::OperationInst>::removeNodeFromList(
    OperationInst *inst) {
  assert(inst->block && "not already in a basic block!");
  inst->block = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::OperationInst>::transferNodesFromList(
    ilist_traits<OperationInst> &otherList, instr_iterator first,
    instr_iterator last) {
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

/// If this value is the result of an OperationInst, return the instruction
/// that defines it.
OperationInst *SSAValue::getDefiningInst() {
  if (auto *result = dyn_cast<InstResult>(this))
    return result->getOwner();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TerminatorInst
//===----------------------------------------------------------------------===//

/// Remove this terminator from its BasicBlock and delete it.
void TerminatorInst::eraseFromBlock() {
  assert(getBlock() && "Instruction has no parent");
  getBlock()->setTerminator(nullptr);
  destroy();
}

/// Return the list of destination entries that this terminator branches to.
MutableArrayRef<BBDestination> TerminatorInst::getDestinations() {
  switch (getKind()) {
  case Kind::Operation:
    assert(0 && "not a terminator");
  case Kind::Branch:
    return cast<BranchInst>(this)->getDestinations();
  case Kind::CondBranch:
    return cast<CondBranchInst>(this)->getDestinations();
  case Kind::Return:
    // Return has no basic block successors.
    return {};
  }
}

//===----------------------------------------------------------------------===//
// ReturnInst
//===----------------------------------------------------------------------===//

/// Create a new OperationInst with the specific fields.
ReturnInst *ReturnInst::create(ArrayRef<CFGValue *> operands) {
  auto byteSize = totalSizeToAlloc<InstOperand>(operands.size());
  void *rawMem = malloc(byteSize);

  // Initialize the ReturnInst part of the instruction.
  auto inst = ::new (rawMem) ReturnInst(operands.size());

  // Initialize the operands and results.
  auto instOperands = inst->getInstOperands();
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    new (&instOperands[i]) InstOperand(inst, operands[i]);
  return inst;
}

void ReturnInst::destroy() {
  this->~ReturnInst();
  free(this);
}

ReturnInst::~ReturnInst() {
  // Explicitly run the destructors for the operands.
  for (auto &operand : getInstOperands())
    operand.~InstOperand();
}

//===----------------------------------------------------------------------===//
// BranchInst
//===----------------------------------------------------------------------===//

BranchInst::BranchInst(BasicBlock *dest)
    : TerminatorInst(Kind::Branch), dest(this, dest) {}

void BranchInst::setDest(BasicBlock *block) { dest.set(block); }

/// Add one value to the operand list.
void BranchInst::addOperand(CFGValue *value) {
  operands.emplace_back(InstOperand(this, value));
}

/// Add a list of values to the operand list.
void BranchInst::addOperands(ArrayRef<CFGValue *> values) {
  operands.reserve(operands.size() + values.size());
  for (auto *value : values)
    addOperand(value);
}

//===----------------------------------------------------------------------===//
// CondBranchInst
//===----------------------------------------------------------------------===//

CondBranchInst::CondBranchInst(CFGValue *condition, BasicBlock *trueDest,
                               BasicBlock *falseDest)
    : TerminatorInst(Kind::CondBranch),
      condition(condition), dests{{this}, {this}}, numTrueOperands(0) {
  dests[falseIndex].set(falseDest);
  dests[trueIndex].set(trueDest);
}

/// Add one value to the true operand list.
void CondBranchInst::addTrueOperand(CFGValue *value) {
  assert(getNumFalseOperands() == 0 &&
         "Must insert all true operands before false operands!");
  operands.emplace_back(InstOperand(this, value));
  ++numTrueOperands;
}

/// Add a list of values to the true operand list.
void CondBranchInst::addTrueOperands(ArrayRef<CFGValue *> values) {
  operands.reserve(operands.size() + values.size());
  for (auto *value : values)
    addTrueOperand(value);
}

/// Add one value to the false operand list.
void CondBranchInst::addFalseOperand(CFGValue *value) {
  operands.emplace_back(InstOperand(this, value));
}

/// Add a list of values to the false operand list.
void CondBranchInst::addFalseOperands(ArrayRef<CFGValue *> values) {
  operands.reserve(operands.size() + values.size());
  for (auto *value : values)
    addFalseOperand(value);
}
