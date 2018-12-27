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
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLIRContext.h"
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

/// Return which operand this is in the operand list.
template <> unsigned InstOperand::getOperandNumber() const {
  return this - &getOwner()->getInstOperands()[0];
}

/// Return which operand this is in the operand list.
template <> unsigned BasicBlockOperand::getOperandNumber() const {
  return this - &getOwner()->getBasicBlockOperands()[0];
}

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

void Instruction::setSuccessor(BasicBlock *block, unsigned index) {
  assert(index < getNumSuccessors());
  getBasicBlockOperands()[index].set(block);
}

/// Create a new Instruction with the specified fields.
Instruction *Instruction::create(Location location, OperationName name,
                                 ArrayRef<CFGValue *> operands,
                                 ArrayRef<Type> resultTypes,
                                 ArrayRef<NamedAttribute> attributes,
                                 ArrayRef<BasicBlock *> successors,
                                 MLIRContext *context) {
  unsigned numSuccessors = successors.size();
  // Input operands are nullptr-separated for each successors in the case of
  // terminators, the nullptr aren't actually stored.
  unsigned numOperands = operands.size() - llvm::count(operands, nullptr);

  auto byteSize =
      totalSizeToAlloc<InstResult, InstOperand, BasicBlockOperand, unsigned>(
          resultTypes.size(), numOperands, numSuccessors, numSuccessors);
  void *rawMem = malloc(byteSize);

  // Initialize the Instruction part of the instruction.
  auto inst = ::new (rawMem)
      Instruction(location, name, resultTypes.size(), numOperands,
                  numSuccessors, attributes, context);

  // Initialize the results and operands.
  auto instResults = inst->getInstResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&instResults[i]) InstResult(resultTypes[i], inst);

  // instOperandIt is the iterator in the tail-allocated memory for the
  // operands, and operandIt is the iterator in the input operands array.
  // operandIt skips nullptr in the input that acts as sentinels marking the
  // separation between multiple basic block successors for terminators.
  auto instOperandIt = inst->getInstOperands().begin();
  unsigned operandIt = 0, operandE = operands.size();
  for (; operandIt != operandE; ++operandIt) {
    if (!operands[operandIt])
      break;
    new (instOperandIt++) InstOperand(inst, operands[operandIt]);
  }

  // Check to see if a sentinel (nullptr) operand was encountered.
  unsigned currentSuccNum = 0;
  if (operandIt != operandE) {
    assert(inst->isTerminator() &&
           "Sentinel operand found in non terminator operand list.");
    auto instBlockOperands = inst->getBasicBlockOperands();
    unsigned *succOperandCountIt = inst->getTrailingObjects<unsigned>();
    unsigned *succOperandCountE = succOperandCountIt + numSuccessors;
    (void)succOperandCountE;

    for (; operandIt != operandE; ++operandIt) {
      // If we encounter a sentinal branch to the next operand update the count
      // variable.
      if (!operands[operandIt]) {
        assert(currentSuccNum < numSuccessors);

        // After the first iteration update the successor operand count
        // variable.
        if (currentSuccNum != 0) {
          ++succOperandCountIt;
          assert(succOperandCountIt != succOperandCountE &&
                 "More sentinal operands than successors.");
        }

        new (&instBlockOperands[currentSuccNum])
            BasicBlockOperand(inst, successors[currentSuccNum]);
        *succOperandCountIt = 0;
        ++currentSuccNum;
        continue;
      }
      new (instOperandIt++) InstOperand(inst, operands[operandIt]);
      ++(*succOperandCountIt);
    }
  }

  // Verify that the amount of sentinal operands is equivalent to the number of
  // successors.
  assert(currentSuccNum == numSuccessors);
  return inst;
}

Instruction *Instruction::clone() const {
  SmallVector<CFGValue *, 8> operands;
  SmallVector<Type, 8> resultTypes;
  SmallVector<BasicBlock *, 1> successors;

  // Put together the results.
  for (auto *result : getResults())
    resultTypes.push_back(result->getType());

  // If the instruction is a terminator the successor and non-successor operand
  // lists are interleaved with sentinal(nullptr) operands.
  if (isTerminator()) {
    // To interleave the operand lists we iterate in reverse and insert the
    // operands in-place.
    operands.resize(getNumOperands() + getNumSuccessors());
    successors.resize(getNumSuccessors());
    int cloneOperandIt = operands.size() - 1, operandIt = getNumOperands() - 1;
    for (int succIt = getNumSuccessors() - 1, succE = 0; succIt >= succE;
         --succIt) {
      successors[succIt] = const_cast<BasicBlock *>(getSuccessor(succIt));

      // Add the successor operands in-place in reverse order.
      for (unsigned i = 0, e = getNumSuccessorOperands(succIt); i != e;
           ++i, --cloneOperandIt, --operandIt) {
        operands[cloneOperandIt] =
            const_cast<CFGValue *>(getOperand(operandIt));
      }

      // Add a null operand for the barrier.
      operands[cloneOperandIt--] = nullptr;
    }

    // Add the rest of the non-successor operands.
    for (; cloneOperandIt >= 0; --cloneOperandIt, --operandIt)
      operands[cloneOperandIt] = const_cast<CFGValue *>(getOperand(operandIt));
    // For non terminators we can simply add each of the instructions in place.
  } else {
    for (auto *operand : getOperands())
      operands.push_back(const_cast<CFGValue *>(operand));
  }

  return create(getLoc(), getName(), operands, resultTypes, getAttrs(),
                successors, getContext());
}

Instruction::Instruction(Location location, OperationName name,
                         unsigned numResults, unsigned numOperands,
                         unsigned numSuccessors,
                         ArrayRef<NamedAttribute> attributes,
                         MLIRContext *context)
    : Operation(/*isInstruction=*/true, name, attributes, context),
      IROperandOwner(IROperandOwner::Kind::Instruction, location),
      numResults(numResults), numSuccs(numSuccessors),
      numOperands(numOperands) {}

Instruction::~Instruction() {
  assert(block == nullptr && "instruction destroyed but still in a block");

  // Explicitly run the destructors for the results
  for (auto &result : getInstResults())
    result.~InstResult();

  // Explicitly run the destructors for the operands.
  for (auto &result : getInstOperands())
    result.~InstOperand();

  // Explicitly run the destructors for the successors.
  if (isTerminator())
    for (auto &successor : getBasicBlockOperands())
      successor.~BasicBlockOperand();
}

void Instruction::eraseOperand(unsigned index) {
  assert(index < getNumOperands());
  auto Operands = getInstOperands();
  // Shift all operands down by 1.
  std::rotate(&Operands[index], &Operands[index + 1],
              &Operands[numOperands - 1]);
  --numOperands;
  Operands[getNumOperands()].~InstOperand();
}

/// Destroy this instruction.
void Instruction::destroy() {
  this->~Instruction();
  free(this);
}

CFGFunction *Instruction::getFunction() {
  auto *block = getBlock();
  return block ? block->getFunction() : nullptr;
}

/// This drops all operand uses from this instruction, which is an essential
/// step in breaking cyclic dependences between references when they are to
/// be deleted.
void Instruction::dropAllReferences() {
  for (auto &op : getInstOperands())
    op.drop();

  if (isTerminator())
    for (auto &dest : getBasicBlockOperands())
      dest.drop();
}

/// Emit a note about this instruction, reporting up to any diagnostic
/// handlers that may be listening.
void Instruction::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void Instruction::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns true.  NOTE: This may terminate the containing application, only use
/// when the IR is in an inconsistent state.
bool Instruction::emitError(const Twine &message) const {
  return getContext()->emitError(getLoc(), message);
}

void llvm::ilist_traits<::mlir::Instruction>::deleteNode(Instruction *inst) {
  inst->destroy();
}

mlir::BasicBlock *
llvm::ilist_traits<::mlir::Instruction>::getContainingBlock() {
  size_t Offset(
      size_t(&((BasicBlock *)nullptr->*BasicBlock::getSublistAccess(nullptr))));
  iplist<Instruction> *Anchor(static_cast<iplist<Instruction> *>(this));
  return reinterpret_cast<BasicBlock *>(reinterpret_cast<char *>(Anchor) -
                                        Offset);
}

/// This is a trait method invoked when an instruction is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::addNodeToList(Instruction *inst) {
  assert(!inst->getBlock() && "already in a basic block!");
  inst->block = getContainingBlock();
}

/// This is a trait method invoked when an instruction is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::removeNodeFromList(
    Instruction *inst) {
  assert(inst->block && "not already in a basic block!");
  inst->block = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::transferNodesFromList(
    ilist_traits<Instruction> &otherList, instr_iterator first,
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
void Instruction::erase() {
  assert(getBlock() && "Instruction has no parent");
  getBlock()->getOperations().erase(this);
}

/// Unlink this operation instruction from its current basic block and insert
/// it right before `existingInst` which may be in the same or another block
/// in the same function.
void Instruction::moveBefore(Instruction *existingInst) {
  assert(existingInst && "Cannot move before a null instruction");
  return moveBefore(existingInst->getBlock(), existingInst->getIterator());
}

/// Unlink this operation instruction from its current basic block and insert
/// it right before `iterator` in the specified basic block.
void Instruction::moveBefore(BasicBlock *block,
                             llvm::iplist<Instruction>::iterator iterator) {
  block->getOperations().splice(iterator, getBlock()->getOperations(),
                                getIterator());
}
