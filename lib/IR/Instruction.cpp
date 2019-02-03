//===- Instruction.cpp - MLIR Instruction Classes -------------------------===//
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

#include "AttributeListStorage.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/IR/Instructions.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// InstResult
//===----------------------------------------------------------------------===//

/// Return the result number of this result.
unsigned InstResult::getResultNumber() const {
  // Results are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getOwner()->getInstResults()[0];
}

//===----------------------------------------------------------------------===//
// InstOperand
//===----------------------------------------------------------------------===//

/// Return which operand this is in the operand list.
template <> unsigned InstOperand::getOperandNumber() const {
  return this - &getOwner()->getInstOperands()[0];
}

/// Return which operand this is in the operand list.
template <> unsigned BlockOperand::getOperandNumber() const {
  return this - &getOwner()->getBlockOperands()[0];
}

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

/// Replace the operands contained in the storage with the ones provided in
/// 'operands'.
void detail::OperandStorage::setOperands(Instruction *owner,
                                         ArrayRef<Value *> operands) {
  // If the number of operands is less than or equal to the current amount, we
  // can just update in place.
  if (operands.size() <= numOperands) {
    auto instOperands = getInstOperands();

    // If the number of new operands is less than the current count, then remove
    // any extra operands.
    for (unsigned i = operands.size(); i != numOperands; ++i)
      instOperands[i].~InstOperand();

    // Set the operands in place.
    numOperands = operands.size();
    for (unsigned i = 0; i != numOperands; ++i)
      instOperands[i].set(operands[i]);
    return;
  }

  // Otherwise, we need to be resizable.
  assert(resizable && "Only resizable operations may add operands");

  // Grow the capacity if necessary.
  auto &resizeUtil = getResizableStorage();
  if (resizeUtil.capacity < operands.size())
    grow(resizeUtil, operands.size());

  // Set the operands.
  InstOperand *opBegin = getRawOperands();
  for (unsigned i = 0; i != numOperands; ++i)
    opBegin[i].set(operands[i]);
  for (unsigned e = operands.size(); numOperands != e; ++numOperands)
    new (&opBegin[numOperands]) InstOperand(owner, operands[numOperands]);
}

/// Erase an operand held by the storage.
void detail::OperandStorage::eraseOperand(unsigned index) {
  assert(index < size());
  auto Operands = getInstOperands();
  --numOperands;

  // Shift all operands down by 1 if the operand to remove is not at the end.
  if (index != numOperands)
    std::rotate(&Operands[index], &Operands[index + 1], &Operands[numOperands]);
  Operands[numOperands].~InstOperand();
}

/// Grow the internal operand storage.
void detail::OperandStorage::grow(ResizableStorage &resizeUtil,
                                  size_t minSize) {
  // Allocate a new storage array.
  resizeUtil.capacity =
      std::max(size_t(llvm::NextPowerOf2(resizeUtil.capacity + 2)), minSize);
  InstOperand *newStorage = static_cast<InstOperand *>(
      llvm::safe_malloc(resizeUtil.capacity * sizeof(InstOperand)));

  // Move the current operands to the new storage.
  auto operands = getInstOperands();
  std::uninitialized_copy(std::make_move_iterator(operands.begin()),
                          std::make_move_iterator(operands.end()), newStorage);

  // Destroy the original operands and update the resizable storage pointer.
  for (auto &operand : operands)
    operand.~InstOperand();
  resizeUtil.setDynamicStorage(newStorage);
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
  switch (this->getKind()) {
  case Kind::OperationInst:
    cast<OperationInst>(this)->destroy();
    break;
  }
}

/// Return the context this operation is associated with.
MLIRContext *Instruction::getContext() const {
  return cast<OperationInst>(this)->getContext();
}

Instruction *Instruction::getParentInst() const {
  return block ? block->getContainingInst() : nullptr;
}

Function *Instruction::getFunction() const {
  return block ? block->getFunction() : nullptr;
}

Value *Instruction::getOperand(unsigned idx) {
  return getInstOperand(idx).get();
}

const Value *Instruction::getOperand(unsigned idx) const {
  return getInstOperand(idx).get();
}

// Value can be used as a dimension id if it is valid as a symbol, or
// it is an induction variable, or it is a result of affine apply operation
// with dimension id arguments.
bool Value::isValidDim() const {
  if (auto *inst = getDefiningInst()) {
    // Top level instruction or constant operation is ok.
    if (inst->getParentInst() == nullptr || inst->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = inst->dyn_cast<AffineApplyOp>())
      return op->isValidDim();
    return false;
  }
  // This value is either a function argument or an induction variable. Both
  // are ok.
  return true;
}

// Value can be used as a symbol if it is a constant, or it is defined at
// the top level, or it is a result of affine apply operation with symbol
// arguments.
bool Value::isValidSymbol() const {
  if (auto *inst = getDefiningInst()) {
    // Top level instruction or constant operation is ok.
    if (inst->getParentInst() == nullptr || inst->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = inst->dyn_cast<AffineApplyOp>())
      return op->isValidSymbol();
    return false;
  }
  // Otherwise, the only valid symbol is a function argument.
  auto *arg = dyn_cast<BlockArgument>(this);
  return arg && arg->isFunctionArgument();
}

void Instruction::setOperand(unsigned idx, Value *value) {
  getInstOperand(idx).set(value);
}

unsigned Instruction::getNumOperands() const {
  switch (getKind()) {
  case Kind::OperationInst:
    return cast<OperationInst>(this)->getNumOperands();
  }
}

MutableArrayRef<InstOperand> Instruction::getInstOperands() {
  switch (getKind()) {
  case Kind::OperationInst:
    return cast<OperationInst>(this)->getInstOperands();
  }
}

/// Emit a note about this instruction, reporting up to any diagnostic
/// handlers that may be listening.
void Instruction::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this instruction, reporting up to any diagnostic
/// handlers that may be listening.
void Instruction::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns true.  NOTE: This may terminate the containing application, only
/// use when the IR is in an inconsistent state.
bool Instruction::emitError(const Twine &message) const {
  return getContext()->emitError(getLoc(), message);
}

/// Given an instruction 'other' that is within the same parent block, return
/// whether the current instruction is before 'other' in the instruction list
/// of the parent block.
/// Note: This function has an average complexity of O(1), but worst case may
/// take O(N) where N is the number of instructions within the parent block.
bool Instruction::isBeforeInBlock(const Instruction *other) const {
  assert(block && "Instructions without parent blocks have no order.");
  assert(other && other->block == block &&
         "Expected other instruction to have the same parent block.");
  // Recompute the parent ordering if necessary.
  if (!block->isInstOrderValid())
    block->recomputeInstOrder();
  return orderIndex < other->orderIndex;
}

/// Returns whether the Instruction is a terminator.
bool Instruction::isTerminator() const {
  if (auto *op = dyn_cast<OperationInst>(this))
    return op->isTerminator();
  return false;
}

//===----------------------------------------------------------------------===//
// ilist_traits for Instruction
//===----------------------------------------------------------------------===//

void llvm::ilist_traits<::mlir::Instruction>::deleteNode(Instruction *inst) {
  inst->destroy();
}

Block *llvm::ilist_traits<::mlir::Instruction>::getContainingBlock() {
  size_t Offset(size_t(&((Block *)nullptr->*Block::getSublistAccess(nullptr))));
  iplist<Instruction> *Anchor(static_cast<iplist<Instruction> *>(this));
  return reinterpret_cast<Block *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a instruction is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::addNodeToList(Instruction *inst) {
  assert(!inst->getBlock() && "already in a instruction block!");
  inst->block = getContainingBlock();

  // Invalidate the block ordering.
  inst->block->invalidateInstOrder();
}

/// This is a trait method invoked when a instruction is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::removeNodeFromList(
    Instruction *inst) {
  assert(inst->block && "not already in a instruction block!");
  inst->block = nullptr;
}

/// This is a trait method invoked when a instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Instruction>::transferNodesFromList(
    ilist_traits<Instruction> &otherList, inst_iterator first,
    inst_iterator last) {
  Block *curParent = getContainingBlock();

  // Invalidate the ordering of the parent block.
  curParent->invalidateInstOrder();

  // If we are transferring instructions within the same block, the block
  // pointer doesn't need to be updated.
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each instruction.
  for (; first != last; ++first)
    first->block = curParent;
}

/// Remove this instruction (and its descendants) from its Block and delete
/// all of them.
void Instruction::erase() {
  assert(getBlock() && "Instruction has no block");
  getBlock()->getInstructions().erase(this);
}

/// Unlink this instruction from its current block and insert it right before
/// `existingInst` which may be in the same or another block in the same
/// function.
void Instruction::moveBefore(Instruction *existingInst) {
  moveBefore(existingInst->getBlock(), existingInst->getIterator());
}

/// Unlink this operation instruction from its current basic block and insert
/// it right before `iterator` in the specified basic block.
void Instruction::moveBefore(Block *block,
                             llvm::iplist<Instruction>::iterator iterator) {
  block->getInstructions().splice(iterator, getBlock()->getInstructions(),
                                  getIterator());
}

/// This drops all operand uses from this instruction, which is an essential
/// step in breaking cyclic dependences between references when they are to
/// be deleted.
void Instruction::dropAllReferences() {
  for (auto &op : getInstOperands())
    op.drop();

  switch (getKind()) {
  case Kind::OperationInst: {
    auto *opInst = cast<OperationInst>(this);
    if (isTerminator())
      for (auto &dest : opInst->getBlockOperands())
        dest.drop();
    for (auto &blockList : opInst->getBlockLists())
      for (Block &block : blockList)
        block.dropAllReferences();
    break;
  }
  }
}

//===----------------------------------------------------------------------===//
// OperationInst
//===----------------------------------------------------------------------===//

/// Create a new OperationInst with the specific fields.
OperationInst *
OperationInst::create(Location location, OperationName name,
                      ArrayRef<Value *> operands, ArrayRef<Type> resultTypes,
                      ArrayRef<NamedAttribute> attributes,
                      ArrayRef<Block *> successors, unsigned numBlockLists,
                      bool resizableOperandList, MLIRContext *context) {
  unsigned numSuccessors = successors.size();

  // Input operands are nullptr-separated for each successors in the case of
  // terminators, the nullptr aren't actually stored.
  unsigned numOperands = operands.size() - numSuccessors;

  // Compute the byte size for the instruction and the operand storage.
  auto byteSize = totalSizeToAlloc<InstResult, BlockOperand, unsigned,
                                   BlockList, detail::OperandStorage>(
      resultTypes.size(), numSuccessors, numSuccessors, numBlockLists,
      /*detail::OperandStorage*/ 1);
  byteSize += llvm::alignTo(detail::OperandStorage::additionalAllocSize(
                                numOperands, resizableOperandList),
                            alignof(OperationInst));
  void *rawMem = malloc(byteSize);

  // Initialize the OperationInst part of the instruction.
  auto inst = ::new (rawMem)
      OperationInst(location, name, resultTypes.size(), numSuccessors,
                    numBlockLists, attributes, context);

  assert((numSuccessors == 0 || inst->isTerminator()) &&
         "unexpected successors in a non-terminator operation");

  // Initialize the block lists.
  for (unsigned i = 0; i != numBlockLists; ++i)
    new (&inst->getBlockList(i)) BlockList(inst);

  // Initialize the results and operands.
  new (&inst->getOperandStorage())
      detail::OperandStorage(numOperands, resizableOperandList);

  auto instResults = inst->getInstResults();
  for (unsigned i = 0, e = resultTypes.size(); i != e; ++i)
    new (&instResults[i]) InstResult(resultTypes[i], inst);

  auto InstOperands = inst->getInstOperands();

  // Initialize normal operands.
  unsigned operandIt = 0, operandE = operands.size();
  unsigned nextOperand = 0;
  for (; operandIt != operandE; ++operandIt) {
    // Null operands are used as sentinals between successor operand lists. If
    // we encounter one here, break and handle the successor operands lists
    // separately below.
    if (!operands[operandIt])
      break;
    new (&InstOperands[nextOperand++]) InstOperand(inst, operands[operandIt]);
  }

  unsigned currentSuccNum = 0;
  if (operandIt == operandE) {
    // Verify that the amount of sentinal operands is equivalent to the number
    // of successors.
    assert(currentSuccNum == numSuccessors);
    return inst;
  }

  assert(inst->isTerminator() &&
         "Sentinal operand found in non terminator operand list.");
  auto instBlockOperands = inst->getBlockOperands();
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
          BlockOperand(inst, successors[currentSuccNum]);
      *succOperandCountIt = 0;
      ++currentSuccNum;
      continue;
    }
    new (&InstOperands[nextOperand++]) InstOperand(inst, operands[operandIt]);
    ++(*succOperandCountIt);
  }

  // Verify that the amount of sentinal operands is equivalent to the number of
  // successors.
  assert(currentSuccNum == numSuccessors);

  return inst;
}

OperationInst::OperationInst(Location location, OperationName name,
                             unsigned numResults, unsigned numSuccessors,
                             unsigned numBlockLists,
                             ArrayRef<NamedAttribute> attributes,
                             MLIRContext *context)
    : Instruction(location), numResults(numResults), numSuccs(numSuccessors),
      numBlockLists(numBlockLists), name(name) {
#ifndef NDEBUG
  for (auto elt : attributes)
    assert(elt.second != nullptr && "Attributes cannot have null entries");
#endif

  this->attrs = AttributeListStorage::get(attributes, context);
}

OperationInst::~OperationInst() {
  // Explicitly run the destructors for the operands and results.
  getOperandStorage().~OperandStorage();

  for (auto &result : getInstResults())
    result.~InstResult();

  // Explicitly run the destructors for the successors.
  if (isTerminator())
    for (auto &successor : getBlockOperands())
      successor.~BlockOperand();

  // Explicitly destroy the block list.
  for (auto &blockList : getBlockLists())
    blockList.~BlockList();
}

/// Return true if there are no users of any results of this operation.
bool OperationInst::use_empty() const {
  for (auto *result : getResults())
    if (!result->use_empty())
      return false;
  return true;
}

ArrayRef<NamedAttribute> OperationInst::getAttrs() const {
  if (!attrs)
    return {};
  return attrs->getElements();
}

void OperationInst::destroy() {
  this->~OperationInst();
  free(this);
}

/// Return the context this operation is associated with.
MLIRContext *OperationInst::getContext() const {
  // If we have a result or operand type, that is a constant time way to get
  // to the context.
  if (getNumResults())
    return getResult(0)->getType().getContext();
  if (getNumOperands())
    return getOperand(0)->getType().getContext();

  // In the very odd case where we have no operands or results, fall back to
  // doing a find.
  return getFunction()->getContext();
}

bool OperationInst::isReturn() const { return isa<ReturnOp>(); }

void OperationInst::setSuccessor(Block *block, unsigned index) {
  assert(index < getNumSuccessors());
  getBlockOperands()[index].set(block);
}

auto OperationInst::getNonSuccessorOperands() const
    -> llvm::iterator_range<const_operand_iterator> {
  return {const_operand_iterator(this, 0),
          const_operand_iterator(this, getSuccessorOperandIndex(0))};
}
auto OperationInst::getNonSuccessorOperands()
    -> llvm::iterator_range<operand_iterator> {
  return {operand_iterator(this, 0),
          operand_iterator(this, getSuccessorOperandIndex(0))};
}

auto OperationInst::getSuccessorOperands(unsigned index) const
    -> llvm::iterator_range<const_operand_iterator> {
  assert(isTerminator() && "Only terminators have successors.");
  unsigned succOperandIndex = getSuccessorOperandIndex(index);
  return {const_operand_iterator(this, succOperandIndex),
          const_operand_iterator(this, succOperandIndex +
                                           getNumSuccessorOperands(index))};
}
auto OperationInst::getSuccessorOperands(unsigned index)
    -> llvm::iterator_range<operand_iterator> {
  assert(isTerminator() && "Only terminators have successors.");
  unsigned succOperandIndex = getSuccessorOperandIndex(index);
  return {operand_iterator(this, succOperandIndex),
          operand_iterator(this,
                           succOperandIndex + getNumSuccessorOperands(index))};
}

/// If an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
void OperationInst::setAttr(Identifier name, Attribute value) {
  assert(value && "attributes may never be null");
  auto origAttrs = getAttrs();

  SmallVector<NamedAttribute, 8> newAttrs(origAttrs.begin(), origAttrs.end());
  auto *context = getContext();

  // If we already have this attribute, replace it.
  for (auto &elt : newAttrs)
    if (elt.first == name) {
      elt.second = value;
      attrs = AttributeListStorage::get(newAttrs, context);
      return;
    }

  // Otherwise, add it.
  newAttrs.push_back({name, value});
  attrs = AttributeListStorage::get(newAttrs, context);
}

/// Remove the attribute with the specified name if it exists.  The return
/// value indicates whether the attribute was present or not.
auto OperationInst::removeAttr(Identifier name) -> RemoveResult {
  auto origAttrs = getAttrs();
  for (unsigned i = 0, e = origAttrs.size(); i != e; ++i) {
    if (origAttrs[i].first == name) {
      SmallVector<NamedAttribute, 8> newAttrs;
      newAttrs.reserve(origAttrs.size() - 1);
      newAttrs.append(origAttrs.begin(), origAttrs.begin() + i);
      newAttrs.append(origAttrs.begin() + i + 1, origAttrs.end());
      attrs = AttributeListStorage::get(newAttrs, getContext());
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}

/// Attempt to constant fold this operation with the specified constant
/// operand values.  If successful, this returns false and fills in the
/// results vector.  If not, this returns true and results is unspecified.
bool OperationInst::constantFold(ArrayRef<Attribute> operands,
                                 SmallVectorImpl<Attribute> &results) const {
  if (auto *abstractOp = getAbstractOperation()) {
    // If we have a registered operation definition matching this one, use it to
    // try to constant fold the operation.
    if (!abstractOp->constantFoldHook(this, operands, results))
      return false;

    // Otherwise, fall back on the dialect hook to handle it.
    return abstractOp->dialect.constantFoldHook(this, operands, results);
  }

  // If this operation hasn't been registered or doesn't have abstract
  // operation, fall back to a dialect which matches the prefix.
  auto opName = getName().getStringRef();
  auto dialectPrefix = opName.split('.').first;
  if (auto *dialect = getContext()->getRegisteredDialect(dialectPrefix)) {
    return dialect->constantFoldHook(this, operands, results);
  }

  return true;
}

/// Attempt to fold this operation using the Op's registered foldHook.
bool OperationInst::fold(SmallVectorImpl<Value *> &results) {
  if (auto *abstractOp = getAbstractOperation()) {
    // If we have a registered operation definition matching this one, use it to
    // try to constant fold the operation.
    if (!abstractOp->foldHook(this, results))
      return false;
  }
  return true;
}

/// Emit an error with the op name prefixed, like "'dim' op " which is
/// convenient for verifiers.
bool OperationInst::emitOpError(const Twine &message) const {
  return emitError(Twine('\'') + getName().getStringRef() + "' op " + message);
}

//===----------------------------------------------------------------------===//
// Instruction Cloning
//===----------------------------------------------------------------------===//

/// Create a deep copy of this instruction, remapping any operands that use
/// values outside of the instruction using the map that is provided (leaving
/// them alone if no entry is present).  Replaces references to cloned
/// sub-instructions to the corresponding instruction that is copied, and adds
/// those mappings to the map.
Instruction *Instruction::clone(BlockAndValueMapping &mapper,
                                MLIRContext *context) const {
  SmallVector<Value *, 8> operands;
  SmallVector<Block *, 2> successors;

  auto *opInst = cast<OperationInst>(this);
  operands.reserve(getNumOperands() + opInst->getNumSuccessors());

  if (!opInst->isTerminator()) {
    // Non-terminators just add all the operands.
    for (auto *opValue : getOperands())
      operands.push_back(mapper.lookupOrDefault(const_cast<Value *>(opValue)));
  } else {
    // We add the operands separated by nullptr's for each successor.
    unsigned firstSuccOperand = opInst->getNumSuccessors()
                                    ? opInst->getSuccessorOperandIndex(0)
                                    : opInst->getNumOperands();
    auto InstOperands = opInst->getInstOperands();

    unsigned i = 0;
    for (; i != firstSuccOperand; ++i)
      operands.push_back(
          mapper.lookupOrDefault(const_cast<Value *>(InstOperands[i].get())));

    successors.reserve(opInst->getNumSuccessors());
    for (unsigned succ = 0, e = opInst->getNumSuccessors(); succ != e; ++succ) {
      successors.push_back(mapper.lookupOrDefault(
          const_cast<Block *>(opInst->getSuccessor(succ))));

      // Add sentinel to delineate successor operands.
      operands.push_back(nullptr);

      // Remap the successors operands.
      for (auto *operand : opInst->getSuccessorOperands(succ))
        operands.push_back(
            mapper.lookupOrDefault(const_cast<Value *>(operand)));
    }
  }

  SmallVector<Type, 8> resultTypes;
  resultTypes.reserve(opInst->getNumResults());
  for (auto *result : opInst->getResults())
    resultTypes.push_back(result->getType());

  unsigned numBlockLists = opInst->getNumBlockLists();
  auto *newOp = OperationInst::create(
      getLoc(), opInst->getName(), operands, resultTypes, opInst->getAttrs(),
      successors, numBlockLists, opInst->hasResizableOperandsList(), context);

  // Clone the block lists.
  for (unsigned i = 0; i != numBlockLists; ++i)
    opInst->getBlockList(i).cloneInto(&newOp->getBlockList(i), mapper, context);

  // Remember the mapping of any results.
  for (unsigned i = 0, e = opInst->getNumResults(); i != e; ++i)
    mapper.map(opInst->getResult(i), newOp->getResult(i));
  return newOp;
}

Instruction *Instruction::clone(MLIRContext *context) const {
  BlockAndValueMapping mapper;
  return clone(mapper, context);
}
