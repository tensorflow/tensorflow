//===- Instructions.h - MLIR CFG Instruction Classes ------------*- C++ -*-===//
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
// This file defines the classes for CFGFunction instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTIONS_H
#define MLIR_IR_INSTRUCTIONS_H

#include "mlir/IR/CFGValue.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class BasicBlock;
class CFGFunction;
class OperationInst;
class TerminatorInst;
} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for OperationInst
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct ilist_traits<::mlir::OperationInst> {
  using OperationInst = ::mlir::OperationInst;
  using instr_iterator = simple_ilist<OperationInst>::iterator;

  static void deleteNode(OperationInst *inst);
  void addNodeToList(OperationInst *inst);
  void removeNodeFromList(OperationInst *inst);
  void transferNodesFromList(ilist_traits<OperationInst> &otherList,
                             instr_iterator first, instr_iterator last);

private:
  mlir::BasicBlock *getContainingBlock();
};

} // end namespace llvm

namespace mlir {

/// The operand of a TerminatorInst contains a BasicBlock.
using BasicBlockOperand = IROperandImpl<BasicBlock, TerminatorInst>;

/// Instruction is the root of the operation and terminator instructions in the
/// hierarchy.
class Instruction {
public:
  enum class Kind { Operation, Branch, CondBranch, Return };

  Kind getKind() const { return kind; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const;

  /// The source location the operation was defined or derived from.
  Location *getLoc() const { return location; }

  /// Return the BasicBlock containing this instruction.
  const BasicBlock *getBlock() const { return block; }
  BasicBlock *getBlock() { return block; }

  /// Return the CFGFunction containing this instruction.
  CFGFunction *getFunction();
  const CFGFunction *getFunction() const {
    return const_cast<Instruction *>(this)->getFunction();
  }

  /// Destroy this instruction and its subclass data.
  void destroy();

  void print(raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const;

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, CFGValue *value) {
    return getInstOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<Instruction, CFGValue>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const Instruction, const CFGValue>;

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }

  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

  MutableArrayRef<InstOperand> getInstOperands();
  ArrayRef<InstOperand> getInstOperands() const {
    return const_cast<Instruction *>(this)->getInstOperands();
  }

  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return getInstOperands()[idx];
  }

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  NOTE: This may terminate
  /// the containing application, only use when the IR is in an inconsistent
  /// state.
  void emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

protected:
  Instruction(Kind kind, Location *location) : kind(kind), location(location) {
    assert(location && "location can never be null");
  }

  // Instructions are deleted through the destroy() member because this class
  // does not have a virtual destructor.  A vtable would bloat the size of
  // every instruction by a word, is not necessary given the closed nature of
  // instruction kinds.
  ~Instruction();

private:
  Kind kind;
  BasicBlock *block = nullptr;

  /// This holds information about the source location the instruction was
  /// defined or derived from.
  Location *location;

  friend struct llvm::ilist_traits<OperationInst>;
  friend class BasicBlock;
};

inline raw_ostream &operator<<(raw_ostream &os, const Instruction &inst) {
  inst.print(os);
  return os;
}

/// Operations are the main instruction kind in MLIR, which represent all of the
/// arithmetic and other basic computation.
class OperationInst final
    : public Operation,
      public Instruction,
      public llvm::ilist_node_with_parent<OperationInst, BasicBlock>,
      private llvm::TrailingObjects<OperationInst, InstOperand, InstResult> {
public:
  /// Create a new OperationInst with the specified fields.
  static OperationInst *create(Location *location, OperationName name,
                               ArrayRef<CFGValue *> operands,
                               ArrayRef<Type *> resultTypes,
                               ArrayRef<NamedAttribute> attributes,
                               MLIRContext *context);

  using Instruction::emitError;
  using Instruction::emitNote;
  using Instruction::emitWarning;
  using Instruction::getContext;
  using Instruction::getLoc;

  OperationInst *clone() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return numOperands; }

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, CFGValue *value) {
    getInstOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<OperationInst, CFGValue>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const OperationInst, const CFGValue>;

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }

  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

  ArrayRef<InstOperand> getInstOperands() const {
    return {getTrailingObjects<InstOperand>(), numOperands};
  }
  MutableArrayRef<InstOperand> getInstOperands() {
    return {getTrailingObjects<InstOperand>(), numOperands};
  }
  // Accessors to InstOperand. Without these methods invoking getInstOperand()
  // calls Instruction::getInstOperands() resulting in execution of
  // an unnecessary switch statement.
  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return getInstOperands()[idx];
  }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  unsigned getNumResults() const { return numResults; }

  CFGValue *getResult(unsigned idx) { return &getInstResult(idx); }
  const CFGValue *getResult(unsigned idx) const { return &getInstResult(idx); }

  // Support non-const result iteration.
  using result_iterator = ResultIterator<OperationInst, CFGValue>;
  result_iterator result_begin() { return result_iterator(this, 0); }
  result_iterator result_end() {
    return result_iterator(this, getNumResults());
  }
  llvm::iterator_range<result_iterator> getResults() {
    return {result_begin(), result_end()};
  }

  // Support const operand iteration.
  using const_result_iterator =
      ResultIterator<const OperationInst, const CFGValue>;
  const_result_iterator result_begin() const {
    return const_result_iterator(this, 0);
  }

  const_result_iterator result_end() const {
    return const_result_iterator(this, getNumResults());
  }

  llvm::iterator_range<const_result_iterator> getResults() const {
    return {result_begin(), result_end()};
  }

  ArrayRef<InstResult> getInstResults() const {
    return {getTrailingObjects<InstResult>(), numResults};
  }

  MutableArrayRef<InstResult> getInstResults() {
    return {getTrailingObjects<InstResult>(), numResults};
  }

  InstResult &getInstResult(unsigned idx) { return getInstResults()[idx]; }

  const InstResult &getInstResult(unsigned idx) const {
    return getInstResults()[idx];
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Unlink this instruction from its BasicBlock and delete it.
  void erase();

  /// Delete an instruction that is not linked into a block.
  void destroy();

  /// Unlink this operation instruction from its current basic block and insert
  /// it right before `existingInst` which may be in the same or another block
  /// of the same function.
  void moveBefore(OperationInst *existingInst);

  /// Unlink this operation instruction from its current basic block and insert
  /// it right before `iterator` in the specified basic block, which must be in
  /// the same function.
  void moveBefore(BasicBlock *block,
                  llvm::iplist<OperationInst>::iterator iterator);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Operation;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Instruction;
  }

private:
  const unsigned numOperands, numResults;

  OperationInst(Location *location, OperationName name, unsigned numOperands,
                unsigned numResults, ArrayRef<NamedAttribute> attributes,
                MLIRContext *context);
  ~OperationInst();

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperationInst, InstOperand, InstResult>;
  size_t numTrailingObjects(OverloadToken<InstOperand>) const {
    return numOperands;
  }
  size_t numTrailingObjects(OverloadToken<InstResult>) const {
    return numResults;
  }
};

/// Terminator instructions are the last part of a basic block, used to
/// represent control flow and returns.
class TerminatorInst : public Instruction {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() != Kind::Operation;
  }

  /// Remove this terminator from its BasicBlock and delete it.
  void erase();

  /// Return the list of BasicBlockOperand operands of this terminator that
  /// this terminator holds.
  MutableArrayRef<BasicBlockOperand> getBasicBlockOperands();

  ArrayRef<BasicBlockOperand> getBasicBlockOperands() const {
    return const_cast<TerminatorInst *>(this)->getBasicBlockOperands();
  }

  unsigned getNumSuccessors() const { return getBasicBlockOperands().size(); }

  const BasicBlock *getSuccessor(unsigned i) const {
    return getBasicBlockOperands()[i].get();
  }

  BasicBlock *getSuccessor(unsigned i) {
    return getBasicBlockOperands()[i].get();
  }

protected:
  TerminatorInst(Kind kind, Location *location) : Instruction(kind, location) {}
  ~TerminatorInst() {}
};

/// The 'br' instruction is an unconditional from one basic block to another,
/// and may pass basic block arguments to the successor.
class BranchInst : public TerminatorInst {
public:
  static BranchInst *create(Location *location, BasicBlock *dest,
                            ArrayRef<CFGValue *> operands = {}) {
    return new BranchInst(location, dest, operands);
  }
  ~BranchInst() {}

  /// Return the block this branch jumps to.
  BasicBlock *getDest() const { return dest.get(); }
  void setDest(BasicBlock *block);

  unsigned getNumOperands() const { return operands.size(); }

  ArrayRef<InstOperand> getInstOperands() const { return operands; }
  MutableArrayRef<InstOperand> getInstOperands() { return operands; }

  /// Add one value to the operand list.
  void addOperand(CFGValue *value);

  /// Add a list of values to the operand list.
  void addOperands(ArrayRef<CFGValue *> values);

  /// Erase a specific argument from the arg list.
  // TODO: void eraseArgument(int Index);

  MutableArrayRef<BasicBlockOperand> getBasicBlockOperands() { return dest; }
  ArrayRef<BasicBlockOperand> getBasicBlockOperands() const { return dest; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Branch;
  }

private:
  explicit BranchInst(Location *location, BasicBlock *dest,
                      ArrayRef<CFGValue *> operands);
  BasicBlockOperand dest;
  std::vector<InstOperand> operands;
};

/// The 'cond_br' instruction is a conditional branch based on a boolean
/// condition to one of two possible successors. It may pass arguments to each
/// successor.
class CondBranchInst : public TerminatorInst {
  // These are the indices into the dests list.
  enum { trueIndex = 0, falseIndex = 1 };

public:
  static CondBranchInst *create(Location *location, CFGValue *condition,
                                BasicBlock *trueDest, BasicBlock *falseDest) {
    return new CondBranchInst(location, condition, trueDest, falseDest);
  }
  ~CondBranchInst() {}

  /// Return the i1 condition.
  CFGValue *getCondition() { return condition; }
  const CFGValue *getCondition() const { return condition; }

  /// Return the destination if the condition is true.
  BasicBlock *getTrueDest() const { return dests[trueIndex].get(); }

  /// Return the destination if the condition is false.
  BasicBlock *getFalseDest() const { return dests[falseIndex].get(); }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<CondBranchInst, CFGValue>;
  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const CondBranchInst, const CFGValue>;

  ArrayRef<InstOperand> getInstOperands() const { return operands; }
  MutableArrayRef<InstOperand> getInstOperands() { return operands; }

  unsigned getNumOperands() const { return operands.size(); }

  //
  // Accessors for operands to the 'true' destination
  //

  CFGValue *getTrueOperand(unsigned idx) {
    return getTrueInstOperand(idx).get();
  }
  const CFGValue *getTrueOperand(unsigned idx) const {
    return getTrueInstOperand(idx).get();
  }
  void setTrueOperand(unsigned idx, CFGValue *value) {
    return getTrueInstOperand(idx).set(value);
  }

  operand_iterator true_operand_begin() { return operand_iterator(this, 0); }
  operand_iterator true_operand_end() {
    return operand_iterator(this, getNumTrueOperands());
  }
  llvm::iterator_range<operand_iterator> getTrueOperands() {
    return {true_operand_begin(), true_operand_end()};
  }

  const_operand_iterator true_operand_begin() const {
    return const_operand_iterator(this, 0);
  }
  const_operand_iterator true_operand_end() const {
    return const_operand_iterator(this, getNumTrueOperands());
  }
  llvm::iterator_range<const_operand_iterator> getTrueOperands() const {
    return {true_operand_begin(), true_operand_end()};
  }

  ArrayRef<InstOperand> getTrueInstOperands() const {
    return const_cast<CondBranchInst *>(this)->getTrueInstOperands();
  }
  MutableArrayRef<InstOperand> getTrueInstOperands() {
    return {operands.data(), operands.data() + getNumTrueOperands()};
  }

  InstOperand &getTrueInstOperand(unsigned idx) { return operands[idx]; }
  const InstOperand &getTrueInstOperand(unsigned idx) const {
    return operands[idx];
  }
  unsigned getNumTrueOperands() const { return numTrueOperands; }

  /// Add one value to the true operand list.
  void addTrueOperand(CFGValue *value);

  /// Add a list of values to the operand list.
  void addTrueOperands(ArrayRef<CFGValue *> values);

  //
  // Accessors for operands to the 'false' destination
  //

  CFGValue *getFalseOperand(unsigned idx) {
    return getFalseInstOperand(idx).get();
  }
  const CFGValue *getFalseOperand(unsigned idx) const {
    return getFalseInstOperand(idx).get();
  }
  void setFalseOperand(unsigned idx, CFGValue *value) {
    return getFalseInstOperand(idx).set(value);
  }

  operand_iterator false_operand_begin() {
    return operand_iterator(this, getNumTrueOperands());
  }
  operand_iterator false_operand_end() {
    return operand_iterator(this, getNumOperands());
  }
  llvm::iterator_range<operand_iterator> getFalseOperands() {
    return {false_operand_begin(), false_operand_end()};
  }

  const_operand_iterator false_operand_begin() const {
    return const_operand_iterator(this, getNumTrueOperands());
  }
  const_operand_iterator false_operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }
  llvm::iterator_range<const_operand_iterator> getFalseOperands() const {
    return {false_operand_begin(), false_operand_end()};
  }

  ArrayRef<InstOperand> getFalseInstOperands() const {
    return const_cast<CondBranchInst *>(this)->getFalseInstOperands();
  }
  MutableArrayRef<InstOperand> getFalseInstOperands() {
    return {operands.data() + getNumTrueOperands(),
            operands.data() + getNumOperands()};
  }

  InstOperand &getFalseInstOperand(unsigned idx) {
    return operands[idx + getNumTrueOperands()];
  }
  const InstOperand &getFalseInstOperand(unsigned idx) const {
    return operands[idx + getNumTrueOperands()];
  }
  unsigned getNumFalseOperands() const {
    return operands.size() - numTrueOperands;
  }

  /// Add one value to the false operand list.
  void addFalseOperand(CFGValue *value);

  /// Add a list of values to the operand list.
  void addFalseOperands(ArrayRef<CFGValue *> values);

  MutableArrayRef<BasicBlockOperand> getBasicBlockOperands() { return dests; }
  ArrayRef<BasicBlockOperand> getBasicBlockOperands() const { return dests; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::CondBranch;
  }

private:
  CondBranchInst(Location *location, CFGValue *condition, BasicBlock *trueDest,
                 BasicBlock *falseDest);

  CFGValue *condition;
  BasicBlockOperand dests[2]; // 0 is the true dest, 1 is the false dest.

  // Operand list. The true operands are stored first, followed by the false
  // operands.
  std::vector<InstOperand> operands;
  unsigned numTrueOperands;
};

/// The 'return' instruction represents the end of control flow within the
/// current function, and can return zero or more results.  The result list is
/// required to align with the result list of the containing function's type.
class ReturnInst final
    : public TerminatorInst,
      private llvm::TrailingObjects<ReturnInst, InstOperand> {
public:
  /// Create a new ReturnInst with the specific fields.
  static ReturnInst *create(Location *location, ArrayRef<CFGValue *> operands);

  unsigned getNumOperands() const { return numOperands; }

  ArrayRef<InstOperand> getInstOperands() const {
    return {getTrailingObjects<InstOperand>(), numOperands};
  }
  MutableArrayRef<InstOperand> getInstOperands() {
    return {getTrailingObjects<InstOperand>(), numOperands};
  }

  void destroy();

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Return;
  }

private:
  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<ReturnInst, InstOperand>;
  size_t numTrailingObjects(OverloadToken<InstOperand>) const {
    return numOperands;
  }

  ReturnInst(Location *location, unsigned numOperands);
  ~ReturnInst();

  unsigned numOperands;
};

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTIONS_H
