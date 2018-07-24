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
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class OperationInst;
class BasicBlock;
class CFGFunction;

/// Instruction is the root of the operation and terminator instructions in the
/// hierarchy.
class Instruction {
public:
  enum class Kind { Operation, Branch, CondBranch, Return };

  Kind getKind() const { return kind; }

  /// Return the BasicBlock containing this instruction.
  BasicBlock *getBlock() const { return block; }

  /// Return the CFGFunction containing this instruction.
  CFGFunction *getFunction() const;

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

protected:
  Instruction(Kind kind) : kind(kind) {}

  // Instructions are deleted through the destroy() member because this class
  // does not have a virtual destructor.  A vtable would bloat the size of
  // every instruction by a word, is not necessary given the closed nature of
  // instruction kinds.
  ~Instruction();

private:
  Kind kind;
  BasicBlock *block = nullptr;

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
  /// Create a new OperationInst with the specific fields.
  static OperationInst *create(Identifier name, ArrayRef<CFGValue *> operands,
                               ArrayRef<Type *> resultTypes,
                               ArrayRef<NamedAttribute> attributes,
                               MLIRContext *context);

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return numOperands; }

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, CFGValue *value) {
    return getInstOperand(idx).set(value);
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
  typedef ResultIterator<OperationInst, CFGValue> result_iterator;
  result_iterator result_begin() { return result_iterator(this, 0); }
  result_iterator result_end() {
    return result_iterator(this, getNumResults());
  }
  llvm::iterator_range<result_iterator> getResults() {
    return {result_begin(), result_end()};
  }

  // Support const operand iteration.
  typedef ResultIterator<const OperationInst, const CFGValue>
      const_result_iterator;
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
  void eraseFromBlock();

  void destroy();

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Operation;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Instruction;
  }

private:
  const unsigned numOperands, numResults;

  OperationInst(Identifier name, unsigned numOperands, unsigned numResults,
                ArrayRef<NamedAttribute> attributes, MLIRContext *context);
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
  void eraseFromBlock();

protected:
  TerminatorInst(Kind kind) : Instruction(kind) {}
  ~TerminatorInst() {}
};

/// The 'br' instruction is an unconditional from one basic block to another,
/// and may pass basic block arguments to the successor.
class BranchInst : public TerminatorInst {
public:
  static BranchInst *create(BasicBlock *dest) { return new BranchInst(dest); }
  ~BranchInst() {}

  /// Return the block this branch jumps to.
  BasicBlock *getDest() const { return dest; }

  unsigned getNumOperands() const { return operands.size(); }

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, CFGValue *value) {
    return getInstOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<BranchInst, CFGValue>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  typedef OperandIterator<const BranchInst, const CFGValue>
      const_operand_iterator;

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }

  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

  ArrayRef<InstOperand> getInstOperands() const { return operands; }
  MutableArrayRef<InstOperand> getInstOperands() { return operands; }

  InstOperand &getInstOperand(unsigned idx) { return operands[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return operands[idx];
  }

  /// Add one value to the operand list.
  void addOperand(CFGValue *value);

  /// Add a list of values to the operand list.
  void addOperands(ArrayRef<CFGValue *> values);

  /// Erase a specific argument from the arg list.
  // TODO: void eraseArgument(int Index);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Branch;
  }

private:
  explicit BranchInst(BasicBlock *dest)
      : TerminatorInst(Kind::Branch), dest(dest) {}

  BasicBlock *dest;
  std::vector<InstOperand> operands;
};

/// The 'cond_br' instruction is a conditional branch based on a boolean
/// condition to one of two possible successors. It may pass arguments to each
/// successor.
class CondBranchInst : public TerminatorInst {
public:
  static CondBranchInst *create(CFGValue *condition, BasicBlock *trueDest,
                                BasicBlock *falseDest) {
    return new CondBranchInst(condition, trueDest, falseDest);
  }
  ~CondBranchInst() {}

  /// Return the i1 condition.
  CFGValue *getCondition() { return condition; }
  const CFGValue *getCondition() const { return condition; }

  /// Return the destination if the condition is true.
  BasicBlock *getTrueDest() const { return trueDest; }

  /// Return the destination if the condition is false.
  BasicBlock *getFalseDest() const { return falseDest; }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<CondBranchInst, CFGValue>;
  // Support const operand iteration.
  typedef OperandIterator<const CondBranchInst, const CFGValue>
      const_operand_iterator;

  //
  // Accessors for the entire operand list. This includes operands to both true
  // and false blocks.
  //

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }
  void setOperand(unsigned idx, CFGValue *value) {
    return getInstOperand(idx).set(value);
  }

  operand_iterator operand_begin() { return operand_iterator(this, 0); }
  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }
  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  const_operand_iterator operand_begin() const {
    return const_operand_iterator(this, 0);
  }
  const_operand_iterator operand_end() const {
    return const_operand_iterator(this, getNumOperands());
  }
  llvm::iterator_range<const_operand_iterator> getOperands() const {
    return {operand_begin(), operand_end()};
  }

  ArrayRef<InstOperand> getInstOperands() const { return operands; }
  MutableArrayRef<InstOperand> getInstOperands() { return operands; }

  InstOperand &getInstOperand(unsigned idx) { return operands[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return operands[idx];
  }
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
    return {&operands[0], &operands[0] + getNumTrueOperands()};
  }
  MutableArrayRef<InstOperand> getTrueInstOperands() {
    return {&operands[0], &operands[0] + getNumTrueOperands()};
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
    return {&operands[0] + getNumTrueOperands(),
            &operands[0] + getNumOperands()};
  }
  MutableArrayRef<InstOperand> getFalseInstOperands() {
    return {&operands[0] + getNumTrueOperands(),
            &operands[0] + getNumOperands()};
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

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::CondBranch;
  }

private:
  explicit CondBranchInst(CFGValue *condition, BasicBlock *trueDest,
                          BasicBlock *falseDest)
      : TerminatorInst(Kind::CondBranch), condition(condition),
        trueDest(trueDest), falseDest(falseDest), numTrueOperands(0) {}

  CFGValue *condition;
  BasicBlock *trueDest;
  BasicBlock *falseDest;
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
  /// Create a new OperationInst with the specific fields.
  static ReturnInst *create(ArrayRef<CFGValue *> operands);

  unsigned getNumOperands() const { return numOperands; }

  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }

  void setOperand(unsigned idx, CFGValue *value) {
    return getInstOperand(idx).set(value);
  }

  // Support non-const operand iteration.
  using operand_iterator = OperandIterator<ReturnInst, CFGValue>;

  operand_iterator operand_begin() { return operand_iterator(this, 0); }

  operand_iterator operand_end() {
    return operand_iterator(this, getNumOperands());
  }

  llvm::iterator_range<operand_iterator> getOperands() {
    return {operand_begin(), operand_end()};
  }

  // Support const operand iteration.
  typedef OperandIterator<const ReturnInst, const CFGValue>
      const_operand_iterator;

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

  InstOperand &getInstOperand(unsigned idx) { return getInstOperands()[idx]; }
  const InstOperand &getInstOperand(unsigned idx) const {
    return getInstOperands()[idx];
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

  explicit ReturnInst(unsigned numOperands)
      : TerminatorInst(Kind::Return), numOperands(numOperands) {}
  ~ReturnInst();

  unsigned numOperands;
};

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for OperationInst
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::OperationInst> {
  using OperationInst = ::mlir::OperationInst;
  using instr_iterator = simple_ilist<OperationInst>::iterator;

  static void deleteNode(OperationInst *inst) { inst->destroy(); }

  void addNodeToList(OperationInst *inst);
  void removeNodeFromList(OperationInst *inst);
  void transferNodesFromList(ilist_traits<OperationInst> &otherList,
                             instr_iterator first, instr_iterator last);

private:
  mlir::BasicBlock *getContainingBlock();
};

} // end namespace llvm

#endif // MLIR_IR_INSTRUCTIONS_H
