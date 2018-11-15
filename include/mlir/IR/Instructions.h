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

/// The operand of a Terminator contains a BasicBlock.
using BasicBlockOperand = IROperandImpl<BasicBlock, Instruction>;

/// Instruction is the root of the operation and terminator instructions in the
/// hierarchy.
class Instruction : public IROperandOwner {
public:
  enum class Kind {
    Operation = (int)IROperandOwner::Kind::OperationInst,
  };

  Kind getKind() const { return (Kind)IROperandOwner::getKind(); }

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

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() >= IROperandOwner::Kind::INST_FIRST;
  }

protected:
  Instruction(Kind kind, Location location)
      : IROperandOwner((IROperandOwner::Kind)kind, location) {}

  // Instructions are deleted through the destroy() member because this class
  // does not have a virtual destructor.  A vtable would bloat the size of
  // every instruction by a word, is not necessary given the closed nature of
  // instruction kinds.
  ~Instruction();

private:
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
//
// The trailing objects of an operation instruction are layed out as follows:
//   - InstResult        : The results of the instruction.
//   - BasicBlockOperand : Use-list of successor blocks if this is a terminator.
//   - unsigned          : Count of operands held for each of the successors.
//
// Note: For Terminators, we rely on the assumption that all non successor
// operands are placed at the beginning of the operands list.
class OperationInst final
    : public Operation,
      public Instruction,
      public llvm::ilist_node_with_parent<OperationInst, BasicBlock>,
      private llvm::TrailingObjects<OperationInst, InstResult,
                                    BasicBlockOperand, unsigned> {
public:
  /// Create a new OperationInst with the specified fields.
  static OperationInst *
  create(Location location, OperationName name, ArrayRef<CFGValue *> operands,
         ArrayRef<Type> resultTypes, ArrayRef<NamedAttribute> attributes,
         ArrayRef<BasicBlock *> successors, MLIRContext *context);

  using Instruction::dump;
  using Instruction::emitError;
  using Instruction::emitNote;
  using Instruction::emitWarning;
  using Instruction::getContext;
  using Instruction::getLoc;
  using Instruction::print;

  OperationInst *clone() const;

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  unsigned getNumOperands() const { return operands.size(); }

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

  MutableArrayRef<InstOperand> getInstOperands() { return operands; }
  ArrayRef<InstOperand> getInstOperands() const { return operands; }

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
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BasicBlockOperand> getBasicBlockOperands() {
    assert(isTerminator() && "Only terminators have a block operands list.");
    return {getTrailingObjects<BasicBlockOperand>(), numSuccs};
  }
  ArrayRef<BasicBlockOperand> getBasicBlockOperands() const {
    return const_cast<OperationInst *>(this)->getBasicBlockOperands();
  }

  MutableArrayRef<InstOperand> getSuccessorInstOperands(unsigned index) {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());
    unsigned succOpIndex = getSuccessorOperandIndex(index);
    auto *operandBegin = operands.data() + succOpIndex;
    return {operandBegin, getNumSuccessorOperands(index)};
  }
  ArrayRef<InstOperand> getSuccessorInstOperands(unsigned index) const {
    return const_cast<OperationInst *>(this)->getSuccessorInstOperands(index);
  }

  unsigned getNumSuccessors() const { return getBasicBlockOperands().size(); }
  unsigned getNumSuccessorOperands(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());
    return getTrailingObjects<unsigned>()[index];
  }

  BasicBlock *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBasicBlockOperands()[index].get();
  }
  BasicBlock *getSuccessor(unsigned index) const {
    return const_cast<OperationInst *>(this)->getSuccessor(index);
  }
  void setSuccessor(BasicBlock *block, unsigned index) {
    assert(index < getNumSuccessors());
    getBasicBlockOperands()[index].set(block);
  }

  /// Add one value to the operand list of the successor at the provided index.
  void addSuccessorOperand(unsigned index, CFGValue *value);

  /// Add a list of values to the operand list of the successor at the provided
  /// index.
  void addSuccessorOperands(unsigned index, ArrayRef<CFGValue *> values);

  /// Erase a specific argument from the arg list.
  // TODO: void eraseSuccessorOperand(unsigned index, unsigned argIndex);

  /// Get the index of the first operand of the successor at the provided
  /// index.
  unsigned getSuccessorOperandIndex(unsigned index) const {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());

    // Count the number of operands for each of the successors after, and
    // including, the one at 'index'. This is based upon the assumption that all
    // non successor operands are placed at the beginning of the operand list.
    auto *successorOpCountBegin = getTrailingObjects<unsigned>();
    unsigned postSuccessorOpCount =
        std::accumulate(successorOpCountBegin + index,
                        successorOpCountBegin + getNumSuccessors(), 0);
    return getNumOperands() - postSuccessorOpCount;
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
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::OperationInst;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Instruction;
  }

private:
  const unsigned numResults, numSuccs;
  std::vector<InstOperand> operands;

  OperationInst(Location location, OperationName name, unsigned numResults,
                unsigned numSuccessors, ArrayRef<NamedAttribute> attributes,
                MLIRContext *context);
  ~OperationInst();

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperationInst, InstResult, BasicBlockOperand,
                               unsigned>;
  size_t numTrailingObjects(OverloadToken<InstResult>) const {
    return numResults;
  }
  size_t numTrailingObjects(OverloadToken<BasicBlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const { return numSuccs; }
};

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTIONS_H
