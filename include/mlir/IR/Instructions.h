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
} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Instruction
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct ilist_traits<::mlir::Instruction> {
  using Instruction = ::mlir::Instruction;
  using instr_iterator = simple_ilist<Instruction>::iterator;

  static void deleteNode(Instruction *inst);
  void addNodeToList(Instruction *inst);
  void removeNodeFromList(Instruction *inst);
  void transferNodesFromList(ilist_traits<Instruction> &otherList,
                             instr_iterator first, instr_iterator last);

private:
  mlir::BasicBlock *getContainingBlock();
};

} // end namespace llvm

namespace mlir {

/// The operand of a Terminator contains a BasicBlock.
using BasicBlockOperand = IROperandImpl<BasicBlock, Instruction>;

// The trailing objects of an instruction are layed out as follows:
//   - InstResult        : The results of the instruction.
//   - BasicBlockOperand : Use-list of successor blocks if this is a terminator.
//   - unsigned          : Count of operands held for each of the successors.
//
// Note: For Terminators, we rely on the assumption that all non successor
// operands are placed at the beginning of the operands list.
class Instruction final
    : public Operation,
      public IROperandOwner,
      public llvm::ilist_node_with_parent<Instruction, BasicBlock>,
      private llvm::TrailingObjects<Instruction, InstResult, BasicBlockOperand,
                                    unsigned> {
public:
  using IROperandOwner::getContext;
  using IROperandOwner::getLoc;
  using IROperandOwner::setLoc;

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

  MutableArrayRef<InstOperand> getInstOperands() { return operands; }
  ArrayRef<InstOperand> getInstOperands() const { return operands; }

  // Accessors to InstOperand.
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
  using result_iterator = ResultIterator<Instruction, CFGValue>;
  result_iterator result_begin() { return result_iterator(this, 0); }
  result_iterator result_end() {
    return result_iterator(this, getNumResults());
  }
  llvm::iterator_range<result_iterator> getResults() {
    return {result_begin(), result_end()};
  }

  // Support const result iteration.
  using const_result_iterator =
      ResultIterator<const Instruction, const CFGValue>;
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

  // Support result type iteration.
  using result_type_iterator =
      ResultTypeIterator<const Instruction, const CFGValue>;
  result_type_iterator result_type_begin() const {
    return result_type_iterator(this, 0);
  }

  result_type_iterator result_type_end() const {
    return result_type_iterator(this, getNumResults());
  }

  llvm::iterator_range<result_type_iterator> getResultTypes() const {
    return {result_type_begin(), result_type_end()};
  }

  //===--------------------------------------------------------------------===//
  // Terminators
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BasicBlockOperand> getBasicBlockOperands() {
    assert(isTerminator() && "Only terminators have a block operands list.");
    return {getTrailingObjects<BasicBlockOperand>(), numSuccs};
  }
  ArrayRef<BasicBlockOperand> getBasicBlockOperands() const {
    return const_cast<Instruction *>(this)->getBasicBlockOperands();
  }

  MutableArrayRef<InstOperand> getSuccessorInstOperands(unsigned index) {
    assert(isTerminator() && "Only terminators have successors.");
    assert(index < getNumSuccessors());
    unsigned succOpIndex = getSuccessorOperandIndex(index);
    auto *operandBegin = operands.data() + succOpIndex;
    return {operandBegin, getNumSuccessorOperands(index)};
  }
  ArrayRef<InstOperand> getSuccessorInstOperands(unsigned index) const {
    return const_cast<Instruction *>(this)->getSuccessorInstOperands(index);
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
  const BasicBlock *getSuccessor(unsigned index) const {
    return const_cast<Instruction *>(this)->getSuccessor(index);
  }
  void setSuccessor(BasicBlock *block, unsigned index);

  /// Add one value to the operand list of the successor at the provided index.
  void addSuccessorOperand(unsigned index, CFGValue *value);

  /// Add a list of values to the operand list of the successor at the provided
  /// index.
  void addSuccessorOperands(unsigned index, ArrayRef<CFGValue *> values);

  /// Erase a specific operand from the operand list of the successor at
  /// 'index'.
  void eraseSuccessorOperand(unsigned succIndex, unsigned opIndex) {
    assert(succIndex < getNumSuccessors());
    assert(opIndex < getNumSuccessorOperands(succIndex));
    eraseOperand(getSuccessorOperandIndex(succIndex) + opIndex);
    --getTrailingObjects<unsigned>()[succIndex];
  }

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

  /// Create a new Instruction with the specified fields.
  static Instruction *
  create(Location location, OperationName name, ArrayRef<CFGValue *> operands,
         ArrayRef<Type> resultTypes, ArrayRef<NamedAttribute> attributes,
         ArrayRef<BasicBlock *> successors, MLIRContext *context);

  Instruction *clone() const;

  /// Return the BasicBlock containing this instruction.
  const BasicBlock *getBlock() const { return block; }
  BasicBlock *getBlock() { return block; }

  /// Return the CFGFunction containing this instruction.
  CFGFunction *getFunction();
  const CFGFunction *getFunction() const {
    return const_cast<Instruction *>(this)->getFunction();
  }

  void print(raw_ostream &os) const;
  void dump() const;

  /// Unlink this instruction from its BasicBlock and delete it.
  void erase();

  /// Destroy this instruction and its subclass data.
  void destroy();

  /// Unlink this instruction from its current basic block and insert
  /// it right before `existingInst` which may be in the same or another block
  /// of the same function.
  void moveBefore(Instruction *existingInst);

  /// Unlink this instruction from its current basic block and insert
  /// it right before `iterator` in the specified basic block, which must be in
  /// the same function.
  void moveBefore(BasicBlock *block,
                  llvm::iplist<Instruction>::iterator iterator);

  /// This drops all operand uses from this instruction, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const IROperandOwner *ptr) {
    return ptr->getKind() == IROperandOwner::Kind::Instruction;
  }
  static bool classof(const Operation *op) {
    return op->getOperationKind() == OperationKind::Instruction;
  }

private:
  const unsigned numResults, numSuccs;
  std::vector<InstOperand> operands;
  BasicBlock *block = nullptr;

  Instruction(Location location, OperationName name, unsigned numResults,
              unsigned numSuccessors, ArrayRef<NamedAttribute> attributes,
              MLIRContext *context);

  // Instructions are deleted through the destroy() member because this class
  // does not have a virtual destructor.
  ~Instruction();

  /// Erase the operand at 'index'.
  void eraseOperand(unsigned index);

  friend struct llvm::ilist_traits<Instruction>;
  friend class BasicBlock;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Instruction, InstResult, BasicBlockOperand,
                               unsigned>;
  size_t numTrailingObjects(OverloadToken<InstResult>) const {
    return numResults;
  }
  size_t numTrailingObjects(OverloadToken<BasicBlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<unsigned>) const { return numSuccs; }
};

inline raw_ostream &operator<<(raw_ostream &os, const Instruction &inst) {
  inst.print(os);
  return os;
}

} // end namespace mlir

#endif // MLIR_IR_INSTRUCTIONS_H
