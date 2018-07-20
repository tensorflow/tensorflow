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
  enum class Kind {
    Operation,
    Branch,
    Return
  };

  Kind getKind() const { return kind; }

  /// Return the BasicBlock containing this instruction.
  BasicBlock *getBlock() const {
    return block;
  }

  /// Return the CFGFunction containing this instruction.
  CFGFunction *getFunction() const;

  /// Destroy this instruction and its subclass data.
  static void destroy(Instruction *inst);

  void print(raw_ostream &os) const;
  void dump() const;

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
  ~OperationInst();

  unsigned getNumOperands() const { return numOperands; }

  // TODO: Add a getOperands() custom sequence that provides a value projection
  // of the operand list.
  CFGValue *getOperand(unsigned idx) { return getInstOperand(idx).get(); }
  const CFGValue *getOperand(unsigned idx) const {
    return getInstOperand(idx).get();
  }

  unsigned getNumResults() const { return numResults; }

  // TODO: Add a getResults() custom sequence that provides a value projection
  // of the result list.
  CFGValue *getResult(unsigned idx) { return &getInstResult(idx); }
  const CFGValue *getResult(unsigned idx) const { return &getInstResult(idx); }

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

  /// Unlink this instruction from its BasicBlock and delete it.
  void eraseFromBlock();

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
  explicit BranchInst(BasicBlock *dest)
    : TerminatorInst(Kind::Branch), dest(dest) {
  }
  ~BranchInst() {}

  /// Return the block this branch jumps to.
  BasicBlock *getDest() const {
    return dest;
  }

  // TODO: need to take operands to specify BB arguments

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Branch;
  }

private:
  BasicBlock *dest;
};


/// The 'return' instruction represents the end of control flow within the
/// current function, and can return zero or more results.  The result list is
/// required to align with the result list of the containing function's type.
class ReturnInst : public TerminatorInst {
public:
  explicit ReturnInst() : TerminatorInst(Kind::Return) {}
  ~ReturnInst() {}

  // TODO: Needs to take an operand list.

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Return;
  }
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

  static void deleteNode(OperationInst *inst) {
    OperationInst::destroy(inst);
  }

  void addNodeToList(OperationInst *inst);
  void removeNodeFromList(OperationInst *inst);
  void transferNodesFromList(ilist_traits<OperationInst> &otherList,
                             instr_iterator first, instr_iterator last);
private:
  mlir::BasicBlock *getContainingBlock();
};

} // end namespace llvm

#endif  // MLIR_IR_INSTRUCTIONS_H
