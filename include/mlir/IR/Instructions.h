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

#include "mlir/Support/LLVM.h"
#include "mlir/IR/Identifier.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

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

  /// Destroy this instruction or one of its subclasses
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
class OperationInst
  : public Operation, public Instruction,
    public llvm::ilist_node_with_parent<OperationInst, BasicBlock> {
public:
  explicit OperationInst(Identifier name, ArrayRef<NamedAttribute> attrs,
                         MLIRContext *context)
      : Operation(name, attrs, context), Instruction(Kind::Operation) {}
  ~OperationInst() {}

  /// Unlink this instruction from its BasicBlock and delete it.
  void eraseFromBlock();

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Instruction *inst) {
    return inst->getKind() == Kind::Operation;
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

  // TODO: need to take BB arguments.

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
