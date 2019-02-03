//===- InstVisitor.h - MLIR Instruction Visitor Class -----------*- C++ -*-===//
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
// This file defines the base classes for Function's instruction visitors and
// walkers. A visit is a O(1) operation that visits just the node in question. A
// walk visits the node it's called on as well as the node's descendants.
//
// Instruction visitors/walkers are used when you want to perform different
// actions for different kinds of instructions without having to use lots of
// casts and a big switch instruction.
//
// To define your own visitor/walker, inherit from these classes, specifying
// your new type for the 'SubClass' template parameter, and "override" visitXXX
// functions in your class. This class is defined in terms of statically
// resolved overloading, not virtual functions.
//
// For example, here is a walker that counts the number of for loops in an
// Function.
//
//  /// Declare the class.  Note that we derive from InstWalker instantiated
//  /// with _our new subclasses_ type.
//  struct LoopCounter : public InstWalker<LoopCounter> {
//    unsigned numLoops;
//    LoopCounter() : numLoops(0) {}
//    void visitForInst(ForInst &fs) { ++numLoops; }
//  };
//
//  And this class would be used like this:
//    LoopCounter lc;
//    lc.walk(function);
//    numLoops = lc.numLoops;
//
// There  are 'visit' methods for OperationInst, ForInst, and
// Function, which recursively process all contained instructions.
//
// Note that if you don't implement visitXXX for some instruction type,
// the visitXXX method for Instruction superclass will be invoked.
//
// The optional second template argument specifies the type that instruction
// visitation functions should return. If you specify this, you *MUST* provide
// an implementation of every visit<#Instruction>(InstType *).
//
// Note that these classes are specifically designed as a template to avoid
// virtual function call overhead.  Defining and using a InstVisitor is just
// as efficient as having your own switch instruction over the instruction
// opcode.

//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTVISITOR_H
#define MLIR_IR_INSTVISITOR_H

#include "mlir/IR/Function.h"
#include "mlir/IR/Instruction.h"

namespace mlir {

/// Base class for instruction visitors.
template <typename SubClass, typename RetTy = void> class InstVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the InstVisitor that you
  // use to visit instructions.

public:
  // Function to visit a instruction.
  RetTy visit(Instruction *s) {
    static_assert(std::is_base_of<InstVisitor, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (s->getKind()) {
    case Instruction::Kind::OperationInst:
      return static_cast<SubClass *>(this)->visitOperationInst(
          cast<OperationInst>(s));
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // When visiting a for inst, if inst, or an operation inst directly, these
  // methods get called to indicate when transitioning into a new unit.
  void visitOperationInst(OperationInst *opInst) {}
};

/// Base class for instruction walkers. A walker can traverse depth first in
/// pre-order or post order. The walk methods without a suffix do a pre-order
/// traversal while those that traverse in post order have a PostOrder suffix.
template <typename SubClass, typename RetTy = void> class InstWalker {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the InstWalker used to
  // walk instructions.

public:
  // Generic walk method - allow walk to all instructions in a range.
  template <class Iterator> void walk(Iterator Start, Iterator End) {
    while (Start != End) {
      walk(&(*Start++));
    }
  }
  template <class Iterator> void walkPostOrder(Iterator Start, Iterator End) {
    while (Start != End) {
      walkPostOrder(&(*Start++));
    }
  }

  // Define walkers for Function and all Function instruction kinds.
  void walk(Function *f) {
    for (auto &block : *f)
      static_cast<SubClass *>(this)->walk(block.begin(), block.end());
  }

  void walkPostOrder(Function *f) {
    for (auto it = f->rbegin(), e = f->rend(); it != e; ++it)
      static_cast<SubClass *>(this)->walkPostOrder(it->begin(), it->end());
  }

  void walkOpInst(OperationInst *opInst) {
    static_cast<SubClass *>(this)->visitOperationInst(opInst);
    for (auto &blockList : opInst->getBlockLists())
      for (auto &block : blockList)
        static_cast<SubClass *>(this)->walk(block.begin(), block.end());
  }

  void walkOpInstPostOrder(OperationInst *opInst) {
    for (auto &blockList : opInst->getBlockLists())
      for (auto &block : blockList)
        static_cast<SubClass *>(this)->walkPostOrder(block.begin(),
                                                     block.end());
    static_cast<SubClass *>(this)->visitOperationInst(opInst);
  }

  // Function to walk a instruction.
  RetTy walk(Instruction *s) {
    static_assert(std::is_base_of<InstWalker, SubClass>::value,
                  "Must pass the derived type to this template!");

    static_cast<SubClass *>(this)->visitInstruction(s);

    switch (s->getKind()) {
    case Instruction::Kind::OperationInst:
      return static_cast<SubClass *>(this)->walkOpInst(cast<OperationInst>(s));
    }
  }

  // Function to walk a instruction in post order DFS.
  RetTy walkPostOrder(Instruction *s) {
    static_assert(std::is_base_of<InstWalker, SubClass>::value,
                  "Must pass the derived type to this template!");
    static_cast<SubClass *>(this)->visitInstruction(s);

    switch (s->getKind()) {
    case Instruction::Kind::OperationInst:
      return static_cast<SubClass *>(this)->walkOpInstPostOrder(
          cast<OperationInst>(s));
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.

  // When visiting a specific inst directly during a walk, these methods get
  // called. These are typically O(1) complexity and shouldn't be recursively
  // processing their descendants in some way. When using RetTy, all of these
  // need to be overridden.
  void visitOperationInst(OperationInst *opInst) {}
  void visitInstruction(Instruction *inst) {}
};

} // end namespace mlir

#endif // MLIR_IR_INSTVISITOR_H
