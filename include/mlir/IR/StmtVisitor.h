//===- StmtVisitor.h - MLIR Instruction Visitor Class -----------*- C++ -*-===//
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
// This file defines the base classes for MLFunction's statement visitors and
// walkers. A visit is a O(1) operation that visits just the node in question. A
// walk visits the node it's called on as well as the node's descendants.
//
// Statement visitors/walkers are used when you want to perform different
// actions for different kinds of statements without having to use lots of casts
// and a big switch statement.
//
// To define your own visitor/walker, inherit from these classes, specifying
// your new type for the 'SubClass' template parameter, and "override" visitXXX
// functions in your class. This class is defined in terms of statically
// resolved overloading, not virtual functions.
//
// For example, here is a walker that counts the number of for loops in an
// MLFunction.
//
//  /// Declare the class.  Note that we derive from StmtWalker instantiated
//  /// with _our new subclasses_ type.
//  struct LoopCounter : public StmtWalker<LoopCounter> {
//    unsigned numLoops;
//    LoopCounter() : numLoops(0) {}
//    void visitForStmt(ForStmt &fs) { ++numLoops; }
//  };
//
//  And this class would be used like this:
//    LoopCounter lc;
//    lc.walk(function);
//    numLoops = lc.numLoops;
//
// There  are 'visit' methods for Operation, ForStmt, IfStmt, and
// MLFunction, which recursively process all contained statements.
//
// Note that if you don't implement visitXXX for some statement type,
// the visitXXX method for Statement superclass will be invoked.
//
// The optional second template argument specifies the type that statement
// visitation functions should return. If you specify this, you *MUST* provide
// an implementation of every visit<#Statement>(StmtType *).
//
// Note that these classes are specifically designed as a template to avoid
// virtual function call overhead.  Defining and using a StmtVisitor is just
// as efficient as having your own switch statement over the statement
// opcode.

//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STMTVISITOR_H
#define MLIR_IR_STMTVISITOR_H

#include "mlir/IR/Function.h"
#include "mlir/IR/Statements.h"

namespace mlir {

/// Base class for statement visitors.
template <typename SubClass, typename RetTy = void> class StmtVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the StmtVisitor that you
  // use to visit statements.

public:
  // Function to visit a statement.
  RetTy visit(Statement *s) {
    static_assert(std::is_base_of<StmtVisitor, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (s->getKind()) {
    case Statement::Kind::For:
      return static_cast<SubClass *>(this)->visitForStmt(cast<ForStmt>(s));
    case Statement::Kind::If:
      return static_cast<SubClass *>(this)->visitIfStmt(cast<IfStmt>(s));
    case Statement::Kind::Operation:
      return static_cast<SubClass *>(this)->visitOperationStmt(
          cast<OperationStmt>(s));
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular statement type.
  // The default behavior is to generalize the statement type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // When visiting a for stmt, if stmt, or an operation stmt directly, these
  // methods get called to indicate when transitioning into a new unit.
  void visitForStmt(ForStmt *forStmt) {}
  void visitIfStmt(IfStmt *ifStmt) {}
  void visitOperationStmt(OperationStmt *opStmt) {}
};

/// Base class for statement walkers. A walker can traverse depth first in
/// pre-order or post order. The walk methods without a suffix do a pre-order
/// traversal while those that traverse in post order have a PostOrder suffix.
template <typename SubClass, typename RetTy = void> class StmtWalker {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the StmtWalker used to
  // walk statements.

public:
  // Generic walk method - allow walk to all statements in a range.
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

  // Define walkers for MLFunction and all MLFunction statement kinds.
  void walk(MLFunction *f) {
    static_cast<SubClass *>(this)->visitMLFunction(f);
    static_cast<SubClass *>(this)->walk(f->getBody()->begin(),
                                        f->getBody()->end());
  }

  void walkPostOrder(MLFunction *f) {
    static_cast<SubClass *>(this)->walkPostOrder(f->getBody()->begin(),
                                                 f->getBody()->end());
    static_cast<SubClass *>(this)->visitMLFunction(f);
  }

  RetTy walkOpStmt(OperationStmt *opStmt) {
    return static_cast<SubClass *>(this)->visitOperationStmt(opStmt);
  }

  void walkForStmt(ForStmt *forStmt) {
    static_cast<SubClass *>(this)->visitForStmt(forStmt);
    auto *body = forStmt->getBody();
    static_cast<SubClass *>(this)->walk(body->begin(), body->end());
  }

  void walkForStmtPostOrder(ForStmt *forStmt) {
    auto *body = forStmt->getBody();
    static_cast<SubClass *>(this)->walkPostOrder(body->begin(), body->end());
    static_cast<SubClass *>(this)->visitForStmt(forStmt);
  }

  void walkIfStmt(IfStmt *ifStmt) {
    static_cast<SubClass *>(this)->visitIfStmt(ifStmt);
    static_cast<SubClass *>(this)->walk(ifStmt->getThen()->begin(),
                                        ifStmt->getThen()->end());
    if (ifStmt->hasElse())
      static_cast<SubClass *>(this)->walk(ifStmt->getElse()->begin(),
                                          ifStmt->getElse()->end());
  }

  void walkIfStmtPostOrder(IfStmt *ifStmt) {
    static_cast<SubClass *>(this)->walkPostOrder(ifStmt->getThen()->begin(),
                                                 ifStmt->getThen()->end());
    if (ifStmt->hasElse())
      static_cast<SubClass *>(this)->walkPostOrder(ifStmt->getElse()->begin(),
                                                   ifStmt->getElse()->end());
    static_cast<SubClass *>(this)->visitIfStmt(ifStmt);
  }

  // Function to walk a statement.
  RetTy walk(Statement *s) {
    static_assert(std::is_base_of<StmtWalker, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (s->getKind()) {
    case Statement::Kind::For:
      return static_cast<SubClass *>(this)->walkForStmt(cast<ForStmt>(s));
    case Statement::Kind::If:
      return static_cast<SubClass *>(this)->walkIfStmt(cast<IfStmt>(s));
    case Statement::Kind::Operation:
      return static_cast<SubClass *>(this)->walkOpStmt(cast<OperationStmt>(s));
    }
  }

  // Function to walk a statement in post order DFS.
  RetTy walkPostOrder(Statement *s) {
    static_assert(std::is_base_of<StmtWalker, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (s->getKind()) {
    case Statement::Kind::For:
      return static_cast<SubClass *>(this)->walkForStmtPostOrder(
          cast<ForStmt>(s));
    case Statement::Kind::If:
      return static_cast<SubClass *>(this)->walkIfStmtPostOrder(
          cast<IfStmt>(s));
    case Statement::Kind::Operation:
      return static_cast<SubClass *>(this)->walkOpStmt(cast<OperationStmt>(s));
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular statement type.
  // The default behavior is to generalize the statement type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.

  // When visiting a specific stmt directly during a walk, these methods get
  // called. These are typically O(1) complexity and shouldn't be recursively
  // processing their descendants in some way. When using RetTy, all of these
  // need to be overridden.
  void visitMLFunction(MLFunction *f) {}
  void visitForStmt(ForStmt *forStmt) {}
  void visitIfStmt(IfStmt *ifStmt) {}
  void visitOperationStmt(OperationStmt *opStmt) {}
};

} // end namespace mlir

#endif // MLIR_IR_STMTVISITOR_H
