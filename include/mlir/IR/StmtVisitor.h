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
// This file defines the Statement visitor class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STMTVISITOR_H
#define MLIR_IR_STMTVISITOR_H

#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"

namespace mlir {

/// Base class for statement visitors.
///
/// Statement visitors are used when you want to perform different actions
/// for different kinds of statements without having to use lots of casts
/// and a big switch statement.
///
/// To define your own visitor, inherit from this class, specifying your
/// new type for the 'SubClass' template parameter, and "override" visitXXX
/// functions in your class. This class is defined in terms of statically
/// resolved overloading, not virtual functions.
///
/// For example, here is a visitor that counts the number of for loops in an
/// MLFunction.
///
///  /// Declare the class.  Note that we derive from StmtVisitor instantiated
///  /// with _our new subclasses_ type.
///  struct LoopCounter : public StmtVisitor<LoopCounter> {
///    unsigned numLoops;
///    LoopCounter() : numLoops(0) {}
///    void visitForStmt(ForStmt &fs) { ++numLoops; }
///  };
///
///  And this class would be used like this:
///    LoopCounter lc;
///    lc.visit(function);
///    numLoops = lc.numLoops;
///
/// There  are 'visit' methods for Operation, ForStmt, IfStmt, and
/// MLFunction, which recursively process all contained statements.
///
/// Note that if you don't implement visitXXX for some statement type,
/// the visitXXX method for Statement superclass will be invoked.
///
/// The optional second template argument specifies the type that statement
/// visitation functions should return. If you specify this, you *MUST* provide
/// an implementation of visitStatement.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead.  Defining and using a StmtVisitor is just
/// as efficient as having your own switch statement over the statement
/// opcode.
template <typename SubClass> class StmtVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the StmtVisitor that you
  // use to visit statements...

public:
  // Generic visit method - allow visitation to all statements in a range.
  template <class Iterator> void visit(Iterator Start, Iterator End) {
    while (Start != End) {
      static_cast<SubClass *>(this)->visit(&(*Start++));
    }
  }

  // Define visitors for MLFunction and all MLFunction statement kinds.
  void visit(MLFunction *f) {
    static_cast<SubClass *>(this)->visitMLFunction(f);
    visit(f->begin(), f->end());
  }

  void visit(OperationStmt *opStmt) {
    static_cast<SubClass *>(this)->visitOperationStmt(opStmt);
  }

  void visit(ForStmt *forStmt) {
    static_cast<SubClass *>(this)->visitForStmt(forStmt);
    visit(forStmt->begin(), forStmt->end());
  }

  void visit(IfStmt *ifStmt) {
    static_cast<SubClass *>(this)->visitIfStmt(ifStmt);
    visit(ifStmt->getThenClause()->begin(), ifStmt->getThenClause()->end());
    visit(ifStmt->getElseClause()->begin(), ifStmt->getElseClause()->end());
  }

  // Function to visit a statement.
  void visit(Statement *s) {
    static_assert(std::is_base_of<StmtVisitor, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (s->getKind()) {
    default:
      llvm_unreachable("Unknown statement type encountered!");
    case Statement::Kind::For:
      return static_cast<SubClass *>(this)->visit(cast<ForStmt>(s));
    case Statement::Kind::If:
      return static_cast<SubClass *>(this)->visit(cast<IfStmt>(s));
    case Statement::Kind::Operation:
      return static_cast<SubClass *>(this)->visit(cast<OperationStmt>(s));
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
  void visitMLFunction(MLFunction *f) {}
};

} // end namespace mlir

#endif // MLIR_IR_STMTVISITOR_H
