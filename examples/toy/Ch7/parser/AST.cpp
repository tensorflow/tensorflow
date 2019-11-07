//===- AST.cpp - Helper for printing out the Toy AST ----------------------===//
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
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/AST.h"

#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(StructLiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);
  void dump(StructAST *node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
#define dispatch(CLASS)                                                        \
  if (CLASS *node = llvm::dyn_cast<CLASS>(expr))                               \
    return dump(node);
  dispatch(VarDeclExprAST);
  dispatch(LiteralExprAST);
  dispatch(StructLiteralExprAST);
  dispatch(NumberExprAST);
  dispatch(VariableExprAST);
  dispatch(ReturnExprAST);
  dispatch(BinaryExprAST);
  dispatch(CallExprAST);
  dispatch(PrintExprAST);
  // No match, fallback to a generic message
  INDENT();
  llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " " << loc(varDecl) << "\n";
  if (auto *initVal = varDecl->getInitVal())
    dump(initVal);
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &expr : *exprList)
    dump(expr.get());
  indent();
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void printLitHelper(ExprAST *litOrNum) {
  // Inside a literal expression we can have either a number or another literal
  if (auto num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
    llvm::errs() << num->getValue();
    return;
  }
  auto *literal = llvm::cast<LiteralExprAST>(litOrNum);

  // Print the dimension for this literal first
  llvm::errs() << "<";
  mlir::interleaveComma(literal->getDims(), llvm::errs());
  llvm::errs() << ">";

  // Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  mlir::interleaveComma(literal->getValues(), llvm::errs(),
                        [&](auto &elt) { printLitHelper(elt.get()); });
  llvm::errs() << "]";
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(LiteralExprAST *node) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(node);
  llvm::errs() << " " << loc(node) << "\n";
}

/// Print a struct literal.
void ASTDumper::dump(StructLiteralExprAST *node) {
  INDENT();
  llvm::errs() << "Struct Literal: ";
  for (auto &value : node->getValues())
    dump(value.get());
  indent();
  llvm::errs() << " " << loc(node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *node) {
  INDENT();
  llvm::errs() << "var: " << node->getName() << " " << loc(node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (node->getExpr().hasValue())
    return dump(*node->getExpr());
  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *node) {
  INDENT();
  llvm::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
  for (auto &arg : node->getArgs())
    dump(arg.get());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *node) {
  INDENT();
  llvm::errs() << "Print [ " << loc(node) << "\n";
  dump(node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type) {
  llvm::errs() << "<";
  if (!type.name.empty())
    llvm::errs() << type.name;
  else
    mlir::interleaveComma(type.shape, llvm::errs());
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "'\n";
  indent();
  llvm::errs() << "Params: [";
  mlir::interleaveComma(node->getArgs(), llvm::errs(),
                        [](auto &arg) { llvm::errs() << arg->getName(); });
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}

/// Print a struct.
void ASTDumper::dump(StructAST *node) {
  INDENT();
  llvm::errs() << "Struct: " << node->getName() << " " << loc(node) << "\n";

  {
    INDENT();
    llvm::errs() << "Variables: [\n";
    for (auto &variable : node->getVariables())
      dump(variable.get());
    indent();
    llvm::errs() << "]\n";
  }
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &record : *node) {
    if (FunctionAST *function = llvm::dyn_cast<FunctionAST>(record.get()))
      dump(function);
    else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get()))
      dump(str);
    else
      llvm::errs() << "<unknown Record, kind " << record->getKind() << ">\n";
  }
}

namespace toy {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace toy
