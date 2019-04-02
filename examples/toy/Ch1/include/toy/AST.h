//===- AST.h - Node definition for the Toy AST ----------------------------===//
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
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_AST_H_
#define MLIR_TUTORIAL_TOY_AST_H_

#include "toy/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace toy {

/// A variable
struct VarType {
  enum { TY_FLOAT, TY_INT } elt_ty;
  std::vector<int> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print, // builtin
    Expr_If,
    Expr_For,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}

  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(Location loc, double Val) : ExprAST(Expr_Num, loc), Val(Val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Num; }
};

///
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  std::vector<std::unique_ptr<ExprAST>> &getValues() { return values; }
  std::vector<int64_t> &getDims() { return dims; }
  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, const std::string &name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Var; }
};

///
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, const std::string &name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_VarDecl; }
};

///
class ReturnExprAST : public ExprAST {
  llvm::Optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, llvm::Optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

  llvm::Optional<ExprAST *> getExpr() {
    if (expr.hasValue())
      return expr->get();
    return llvm::NoneType();
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  char getOp() { return Op; }
  ExprAST *getLHS() { return LHS.get(); }
  ExprAST *getRHS() { return RHS.get(); }

  BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : ExprAST(Expr_BinOp, loc), Op(Op), LHS(std::move(LHS)),
        RHS(std::move(RHS)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(Location loc, const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : ExprAST(Expr_Call, loc), Callee(Callee), Args(std::move(Args)) {}

  llvm::StringRef getCallee() { return Callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return Args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> Arg)
      : ExprAST(Expr_Print, loc), Arg(std::move(Arg)) {}

  ExprAST *getArg() { return Arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(location), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  const std::string &getName() const { return name; }
  const std::vector<std::unique_ptr<VariableExprAST>> &getArgs() {
    return args;
  }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprASTList> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprASTList> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
  PrototypeAST *getProto() { return Proto.get(); }
  ExprASTList *getBody() { return Body.get(); }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() -> decltype(functions.begin()) { return functions.begin(); }
  auto end() -> decltype(functions.end()) { return functions.end(); }
};

void dump(ModuleAST &);

} // namespace toy

#endif // MLIR_TUTORIAL_TOY_AST_H_
