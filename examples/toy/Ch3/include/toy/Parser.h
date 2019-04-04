//===- Parser.h - Toy Language Parser -------------------------------------===//
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
// This file implements the parser for the Toy language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_PARSER_H
#define MLIR_TUTORIAL_TOY_PARSER_H

#include "toy/AST.h"
#include "toy/Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

namespace toy {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> ParseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in this vector.
    std::vector<FunctionAST> functions;
    while (auto F = ParseDefinition()) {
      functions.push_back(std::move(*F));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return llvm::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> ParseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    // return takes an optional argument
    llvm::Optional<std::unique_ptr<ExprAST>> expr;
    if (lexer.getCurToken() != ';') {
      expr = ParseExpression();
      if (!expr)
        return nullptr;
    }
    return llvm::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> ParseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto Result =
        llvm::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(Result);
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> ParseTensorLitteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(ParseTensorLitteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(ParseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]
    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());
    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
          return llvm::isa<LiteralExprAST>(expr.get());
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expession");

      // Append the nested dimensions to the current level
      auto &firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expession");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expession");
      }
    }
    return llvm::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                             std::move(dims));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> ParseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto V = ParseExpression();
    if (!V)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return V;
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> ParseIdentifierExpr() {
    std::string name = lexer.getId();

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    if (lexer.getCurToken() != '(') // Simple variable ref.
      return llvm::make_unique<VariableExprAST>(std::move(loc), name);

    // This is a function call.
    lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto Arg = ParseExpression())
          Args.push_back(std::move(Arg));
        else
          return nullptr;

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (name == "print") {
      if (Args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return llvm::make_unique<PrintExprAST>(std::move(loc),
                                             std::move(Args[0]));
    }

    // Call to a user-defined function
    return llvm::make_unique<CallExprAST>(std::move(loc), name,
                                          std::move(Args));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> ParsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return ParseIdentifierExpr();
    case tok_number:
      return ParseNumberExpr();
    case '(':
      return ParseParenExpr();
    case '[':
      return ParseTensorLitteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                         std::unique_ptr<ExprAST> LHS) {
    // If this is a binop, find its precedence.
    while (true) {
      int TokPrec = GetTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (TokPrec < ExprPrec)
        return LHS;

      // Okay, we know this is a binop.
      int BinOp = lexer.getCurToken();
      lexer.consume(Token(BinOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto RHS = ParsePrimary();
      if (!RHS)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with RHS than the operator after RHS, let
      // the pending operator take RHS as its LHS.
      int NextPrec = GetTokPrecedence();
      if (TokPrec < NextPrec) {
        RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
        if (!RHS)
          return nullptr;
      }

      // Merge LHS/RHS.
      LHS = llvm::make_unique<BinaryExprAST>(std::move(loc), BinOp,
                                             std::move(LHS), std::move(RHS));
    }
  }

  /// expression::= primary binoprhs
  std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS)
      return nullptr;

    return ParseBinOpRHS(0, std::move(LHS));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::unique_ptr<VarType> ParseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    lexer.getNextToken(); // eat <

    auto type = llvm::make_unique<VarType>();

    while (lexer.getCurToken() == tok_number) {
      type->shape.push_back(lexer.getValue());
      lexer.getNextToken();
      if (lexer.getCurToken() == ',')
        lexer.getNextToken();
    }

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    lexer.getNextToken(); // eat >
    return type;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= var identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> ParseDeclaration() {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat var

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identified",
                                        "after 'var' declaration");
    std::string id = lexer.getId();
    lexer.getNextToken(); // eat id

    std::unique_ptr<VarType> type; // Type is optional, it can be inferred
    if (lexer.getCurToken() == '<') {
      type = ParseType();
      if (!type)
        return nullptr;
    }

    if (!type)
      type = llvm::make_unique<VarType>();
    lexer.consume(Token('='));
    auto expr = ParseExpression();
    return llvm::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                             std::move(*type), std::move(expr));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> ParseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = llvm::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto varDecl = ParseDeclaration();
        if (!varDecl)
          return nullptr;
        exprList->push_back(std::move(varDecl));
      } else if (lexer.getCurToken() == tok_return) {
        // Return statement
        auto ret = ParseReturn();
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else {
        // General expression
        auto expr = ParseExpression();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return exprList;
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> ParsePrototype() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_def);
    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string FnName = lexer.getId();
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<VariableExprAST>> args;
    if (lexer.getCurToken() != ')') {
      do {
        std::string name = lexer.getId();
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        auto decl = llvm::make_unique<VariableExprAST>(std::move(loc), name);
        args.push_back(std::move(decl));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>("}", "to end function prototype");

    // success.
    lexer.consume(Token(')'));
    return llvm::make_unique<PrototypeAST>(std::move(loc), FnName,
                                           std::move(args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> ParseDefinition() {
    auto Proto = ParsePrototype();
    if (!Proto)
      return nullptr;

    if (auto block = ParseBlock())
      return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(block));
    return nullptr;
  }

  /// Get the precedence of the pending binary operator token.
  int GetTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace toy

#endif // MLIR_TUTORIAL_TOY_PARSER_H
