//===- Parser.cpp - MLIR Parser Implementation ----------------------------===//
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
// This file implements the parser for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser.h"
#include "Lexer.h"
#include "mlir/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SourceMgr;

namespace {
/// Simple enum to make code read better.  Failure is "true" in a boolean
/// context.
enum ParseResult {
  ParseSuccess,
  ParseFailure
};

/// Main parser implementation.
class Parser {
 public:
  Parser(llvm::SourceMgr &sourceMgr) : lex(sourceMgr), curToken(lex.lexToken()){
    module.reset(new Module());
  }

  Module *parseModule();
private:
  // State.
  Lexer lex;

  // This is the next token that hasn't been consumed yet.
  Token curToken;

  // This is the result module we are parsing into.
  std::unique_ptr<Module> module;

private:
  // Helper methods.

  /// Emit an error and return failure.
  ParseResult emitError(const Twine &message);

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    curToken = lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::TokenKind kind) {
    assert(curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  // Type parsing.

  // Top level entity parsing.
  ParseResult parseFunctionSignature(StringRef &name);
  ParseResult parseExtFunc();
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

ParseResult Parser::emitError(const Twine &message) {
  // TODO(clattner): If/when we want to implement a -verify mode, this will need
  // to package up errors into SMDiagnostic and report them.
  lex.getSourceMgr().PrintMessage(curToken.getLoc(), SourceMgr::DK_Error,
                                  message);
  return ParseFailure;
}


//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

// ... TODO

//===----------------------------------------------------------------------===//
// Top-level entity parsing.
//===----------------------------------------------------------------------===//

/// Parse a function signature, starting with a name and including the parameter
/// list.
///
///   argument-list ::= type (`,` type)* | /*empty*/
///   function-signature ::= function-id `(` argument-list `)` (`->` type-list)?
///
ParseResult Parser::parseFunctionSignature(StringRef &name) {
  if (curToken.isNot(Token::at_identifier))
    return emitError("expected a function identifier like '@foo'");

  name = curToken.getSpelling().drop_front();
  consumeToken(Token::at_identifier);

  if (curToken.isNot(Token::l_paren))
    return emitError("expected '(' in function signature");
  consumeToken(Token::l_paren);

  // TODO: This should actually parse the full grammar here.

  if (curToken.isNot(Token::r_paren))
    return emitError("expected ')' in function signature");
  consumeToken(Token::r_paren);

  return ParseSuccess;
}


/// External function declarations.
///
///   ext-func ::= `extfunc` function-signature
///
ParseResult Parser::parseExtFunc() {
  consumeToken(Token::kw_extfunc);

  StringRef name;
  if (parseFunctionSignature(name))
    return ParseFailure;


  // Okay, the external function definition was parsed correctly.
  module->functionList.push_back(new Function(name));
  return ParseSuccess;
}


/// This is the top-level module parser.
Module *Parser::parseModule() {
  while (1) {
    switch (curToken.getKind()) {
    default:
      emitError("expected a top level entity");
      return nullptr;

    // If we got to the end of the file, then we're done.
    case Token::eof:
      return module.release();

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  Someday we could introduce error recovery if there was demand for
    // it.
    case Token::error:
      return nullptr;

    case Token::kw_extfunc:
      if (parseExtFunc())
        return nullptr;
      break;

    // TODO: cfgfunc, mlfunc, affine entity declarations, etc.
    }
  }
}

//===----------------------------------------------------------------------===//

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, it emits diagnostics and returns null.
Module *mlir::parseSourceFile(llvm::SourceMgr &sourceMgr) {
  return Parser(sourceMgr).parseModule();
}
