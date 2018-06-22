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

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::TokenKind kind) {
    if (curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  ParseResult parseCommaSeparatedList(Token::TokenKind rightToken,
                               const std::function<ParseResult()> &parseElement,
                                      bool allowEmptyList = true);

  // Type parsing.
  ParseResult parsePrimitiveType();
  ParseResult parseElementType();
  ParseResult parseVectorType();
  ParseResult parseDimensionListRanked(SmallVectorImpl<int> &dimensions);
  ParseResult parseTensorType();
  ParseResult parseMemRefType();
  ParseResult parseFunctionType();
  ParseResult parseType();
  ParseResult parseTypeList();

  // Top level entity parsing.
  ParseResult parseFunctionSignature(StringRef &name);
  ParseResult parseExtFunc();
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

ParseResult Parser::emitError(const Twine &message) {
  // If we hit a parse error in response to a lexer error, then the lexer
  // already emitted an error.
  if (curToken.is(Token::error))
    return ParseFailure;

  // TODO(clattner): If/when we want to implement a -verify mode, this will need
  // to package up errors into SMDiagnostic and report them.
  lex.getSourceMgr().PrintMessage(curToken.getLoc(), SourceMgr::DK_Error,
                                  message);
  return ParseFailure;
}

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
///
///   abstract-list ::= rightToken                  // if allowEmptyList == true
///   abstract-list ::= element (',' element)* rightToken
///
ParseResult Parser::
parseCommaSeparatedList(Token::TokenKind rightToken,
                        const std::function<ParseResult()> &parseElement,
                        bool allowEmptyList) {
  // Handle the empty case.
  if (curToken.is(rightToken)) {
    if (!allowEmptyList)
      return emitError("expected list element");
    consumeToken(rightToken);
    return ParseSuccess;
  }

  // Non-empty case starts with an element.
  if (parseElement())
    return ParseFailure;

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElement())
      return ParseFailure;
  }

  // Consume the end character.
  if (!consumeIf(rightToken))
    return emitError("expected ',' or ')'");

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

/// Parse the low-level fixed dtypes in the system.
///
///   primitive-type
///      ::= `f16` | `bf16` | `f32` | `f64`      // Floating point
///        | `i1` | `i8` | `i16` | `i32` | `i64` // Sized integers
///        | `int`
///
ParseResult Parser::parsePrimitiveType() {
  // TODO: Build IR objects.
  switch (curToken.getKind()) {
  default: return emitError("expected type");
  case Token::kw_bf16:
    consumeToken(Token::kw_bf16);
    return ParseSuccess;
  case Token::kw_f16:
    consumeToken(Token::kw_f16);
    return ParseSuccess;
  case Token::kw_f32:
    consumeToken(Token::kw_f32);
    return ParseSuccess;
  case Token::kw_f64:
    consumeToken(Token::kw_f64);
    return ParseSuccess;
  case Token::kw_i1:
    consumeToken(Token::kw_i1);
    return ParseSuccess;
  case Token::kw_i16:
    consumeToken(Token::kw_i16);
    return ParseSuccess;
  case Token::kw_i32:
    consumeToken(Token::kw_i32);
    return ParseSuccess;
  case Token::kw_i64:
    consumeToken(Token::kw_i64);
    return ParseSuccess;
  case Token::kw_i8:
    consumeToken(Token::kw_i8);
    return ParseSuccess;
  case Token::kw_int:
    consumeToken(Token::kw_int);
    return ParseSuccess;
  }
}

/// Parse the element type of a tensor or memref type.
///
///   element-type ::= primitive-type | vector-type
///
ParseResult Parser::parseElementType() {
  if (curToken.is(Token::kw_vector))
    return parseVectorType();

  return parsePrimitiveType();
}

/// Parse a vector type.
///
///   vector-type ::= `vector` `<` const-dimension-list primitive-type `>`
///   const-dimension-list ::= (integer-literal `x`)+
///
ParseResult Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (!consumeIf(Token::less))
    return emitError("expected '<' in vector type");

  if (curToken.isNot(Token::integer))
    return emitError("expected dimension size in vector type");

  SmallVector<unsigned, 4> dimensions;
  while (curToken.is(Token::integer)) {
    // Make sure this integer value is in bound and valid.
    auto dimension = curToken.getUnsignedIntegerValue();
    if (!dimension.hasValue())
      return emitError("invalid dimension in vector type");
    dimensions.push_back(dimension.getValue());

    consumeToken(Token::integer);

    // Make sure we have an 'x' or something like 'xbf32'.
    if (curToken.isNot(Token::bare_identifier) ||
        curToken.getSpelling()[0] != 'x')
      return emitError("expected 'x' in vector dimension list");

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (curToken.getSpelling().size() != 1)
      lex.resetPointer(curToken.getSpelling().data()+1);

    // Consume the 'x'.
    consumeToken(Token::bare_identifier);
  }

  // Parse the element type.
  if (parsePrimitiveType())
    return ParseFailure;

  if (!consumeIf(Token::greater))
    return emitError("expected '>' in vector type");

  // TODO: Form IR object.

  return ParseSuccess;
}

/// Parse a dimension list of a tensor or memref type.  This populates the
/// dimension list, returning -1 for the '?' dimensions.
///
///   dimension-list-ranked ::= (dimension `x`)*
///   dimension ::= `?` | integer-literal
///
ParseResult Parser::parseDimensionListRanked(SmallVectorImpl<int> &dimensions) {
  while (curToken.isAny(Token::integer, Token::question)) {
    if (consumeIf(Token::question)) {
      dimensions.push_back(-1);
    } else {
      // Make sure this integer value is in bound and valid.
      auto dimension = curToken.getUnsignedIntegerValue();
      if (!dimension.hasValue() || (int)dimension.getValue() < 0)
        return emitError("invalid dimension");
      dimensions.push_back((int)dimension.getValue());
      consumeToken(Token::integer);
    }

    // Make sure we have an 'x' or something like 'xbf32'.
    if (curToken.isNot(Token::bare_identifier) ||
        curToken.getSpelling()[0] != 'x')
      return emitError("expected 'x' in dimension list");

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (curToken.getSpelling().size() != 1)
      lex.resetPointer(curToken.getSpelling().data()+1);

    // Consume the 'x'.
    consumeToken(Token::bare_identifier);
  }

  return ParseSuccess;
}

/// Parse a tensor type.
///
///   tensor-type ::= `tensor` `<` dimension-list element-type `>`
///   dimension-list ::= dimension-list-ranked | `??`
///
ParseResult Parser::parseTensorType() {
  consumeToken(Token::kw_tensor);

  if (!consumeIf(Token::less))
    return emitError("expected '<' in tensor type");

  bool isUnranked;
  SmallVector<int, 4> dimensions;

  if (consumeIf(Token::questionquestion)) {
    isUnranked = true;
  } else {
    isUnranked = false;
    if (parseDimensionListRanked(dimensions))
      return ParseFailure;
  }

  // Parse the element type.
  if (parseElementType())
    return ParseFailure;

  if (!consumeIf(Token::greater))
    return emitError("expected '>' in tensor type");

  // TODO: Form IR object.

  return ParseSuccess;
}

/// Parse a memref type.
///
///   memref-type ::= `memref` `<` dimension-list-ranked element-type
///                   (`,` semi-affine-map-composition)? (`,` memory-space)? `>`
///
///   semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
///   memory-space ::= integer-literal /* | TODO: address-space-id */
///
ParseResult Parser::parseMemRefType() {
  consumeToken(Token::kw_memref);

  if (!consumeIf(Token::less))
    return emitError("expected '<' in memref type");

  SmallVector<int, 4> dimensions;
  if (parseDimensionListRanked(dimensions))
    return ParseFailure;

  // Parse the element type.
  if (parseElementType())
    return ParseFailure;

  // TODO: Parse semi-affine-map-composition.
  // TODO: Parse memory-space.

  if (!consumeIf(Token::greater))
    return emitError("expected '>' in memref type");

  // TODO: Form IR object.

  return ParseSuccess;
}



/// Parse a function type.
///
///   function-type ::= type-list-parens `->` type-list
///
ParseResult Parser::parseFunctionType() {
  assert(curToken.is(Token::l_paren));

  if (parseTypeList())
    return ParseFailure;

  if (!consumeIf(Token::arrow))
    return emitError("expected '->' in function type");

  if (parseTypeList())
    return ParseFailure;

  // TODO: Build IR object.
  return ParseSuccess;
}


/// Parse an arbitrary type.
///
///   type ::= primitive-type
///          | vector-type
///          | tensor-type
///          | memref-type
///          | function-type
///   element-type ::= primitive-type | vector-type
///
ParseResult Parser::parseType() {
  switch (curToken.getKind()) {
  case Token::kw_memref: return parseMemRefType();
  case Token::kw_tensor: return parseTensorType();
  case Token::kw_vector: return parseVectorType();
  case Token::l_paren:   return parseFunctionType();
  default:
    return parsePrimitiveType();
  }
}

/// Parse a "type list", which is a singular type, or a parenthesized list of
/// types.
///
///   type-list ::= type-list-parens | type
///   type-list-parens ::= `(` `)`
///                      | `(` type (`,` type)* `)`
///
ParseResult Parser::parseTypeList() {
  // If there is no parens, then it must be a singular type.
  if (!consumeIf(Token::l_paren))
    return parseType();

  if (parseCommaSeparatedList(Token::r_paren,
                              [&]() -> ParseResult {
    // TODO: Add to list of IR values we're parsing.
    return parseType();
  })) {
    return ParseFailure;
  }

  // TODO: Build IR objects.
  return ParseSuccess;
}


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

  if (parseTypeList())
    return ParseFailure;

  // Parse the return type if present.
  if (consumeIf(Token::arrow)) {
    if (parseTypeList())
      return ParseFailure;

    // TODO: Build IR object.
  }

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
