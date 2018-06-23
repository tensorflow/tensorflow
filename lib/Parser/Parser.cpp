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
#include "mlir/IR/Types.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SourceMgr;

namespace {
/// Simple enum to make code read better in cases that would otherwise return a
/// bool value.  Failure is "true" in a boolean context.
enum ParseResult {
  ParseSuccess,
  ParseFailure
};

/// Main parser implementation.
class Parser {
 public:
  Parser(llvm::SourceMgr &sourceMgr, MLIRContext *context)
     : context(context), lex(sourceMgr), curToken(lex.lexToken()){
    module.reset(new Module());
  }

  Module *parseModule();
private:
  // State.
  MLIRContext *const context;

  // The lexer for the source file we're parsing.
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

  // We have two forms of parsing methods - those that return a non-null
  // pointer on success, and those that return a ParseResult to indicate whether
  // they returned a failure.  The second class fills in by-reference arguments
  // as the results of their action.

  // Type parsing.
  PrimitiveType *parsePrimitiveType();
  Type *parseElementType();
  VectorType *parseVectorType();
  ParseResult parseDimensionListRanked(SmallVectorImpl<int> &dimensions);
  Type *parseTensorType();
  Type *parseMemRefType();
  Type *parseFunctionType();
  Type *parseType();
  ParseResult parseTypeList(SmallVectorImpl<Type*> &elements);

  // Top level entity parsing.
  ParseResult parseFunctionSignature(StringRef &name, FunctionType *&type);
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
PrimitiveType *Parser::parsePrimitiveType() {
  switch (curToken.getKind()) {
  default:
    return (emitError("expected type"), nullptr);
  case Token::kw_bf16:
    consumeToken(Token::kw_bf16);
    return Type::getBF16(context);
  case Token::kw_f16:
    consumeToken(Token::kw_f16);
    return Type::getF16(context);
  case Token::kw_f32:
    consumeToken(Token::kw_f32);
    return Type::getF32(context);
  case Token::kw_f64:
    consumeToken(Token::kw_f64);
    return Type::getF64(context);
  case Token::kw_i1:
    consumeToken(Token::kw_i1);
    return Type::getI1(context);
  case Token::kw_i8:
    consumeToken(Token::kw_i8);
    return Type::getI8(context);
  case Token::kw_i16:
    consumeToken(Token::kw_i16);
    return Type::getI16(context);
  case Token::kw_i32:
    consumeToken(Token::kw_i32);
    return Type::getI32(context);
  case Token::kw_i64:
    consumeToken(Token::kw_i64);
    return Type::getI64(context);
  case Token::kw_int:
    consumeToken(Token::kw_int);
    return Type::getInt(context);
  }
}

/// Parse the element type of a tensor or memref type.
///
///   element-type ::= primitive-type | vector-type
///
Type *Parser::parseElementType() {
  if (curToken.is(Token::kw_vector))
    return parseVectorType();

  return parsePrimitiveType();
}

/// Parse a vector type.
///
///   vector-type ::= `vector` `<` const-dimension-list primitive-type `>`
///   const-dimension-list ::= (integer-literal `x`)+
///
VectorType *Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (!consumeIf(Token::less))
    return (emitError("expected '<' in vector type"), nullptr);

  if (curToken.isNot(Token::integer))
    return (emitError("expected dimension size in vector type"), nullptr);

  SmallVector<unsigned, 4> dimensions;
  while (curToken.is(Token::integer)) {
    // Make sure this integer value is in bound and valid.
    auto dimension = curToken.getUnsignedIntegerValue();
    if (!dimension.hasValue())
      return (emitError("invalid dimension in vector type"), nullptr);
    dimensions.push_back(dimension.getValue());

    consumeToken(Token::integer);

    // Make sure we have an 'x' or something like 'xbf32'.
    if (curToken.isNot(Token::bare_identifier) ||
        curToken.getSpelling()[0] != 'x')
      return (emitError("expected 'x' in vector dimension list"), nullptr);

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (curToken.getSpelling().size() != 1)
      lex.resetPointer(curToken.getSpelling().data()+1);

    // Consume the 'x'.
    consumeToken(Token::bare_identifier);
  }

  // Parse the element type.
  auto *elementType = parsePrimitiveType();
  if (!elementType)
    return nullptr;

  if (!consumeIf(Token::greater))
    return (emitError("expected '>' in vector type"), nullptr);

  return VectorType::get(dimensions, elementType);
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
Type *Parser::parseTensorType() {
  consumeToken(Token::kw_tensor);

  if (!consumeIf(Token::less))
    return (emitError("expected '<' in tensor type"), nullptr);

  bool isUnranked;
  SmallVector<int, 4> dimensions;

  if (consumeIf(Token::questionquestion)) {
    isUnranked = true;
  } else {
    isUnranked = false;
    if (parseDimensionListRanked(dimensions))
      return nullptr;
  }

  // Parse the element type.
  auto elementType = parseElementType();
  if (!elementType)
    return nullptr;

  if (!consumeIf(Token::greater))
    return (emitError("expected '>' in tensor type"), nullptr);

  // FIXME: Add an IR representation for tensor types.
  return Type::getI1(context);
}

/// Parse a memref type.
///
///   memref-type ::= `memref` `<` dimension-list-ranked element-type
///                   (`,` semi-affine-map-composition)? (`,` memory-space)? `>`
///
///   semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
///   memory-space ::= integer-literal /* | TODO: address-space-id */
///
Type *Parser::parseMemRefType() {
  consumeToken(Token::kw_memref);

  if (!consumeIf(Token::less))
    return (emitError("expected '<' in memref type"), nullptr);

  SmallVector<int, 4> dimensions;
  if (parseDimensionListRanked(dimensions))
    return nullptr;

  // Parse the element type.
  auto elementType = parseElementType();
  if (!elementType)
    return nullptr;

  // TODO: Parse semi-affine-map-composition.
  // TODO: Parse memory-space.

  if (!consumeIf(Token::greater))
    return (emitError("expected '>' in memref type"), nullptr);

  // FIXME: Add an IR representation for memref types.
  return Type::getI1(context);
}



/// Parse a function type.
///
///   function-type ::= type-list-parens `->` type-list
///
Type *Parser::parseFunctionType() {
  assert(curToken.is(Token::l_paren));

  SmallVector<Type*, 4> arguments;
  if (parseTypeList(arguments))
    return nullptr;

  if (!consumeIf(Token::arrow))
    return (emitError("expected '->' in function type"), nullptr);

  SmallVector<Type*, 4> results;
  if (parseTypeList(results))
    return nullptr;

  return FunctionType::get(arguments, results, context);
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
Type *Parser::parseType() {
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
ParseResult Parser::parseTypeList(SmallVectorImpl<Type*> &elements) {
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseType();
    elements.push_back(elt);
    return elt ? ParseSuccess : ParseFailure;
  };

  // If there is no parens, then it must be a singular type.
  if (!consumeIf(Token::l_paren))
    return parseElt();

  if (parseCommaSeparatedList(Token::r_paren, parseElt))
    return ParseFailure;

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
ParseResult Parser::parseFunctionSignature(StringRef &name,
                                           FunctionType *&type) {
  if (curToken.isNot(Token::at_identifier))
    return emitError("expected a function identifier like '@foo'");

  name = curToken.getSpelling().drop_front();
  consumeToken(Token::at_identifier);

  if (curToken.isNot(Token::l_paren))
    return emitError("expected '(' in function signature");

   SmallVector<Type*, 4> arguments;
   if (parseTypeList(arguments))
    return ParseFailure;

  // Parse the return type if present.
  SmallVector<Type*, 4> results;
  if (consumeIf(Token::arrow)) {
    if (parseTypeList(results))
      return ParseFailure;
  }
  type = FunctionType::get(arguments, results, context);
  return ParseSuccess;
}


/// External function declarations.
///
///   ext-func ::= `extfunc` function-signature
///
ParseResult Parser::parseExtFunc() {
  consumeToken(Token::kw_extfunc);

  StringRef name;
  FunctionType *type = nullptr;
  if (parseFunctionSignature(name, type))
    return ParseFailure;


  // Okay, the external function definition was parsed correctly.
  module->functionList.push_back(new Function(name, type));
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
Module *mlir::parseSourceFile(llvm::SourceMgr &sourceMgr, MLIRContext *context){
  return Parser(sourceMgr, context).parseModule();
}
