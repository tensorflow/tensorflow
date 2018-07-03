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
#include <stack>

#include "mlir/Parser.h"
#include "Lexer.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SourceMgr;
using llvm::SMLoc;

namespace {
class CFGFunctionParserState;
class AffineMapParserState;

/// Simple enum to make code read better in cases that would otherwise return a
/// bool value.  Failure is "true" in a boolean context.
enum ParseResult {
  ParseSuccess,
  ParseFailure
};

/// Main parser implementation.
class Parser {
public:
  Parser(llvm::SourceMgr &sourceMgr, MLIRContext *context,
         SMDiagnosticHandlerTy errorReporter)
      : context(context), lex(sourceMgr, errorReporter),
        curToken(lex.lexToken()), errorReporter(std::move(errorReporter)) {
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

  // The diagnostic error reporter.
  SMDiagnosticHandlerTy errorReporter;

  // This is the result module we are parsing into.
  std::unique_ptr<Module> module;

  // A map from affine map identifier to AffineMap.
  // TODO(andydavis) Remove use of unique_ptr when AffineMaps are bump pointer
  // allocated.
  llvm::StringMap<std::unique_ptr<AffineMap>> affineMaps;

private:
  // Helper methods.

  /// Emit an error and return failure.
  ParseResult emitError(const Twine &message) {
    return emitError(curToken.getLoc(), message);
  }
  ParseResult emitError(SMLoc loc, const Twine &message);

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    curToken = lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::Kind kind) {
    if (curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  ParseResult parseCommaSeparatedList(Token::Kind rightToken,
                               const std::function<ParseResult()> &parseElement,
                                      bool allowEmptyList = true);

  // We have two forms of parsing methods - those that return a non-null
  // pointer on success, and those that return a ParseResult to indicate whether
  // they returned a failure.  The second class fills in by-reference arguments
  // as the results of their action.

  // Type parsing.
  Type *parsePrimitiveType();
  Type *parseElementType();
  VectorType *parseVectorType();
  ParseResult parseDimensionListRanked(SmallVectorImpl<int> &dimensions);
  Type *parseTensorType();
  Type *parseMemRefType();
  Type *parseFunctionType();
  Type *parseType();
  ParseResult parseTypeList(SmallVectorImpl<Type*> &elements);

  // Identifiers
  ParseResult parseDimIdList(SmallVectorImpl<StringRef> &dims,
                             SmallVectorImpl<StringRef> &symbols);
  ParseResult parseSymbolIdList(SmallVectorImpl<StringRef> &dims,
                                SmallVectorImpl<StringRef> &symbols);
  StringRef parseDimOrSymbolId(SmallVectorImpl<StringRef> &dims,
                               SmallVectorImpl<StringRef> &symbols,
                               bool symbol);

  // Polyhedral structures
  ParseResult parseAffineMapDef();
  AffineMap *parseAffineMapInline(StringRef mapId);
  AffineExpr *parseAffineExpr(AffineMapParserState &state);

  // Functions.
  ParseResult parseFunctionSignature(StringRef &name, FunctionType *&type);
  ParseResult parseExtFunc();
  ParseResult parseCFGFunc();
  ParseResult parseMLFunc();
  ParseResult parseBasicBlock(CFGFunctionParserState &functionState);
  MLStatement *parseMLStatement(MLFunction *currentFunction);

  OperationInst *parseCFGOperation(CFGFunctionParserState &functionState);
  TerminatorInst *parseTerminator(CFGFunctionParserState &functionState);

};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

ParseResult Parser::emitError(SMLoc loc, const Twine &message) {
  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (curToken.is(Token::error))
    return ParseFailure;

  errorReporter(
      lex.getSourceMgr().GetMessage(loc, SourceMgr::DK_Error, message));
  return ParseFailure;
}

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
///
///   abstract-list ::= rightToken                  // if allowEmptyList == true
///   abstract-list ::= element (',' element)* rightToken
///
ParseResult Parser::
parseCommaSeparatedList(Token::Kind rightToken,
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
    return emitError("expected ',' or '" + Token::getTokenSpelling(rightToken) +
                     "'");

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

/// Parse the low-level fixed dtypes in the system.
///
///   primitive-type ::= `f16` | `bf16` | `f32` | `f64`
///   primitive-type ::= integer-type
///   primitive-type ::= `affineint`
///
Type *Parser::parsePrimitiveType() {
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
  case Token::kw_affineint:
    consumeToken(Token::kw_affineint);
    return Type::getAffineInt(context);
  case Token::inttype: {
    auto width = curToken.getIntTypeBitwidth();
    if (!width.hasValue())
      return (emitError("invalid integer width"), nullptr);
    consumeToken(Token::inttype);
    return Type::getInt(width.getValue(), context);
  }
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

  if (isUnranked)
    return UnrankedTensorType::get(elementType);
  return RankedTensorType::get(dimensions, elementType);
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
  return Type::getInt(1, context);
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

namespace {
/// This class represents the transient parser state while parsing an affine
/// expression.
class AffineMapParserState {
 public:
  explicit AffineMapParserState(ArrayRef<StringRef> dims,
                                ArrayRef<StringRef> symbols) :
    dims_(dims), symbols_(symbols) {}

  unsigned dimCount() const { return dims_.size(); }
  unsigned symbolCount() const { return symbols_.size(); }

  // Stack operations for affine expression parsing
  // TODO(bondhugula): all of this will be improved/made more principled
  void pushAffineExpr(AffineExpr *expr) { exprStack.push(expr); }
  AffineExpr *popAffineExpr() {
    auto *t = exprStack.top();
    exprStack.pop();
    return t;
  }
  AffineExpr *topAffineExpr() { return exprStack.top(); }

  ArrayRef<StringRef> getDims() const { return dims_; }
  ArrayRef<StringRef> getSymbols() const { return symbols_; }

 private:
  const ArrayRef<StringRef> dims_;
  const ArrayRef<StringRef> symbols_;

  // TEMP: stack to hold affine expressions
  std::stack<AffineExpr *> exprStack;
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// Polyhedral structures.
//===----------------------------------------------------------------------===//

/// Affine map declaration.
///
///  affine-map-def ::= affine-map-id `=` affine-map-inline
///
ParseResult Parser::parseAffineMapDef() {
  assert(curToken.is(Token::affine_map_identifier));

  StringRef affineMapId = curToken.getSpelling().drop_front();
  consumeToken(Token::affine_map_identifier);

  // Check that 'affineMapId' is unique.
  // TODO(andydavis) Add a unit test for this case.
  if (affineMaps.count(affineMapId) > 0)
    return emitError("redefinition of affine map id '" + affineMapId + "'");
  // Parse the '='
  if (!consumeIf(Token::equal))
    return emitError("expected '=' in affine map outlined definition");

  auto *affineMap = parseAffineMapInline(affineMapId);
  affineMaps[affineMapId].reset(affineMap);
  if (!affineMap) return ParseFailure;

  module->affineMapList.push_back(affineMap);
  return affineMap ? ParseSuccess : ParseFailure;
}

///
/// Parse a multi-dimensional affine expression
/// affine-expr ::= `(` affine-expr `)`
///              | affine-expr `+` affine-expr
///              | affine-expr `-` affine-expr
///              | `-`? integer-literal `*` affine-expr
///              | `ceildiv` `(` affine-expr `,` integer-literal `)`
///              | `floordiv` `(` affine-expr `,` integer-literal `)`
///              | affine-expr `mod` integer-literal
///              | bare-id
///              | `-`? integer-literal
/// multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
///
/// Use 'state' to check if valid identifiers appear.
///
AffineExpr *Parser::parseAffineExpr(AffineMapParserState &state) {
  // TODO(bondhugula): complete support for this
  // The code below is all placeholder / it is wrong / not complete
  // Operator precedence not considered; pure left to right associativity
  if (curToken.is(Token::comma)) {
    emitError("expecting affine expression");
    return nullptr;
  }

  while (curToken.isNot(Token::comma, Token::r_paren,
                        Token::eof, Token::error)) {
    switch (curToken.getKind()) {
      case Token::bare_identifier: {
        // TODO(bondhugula): look up state to see if it's a symbol or dim_id and
        // get its position
        AffineExpr *expr = AffineDimExpr::get(0, context);
        state.pushAffineExpr(expr);
        consumeToken(Token::bare_identifier);
        break;
      }
      case Token::plus: {
        consumeToken(Token::plus);
        if (state.topAffineExpr()) {
          auto lChild = state.popAffineExpr();
          auto rChild = parseAffineExpr(state);
          if (rChild) {
            auto binaryOpExpr = AffineAddExpr::get(lChild, rChild, context);
            state.popAffineExpr();
            state.pushAffineExpr(binaryOpExpr);
          } else {
            emitError("right operand of + missing");
          }
        } else {
          emitError("left operand of + missing");
        }
        break;
      }
      case Token::integer: {
        AffineExpr *expr = AffineConstantExpr::get(
            curToken.getUnsignedIntegerValue().getValue(), context);
        state.pushAffineExpr(expr);
        consumeToken(Token::integer);
        break;
      }
      case Token::l_paren: {
        consumeToken(Token::l_paren);
        break;
      }
      case Token::r_paren: {
        consumeToken(Token::r_paren);
        break;
      }
      default: {
        emitError("affine map expr parse impl incomplete/unexpected token");
        return nullptr;
      }
    }
  }
  if (!state.topAffineExpr()) {
    // An error will be emitted by parse comma separated list on an empty list
    return nullptr;
  }
  return state.topAffineExpr();
}

// Return empty string if no bare id was found
StringRef Parser::parseDimOrSymbolId(SmallVectorImpl<StringRef> &dims,
                                     SmallVectorImpl<StringRef> &symbols,
                                     bool symbol = false) {
  if (curToken.isNot(Token::bare_identifier)) {
    emitError("expected bare identifier");
    return StringRef();
  }
  // TODO(bondhugula): check whether the id already exists in either
  // state.symbols or state.dims; report error if it does; otherwise create a
  // new one.
  StringRef ref = curToken.getSpelling();
  consumeToken(Token::bare_identifier);
  return ref;
}

ParseResult Parser::parseSymbolIdList(SmallVectorImpl<StringRef> &dims,
                                      SmallVectorImpl<StringRef> &symbols) {
  if (!consumeIf(Token::l_bracket)) return emitError("expected '['");

  auto parseElt = [&]() -> ParseResult {
    auto elt = parseDimOrSymbolId(dims, symbols, true);
    // FIXME(bondhugula): assuming dim arg for now
    if (!elt.empty()) {
      symbols.push_back(elt);
      return ParseSuccess;
    }
    return ParseFailure;
  };
  return parseCommaSeparatedList(Token::r_bracket, parseElt);
}

// TODO(andy,bondhugula)
ParseResult Parser::parseDimIdList(SmallVectorImpl<StringRef> &dims,
                                   SmallVectorImpl<StringRef> &symbols) {
  if (!consumeIf(Token::l_paren))
    return emitError("expected '(' at start of dimensional identifiers list");

  auto parseElt = [&]() -> ParseResult {
    auto elt = parseDimOrSymbolId(dims, symbols, false);
    if (!elt.empty()) {
      dims.push_back(elt);
      return ParseSuccess;
    }
    return ParseFailure;
  };

  return parseCommaSeparatedList(Token::r_paren, parseElt);
}

/// Affine map definition.
///
///  affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        ( `size` `(` dim-size (`,` dim-size)* `)` )?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
AffineMap *Parser::parseAffineMapInline(StringRef mapId) {
  SmallVector<StringRef, 4> dims;
  SmallVector<StringRef, 4> symbols;

  // List of dimensional identifiers.
  if (parseDimIdList(dims, symbols)) return nullptr;

  // Symbols are optional.
  if (curToken.is(Token::l_bracket)) {
    if (parseSymbolIdList(dims, symbols)) return nullptr;
  }
  if (!consumeIf(Token::arrow)) {
    emitError("expected '->' or '['");
    return nullptr;
  }
  if (!consumeIf(Token::l_paren)) {
    emitError("expected '(' at start of affine map range");
    return nullptr;
  }

  AffineMapParserState affState(dims, symbols);

  SmallVector<AffineExpr *, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseAffineExpr(affState);
    ParseResult res = elt ? ParseSuccess : ParseFailure;
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of 1-d
  // affine expressions)
  if (parseCommaSeparatedList(Token::r_paren, parseElt, false)) return nullptr;

  // Parsed a valid affine map
  auto *affineMap =
      AffineMap::get(affState.dimCount(), affState.symbolCount(), exprs,
                     context);

  return affineMap;
}

//===----------------------------------------------------------------------===//
// Functions
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
  module->functionList.push_back(new ExtFunction(name, type));
  return ParseSuccess;
}


namespace {
/// This class represents the transient parser state for the internals of a
/// function as we are parsing it, e.g. the names for basic blocks.  It handles
/// forward references.
class CFGFunctionParserState {
 public:
  CFGFunction *function;
  llvm::StringMap<std::pair<BasicBlock*, SMLoc>> blocksByName;

  CFGFunctionParserState(CFGFunction *function) : function(function) {}

  /// Get the basic block with the specified name, creating it if it doesn't
  /// already exist.  The location specified is the point of use, which allows
  /// us to diagnose references to blocks that are not defined precisely.
  BasicBlock *getBlockNamed(StringRef name, SMLoc loc) {
    auto &blockAndLoc = blocksByName[name];
    if (!blockAndLoc.first) {
      blockAndLoc.first = new BasicBlock();
      blockAndLoc.second = loc;
    }
    return blockAndLoc.first;
  }
};
} // end anonymous namespace


/// CFG function declarations.
///
///   cfg-func ::= `cfgfunc` function-signature `{` basic-block+ `}`
///
ParseResult Parser::parseCFGFunc() {
  consumeToken(Token::kw_cfgfunc);

  StringRef name;
  FunctionType *type = nullptr;
  if (parseFunctionSignature(name, type))
    return ParseFailure;

  if (!consumeIf(Token::l_brace))
    return emitError("expected '{' in CFG function");

  // Okay, the CFG function signature was parsed correctly, create the function.
  auto function = new CFGFunction(name, type);

  // Make sure we have at least one block.
  if (curToken.is(Token::r_brace))
    return emitError("CFG functions must have at least one basic block");

  CFGFunctionParserState functionState(function);

  // Parse the list of blocks.
  while (!consumeIf(Token::r_brace))
    if (parseBasicBlock(functionState))
      return ParseFailure;

  // Verify that all referenced blocks were defined.  Iteration over a
  // StringMap isn't determinstic, but this is good enough for our purposes.
  for (auto &elt : functionState.blocksByName) {
    auto *bb = elt.second.first;
    if (!bb->getFunction())
      return emitError(elt.second.second,
                       "reference to an undefined basic block '" +
                       elt.first() + "'");
  }

  module->functionList.push_back(function);
  return ParseSuccess;
}

/// Basic block declaration.
///
///   basic-block ::= bb-label instruction* terminator-stmt
///   bb-label    ::= bb-id bb-arg-list? `:`
///   bb-id       ::= bare-id
///   bb-arg-list ::= `(` ssa-id-and-type-list? `)`
///
ParseResult Parser::parseBasicBlock(CFGFunctionParserState &functionState) {
  SMLoc nameLoc = curToken.getLoc();
  auto name = curToken.getSpelling();
  if (!consumeIf(Token::bare_identifier))
    return emitError("expected basic block name");

  auto block = functionState.getBlockNamed(name, nameLoc);

  // If this block has already been parsed, then this is a redefinition with the
  // same block name.
  if (block->getFunction())
    return emitError(nameLoc, "redefinition of block '" + name.str() + "'");

  // Add the block to the function.
  functionState.function->push_back(block);

  // TODO: parse bb argument list.

  if (!consumeIf(Token::colon))
    return emitError("expected ':' after basic block name");

  // Parse the list of operations that make up the body of the block.
  while (curToken.isNot(Token::kw_return, Token::kw_br)) {
    auto *inst = parseCFGOperation(functionState);
    if (!inst)
      return ParseFailure;

    block->getOperations().push_back(inst);
  }

  auto *term = parseTerminator(functionState);
  if (!term)
    return ParseFailure;
  block->setTerminator(term);

  return ParseSuccess;
}


/// Parse the CFG operation.
///
/// TODO(clattner): This is a change from the MLIR spec as written, it is an
/// experiment that will eliminate "builtin" instructions as a thing.
///
///  cfg-operation ::=
///    (ssa-id `=`)? string '(' ssa-use-list? ')' attribute-dict?
///    `:` function-type
///
OperationInst *Parser::
parseCFGOperation(CFGFunctionParserState &functionState) {

  // TODO: parse ssa-id.

  if (curToken.isNot(Token::string))
    return (emitError("expected operation name in quotes"), nullptr);

  auto name = curToken.getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);

  consumeToken(Token::string);

  if (!consumeIf(Token::l_paren))
    return (emitError("expected '(' in operation"), nullptr);

  // TODO: Parse operands.
  if (!consumeIf(Token::r_paren))
    return (emitError("expected '(' in operation"), nullptr);

  auto nameId = Identifier::get(name, context);
  return new OperationInst(nameId);
}


/// Parse the terminator instruction for a basic block.
///
///   terminator-stmt ::= `br` bb-id branch-use-list?
///   branch-use-list ::= `(` ssa-use-and-type-list? `)`
///   terminator-stmt ::=
///     `cond_br` ssa-use `,` bb-id branch-use-list? `,` bb-id branch-use-list?
///   terminator-stmt ::= `return` ssa-use-and-type-list?
///
TerminatorInst *Parser::parseTerminator(CFGFunctionParserState &functionState) {
  switch (curToken.getKind()) {
  default:
    return (emitError("expected terminator at end of basic block"), nullptr);

  case Token::kw_return:
    consumeToken(Token::kw_return);
    return new ReturnInst();

  case Token::kw_br: {
    consumeToken(Token::kw_br);
    auto destBB = functionState.getBlockNamed(curToken.getSpelling(),
                                              curToken.getLoc());
    if (!consumeIf(Token::bare_identifier))
      return (emitError("expected basic block name"), nullptr);
    return new BranchInst(destBB);
  }
  }
}

/// ML function declarations.
///
///   ml-func ::= `mlfunc` ml-func-signature `{` ml-stmt* ml-return-stmt `}`
///
ParseResult Parser::parseMLFunc() {
  consumeToken(Token::kw_mlfunc);

  StringRef name;
  FunctionType *type = nullptr;

  // FIXME: Parse ML function signature (args + types)
  // by passing pointer to SmallVector<identifier> into parseFunctionSignature
  if (parseFunctionSignature(name, type))
    return ParseFailure;

  if (!consumeIf(Token::l_brace))
    return emitError("expected '{' in ML function");

  // Okay, the ML function signature was parsed correctly, create the function.
  auto function = new MLFunction(name, type);

  // Make sure we have at least one statement.
  if (curToken.is(Token::r_brace))
    return emitError("ML function must end with return statement");

  // Parse the list of instructions.
  while (!consumeIf(Token::kw_return)) {
    auto *stmt = parseMLStatement(function);
    if (!stmt)
      return ParseFailure;
    function->stmtList.push_back(stmt);
  }

  // TODO: parse return statement operands
  if (!consumeIf(Token::r_brace))
    emitError("expected '}' in ML function");

  module->functionList.push_back(function);

  return ParseSuccess;
}

/// Parse an MLStatement
/// TODO
///
MLStatement *Parser::parseMLStatement(MLFunction *currentFunction) {
  switch (curToken.getKind()) {
  default:
    return (emitError("expected ML statement"), nullptr);

  // TODO: add parsing of ML statements
  }
}

//===----------------------------------------------------------------------===//
// Top-level entity parsing.
//===----------------------------------------------------------------------===//

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
      if (parseExtFunc()) return nullptr;
      break;

    case Token::kw_cfgfunc:
      if (parseCFGFunc()) return nullptr;
      break;
    case Token::affine_map_identifier:
      if (parseAffineMapDef()) return nullptr;
      break;

    case Token::kw_mlfunc:
      if (parseMLFunc()) return nullptr;
      break;

     // TODO: affine entity declarations, etc.
    }
  }
}

//===----------------------------------------------------------------------===//

void mlir::defaultErrorReporter(const llvm::SMDiagnostic &error) {
  const auto &sourceMgr = *error.getSourceMgr();
  sourceMgr.PrintMessage(error.getLoc(), error.getKind(), error.getMessage());
}

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, it emits diagnostics and returns null.
Module *mlir::parseSourceFile(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                              SMDiagnosticHandlerTy errorReporter) {
  return Parser(sourceMgr, context,
                errorReporter ? std::move(errorReporter) : defaultErrorReporter)
      .parseModule();
}
