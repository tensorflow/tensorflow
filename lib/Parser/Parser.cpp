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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SourceMgr;
using llvm::SMLoc;

/// Simple enum to make code read better in cases that would otherwise return a
/// bool value.  Failure is "true" in a boolean context.
enum ParseResult {
  ParseSuccess,
  ParseFailure
};

namespace {
class Parser;

/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position etc.  The Parser base class provides
/// methods to access this.
class ParserState {
public:
  ParserState(llvm::SourceMgr &sourceMgr, Module *module,
              SMDiagnosticHandlerTy errorReporter)
      : context(module->getContext()), module(module),
        lex(sourceMgr, errorReporter), curToken(lex.lexToken()),
        errorReporter(std::move(errorReporter)) {}

  // A map from affine map identifier to AffineMap.
  llvm::StringMap<AffineMap *> affineMapDefinitions;

private:
  ParserState(const ParserState &) = delete;
  void operator=(const ParserState &) = delete;

  friend class Parser;

  // The context we're parsing into.
  MLIRContext *const context;

  // This is the module we are parsing into.
  Module *const module;

  // The lexer for the source file we're parsing.
  Lexer lex;

  // This is the next token that hasn't been consumed yet.
  Token curToken;

  // The diagnostic error reporter.
  SMDiagnosticHandlerTy const errorReporter;
};
} // end anonymous namespace

namespace {

/// This class implement support for parsing global entities like types and
/// shared entities like SSA names.  It is intended to be subclassed by
/// specialized subparsers that include state, e.g. when a local symbol table.
class Parser {
public:
  Builder builder;

  Parser(ParserState &state) : builder(state.context), state(state) {}

  // Helper methods to get stuff from the parser-global state.
  ParserState &getState() const { return state; }
  MLIRContext *getContext() const { return state.context; }
  Module *getModule() { return state.module; }

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  /// Emit an error and return failure.
  ParseResult emitError(const Twine &message) {
    return emitError(state.curToken.getLoc(), message);
  }
  ParseResult emitError(SMLoc loc, const Twine &message);

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    state.curToken = state.lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::Kind kind) {
    if (state.curToken.isNot(kind))
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

  // Attribute parsing.
  Attribute *parseAttribute();
  ParseResult parseAttributeDict(SmallVectorImpl<NamedAttribute> &attributes);

  // Polyhedral structures.
  AffineMap *parseAffineMapInline();

  // SSA
  ParseResult parseSSAUse();
  ParseResult parseOptionalSSAUseList(Token::Kind endToken);
  ParseResult parseSSAUseAndType();
  ParseResult parseOptionalSSAUseAndTypeList(Token::Kind endToken);

private:
  // The Parser is subclassed and reinstantiated.  Do not add additional
  // non-trivial state here, add it to the ParserState class.
  ParserState &state;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

ParseResult Parser::emitError(SMLoc loc, const Twine &message) {
  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(Token::error))
    return ParseFailure;

  auto &sourceMgr = state.lex.getSourceMgr();
  state.errorReporter(sourceMgr.GetMessage(loc, SourceMgr::DK_Error, message));
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
  if (getToken().is(rightToken)) {
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
  switch (getToken().getKind()) {
  default:
    return (emitError("expected type"), nullptr);
  case Token::kw_bf16:
    consumeToken(Token::kw_bf16);
    return builder.getBF16Type();
  case Token::kw_f16:
    consumeToken(Token::kw_f16);
    return builder.getF16Type();
  case Token::kw_f32:
    consumeToken(Token::kw_f32);
    return builder.getF32Type();
  case Token::kw_f64:
    consumeToken(Token::kw_f64);
    return builder.getF64Type();
  case Token::kw_affineint:
    consumeToken(Token::kw_affineint);
    return builder.getAffineIntType();
  case Token::inttype: {
    auto width = getToken().getIntTypeBitwidth();
    if (!width.hasValue())
      return (emitError("invalid integer width"), nullptr);
    consumeToken(Token::inttype);
    return builder.getIntegerType(width.getValue());
  }
  }
}

/// Parse the element type of a tensor or memref type.
///
///   element-type ::= primitive-type | vector-type
///
Type *Parser::parseElementType() {
  if (getToken().is(Token::kw_vector))
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

  if (getToken().isNot(Token::integer))
    return (emitError("expected dimension size in vector type"), nullptr);

  SmallVector<unsigned, 4> dimensions;
  while (getToken().is(Token::integer)) {
    // Make sure this integer value is in bound and valid.
    auto dimension = getToken().getUnsignedIntegerValue();
    if (!dimension.hasValue())
      return (emitError("invalid dimension in vector type"), nullptr);
    dimensions.push_back(dimension.getValue());

    consumeToken(Token::integer);

    // Make sure we have an 'x' or something like 'xbf32'.
    if (getToken().isNot(Token::bare_identifier) ||
        getTokenSpelling()[0] != 'x')
      return (emitError("expected 'x' in vector dimension list"), nullptr);

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (getTokenSpelling().size() != 1)
      state.lex.resetPointer(getTokenSpelling().data() + 1);

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
  while (getToken().isAny(Token::integer, Token::question)) {
    if (consumeIf(Token::question)) {
      dimensions.push_back(-1);
    } else {
      // Make sure this integer value is in bound and valid.
      auto dimension = getToken().getUnsignedIntegerValue();
      if (!dimension.hasValue() || (int)dimension.getValue() < 0)
        return emitError("invalid dimension");
      dimensions.push_back((int)dimension.getValue());
      consumeToken(Token::integer);
    }

    // Make sure we have an 'x' or something like 'xbf32'.
    if (getToken().isNot(Token::bare_identifier) ||
        getTokenSpelling()[0] != 'x')
      return emitError("expected 'x' in dimension list");

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (getTokenSpelling().size() != 1)
      state.lex.resetPointer(getTokenSpelling().data() + 1);

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
    return builder.getTensorType(elementType);
  return builder.getTensorType(dimensions, elementType);
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
  return builder.getIntegerType(1);
}

/// Parse a function type.
///
///   function-type ::= type-list-parens `->` type-list
///
Type *Parser::parseFunctionType() {
  assert(getToken().is(Token::l_paren));

  SmallVector<Type*, 4> arguments;
  if (parseTypeList(arguments))
    return nullptr;

  if (!consumeIf(Token::arrow))
    return (emitError("expected '->' in function type"), nullptr);

  SmallVector<Type*, 4> results;
  if (parseTypeList(results))
    return nullptr;

  return builder.getFunctionType(arguments, results);
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
  switch (getToken().getKind()) {
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
// Attribute parsing.
//===----------------------------------------------------------------------===//


/// Attribute parsing.
///
///  attribute-value ::= bool-literal
///                    | integer-literal
///                    | float-literal
///                    | string-literal
///                    | `[` (attribute-value (`,` attribute-value)*)? `]`
///
Attribute *Parser::parseAttribute() {
  switch (getToken().getKind()) {
  case Token::kw_true:
    consumeToken(Token::kw_true);
    return builder.getBoolAttr(true);
  case Token::kw_false:
    consumeToken(Token::kw_false);
    return builder.getBoolAttr(false);

  case Token::integer: {
    auto val = getToken().getUInt64IntegerValue();
    if (!val.hasValue() || (int64_t)val.getValue() < 0)
      return (emitError("integer too large for attribute"), nullptr);
    consumeToken(Token::integer);
    return builder.getIntegerAttr((int64_t)val.getValue());
  }

  case Token::minus: {
    consumeToken(Token::minus);
    if (getToken().is(Token::integer)) {
      auto val = getToken().getUInt64IntegerValue();
      if (!val.hasValue() || (int64_t)-val.getValue() >= 0)
        return (emitError("integer too large for attribute"), nullptr);
      consumeToken(Token::integer);
      return builder.getIntegerAttr((int64_t)-val.getValue());
    }

    return (emitError("expected constant integer or floating point value"),
            nullptr);
  }

  case Token::string: {
    auto val = getToken().getStringValue();
    consumeToken(Token::string);
    return builder.getStringAttr(val);
  }

  case Token::l_bracket: {
    consumeToken(Token::l_bracket);
    SmallVector<Attribute*, 4> elements;

    auto parseElt = [&]() -> ParseResult {
      elements.push_back(parseAttribute());
      return elements.back() ? ParseSuccess : ParseFailure;
    };

    if (parseCommaSeparatedList(Token::r_bracket, parseElt))
      return nullptr;
    return builder.getArrayAttr(elements);
  }
  default:
    // TODO: Handle floating point.
    return (emitError("expected constant attribute value"), nullptr);
  }
}

/// Attribute dictionary.
///
///   attribute-dict ::= `{` `}`
///                    | `{` attribute-entry (`,` attribute-entry)* `}`
///   attribute-entry ::= bare-id `:` attribute-value
///
ParseResult Parser::parseAttributeDict(
    SmallVectorImpl<NamedAttribute> &attributes) {
  consumeToken(Token::l_brace);

  auto parseElt = [&]() -> ParseResult {
    // We allow keywords as attribute names.
    if (getToken().isNot(Token::bare_identifier, Token::inttype) &&
        !getToken().isKeyword())
      return emitError("expected attribute name");
    auto nameId = builder.getIdentifier(getTokenSpelling());
    consumeToken();

    if (!consumeIf(Token::colon))
      return emitError("expected ':' in attribute list");

    auto attr = parseAttribute();
    if (!attr) return ParseFailure;

    attributes.push_back({nameId, attr});
    return ParseSuccess;
  };

  if (parseCommaSeparatedList(Token::r_brace, parseElt))
    return ParseFailure;

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Polyhedral structures.
//===----------------------------------------------------------------------===//

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false in
/// the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

namespace {
/// This is a specialized parser for AffineMap's, maintaining the state
/// transient to their bodies.
class AffineMapParser : public Parser {
public:
  explicit AffineMapParser(ParserState &state) : Parser(state) {}

  AffineMap *parseAffineMapInline();

private:
  unsigned getNumDims() const { return dims.size(); }
  unsigned getNumSymbols() const { return symbols.size(); }

  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList();
  ParseResult parseSymbolIdList();
  ParseResult parseDimOrSymbolId(bool isDim);

  AffineExpr *parseAffineExpr();
  AffineExpr *parseParentheticalExpr();
  AffineExpr *parseNegateExpression(AffineExpr *lhs);
  AffineExpr *parseIntegerExpr();
  AffineExpr *parseBareIdExpr();

  AffineExpr *getBinaryAffineOpExpr(AffineHighPrecOp op, AffineExpr *lhs,
                                    AffineExpr *rhs);
  AffineExpr *getBinaryAffineOpExpr(AffineLowPrecOp op, AffineExpr *lhs,
                                    AffineExpr *rhs);
  AffineExpr *parseAffineOperandExpr(AffineExpr *lhs);
  AffineExpr *parseAffineLowPrecOpExpr(AffineExpr *llhs,
                                       AffineLowPrecOp llhsOp);
  AffineExpr *parseAffineHighPrecOpExpr(AffineExpr *llhs,
                                        AffineHighPrecOp llhsOp);

private:
  // TODO(bondhugula): could just use an vector/ArrayRef and scan the numbers.
  llvm::StringMap<unsigned> dims;
  llvm::StringMap<unsigned> symbols;
};
} // end anonymous namespace

/// Create an affine binary high precedence op expression (mul's, div's, mod)
AffineExpr *AffineMapParser::getBinaryAffineOpExpr(AffineHighPrecOp op,
                                                   AffineExpr *lhs,
                                                   AffineExpr *rhs) {
  switch (op) {
  case Mul:
    if (!lhs->isSymbolic() && !rhs->isSymbolic()) {
      emitError("non-affine expression: at least one of the multiply "
                "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return builder.getMulExpr(lhs, rhs);
  case FloorDiv:
    if (!rhs->isSymbolic()) {
      emitError("non-affine expression: right operand of floordiv "
                "has to be either a constant or symbolic");
      return nullptr;
    }
    return builder.getFloorDivExpr(lhs, rhs);
  case CeilDiv:
    if (!rhs->isSymbolic()) {
      emitError("non-affine expression: right operand of ceildiv "
                "has to be either a constant or symbolic");
      return nullptr;
    }
    return builder.getCeilDivExpr(lhs, rhs);
  case Mod:
    if (!rhs->isSymbolic()) {
      emitError("non-affine expression: right operand of mod "
                "has to be either a constant or symbolic");
      return nullptr;
    }
    return builder.getModExpr(lhs, rhs);
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
}

/// Create an affine binary low precedence op expression (add, sub).
AffineExpr *AffineMapParser::getBinaryAffineOpExpr(AffineLowPrecOp op,
                                                   AffineExpr *lhs,
                                                   AffineExpr *rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return builder.getAddExpr(lhs, rhs);
  case AffineLowPrecOp::Sub:
    return builder.getSubExpr(lhs, rhs);
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
}

/// Consume this token if it is a lower precedence affine op (there are only two
/// precedence levels).
AffineLowPrecOp AffineMapParser::consumeIfLowPrecOp() {
  switch (getToken().getKind()) {
  case Token::plus:
    consumeToken(Token::plus);
    return AffineLowPrecOp::Add;
  case Token::minus:
    consumeToken(Token::minus);
    return AffineLowPrecOp::Sub;
  default:
    return AffineLowPrecOp::LNoOp;
  }
}

/// Consume this token if it is a higher precedence affine op (there are only
/// two precedence levels)
AffineHighPrecOp AffineMapParser::consumeIfHighPrecOp() {
  switch (getToken().getKind()) {
  case Token::star:
    consumeToken(Token::star);
    return Mul;
  case Token::kw_floordiv:
    consumeToken(Token::kw_floordiv);
    return FloorDiv;
  case Token::kw_ceildiv:
    consumeToken(Token::kw_ceildiv);
    return CeilDiv;
  case Token::kw_mod:
    consumeToken(Token::kw_mod);
    return Mod;
  default:
    return HNoOp;
  }
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a AffineHighPrecOp (mul, div, mod).
/// All affine binary ops are left associative.
/// Given llhs, returns (llhs llhsOp lhs) op rhs, or (lhs op rhs) if llhs is
/// null. If no rhs can be found, returns (llhs llhsOp lhs) or lhs if llhs is
/// null.
AffineExpr *
AffineMapParser::parseAffineHighPrecOpExpr(AffineExpr *llhs,
                                           AffineHighPrecOp llhsOp) {
  AffineExpr *lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      AffineExpr *expr = getBinaryAffineOpExpr(llhsOp, llhs, lhs);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(expr, op);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(lhs, op);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getBinaryAffineOpExpr(llhsOp, llhs, lhs);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
AffineExpr *AffineMapParser::parseParentheticalExpr() {
  if (!consumeIf(Token::l_paren))
    return (emitError("expected '('"), nullptr);
  if (getToken().is(Token::r_paren))
    return (emitError("no expression inside parentheses"), nullptr);
  auto *expr = parseAffineExpr();
  if (!expr)
    return nullptr;
  if (!consumeIf(Token::r_paren))
    return (emitError("expected ')'"), nullptr);
  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
AffineExpr *AffineMapParser::parseNegateExpression(AffineExpr *lhs) {
  if (!consumeIf(Token::minus))
    return (emitError("expected '-'"), nullptr);

  AffineExpr *operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand)
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return (emitError("missing operand of negation"), nullptr);
  auto *minusOne = builder.getConstantExpr(-1);
  return builder.getMulExpr(minusOne, operand);
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr *AffineMapParser::parseBareIdExpr() {
  if (getToken().isNot(Token::bare_identifier))
    return (emitError("expected bare identifier"), nullptr);

  StringRef sRef = getTokenSpelling();
  if (dims.count(sRef)) {
    consumeToken(Token::bare_identifier);
    return builder.getDimExpr(dims.lookup(sRef));
  }
  if (symbols.count(sRef)) {
    consumeToken(Token::bare_identifier);
    return builder.getSymbolExpr(symbols.lookup(sRef));
  }
  return (emitError("identifier is neither dimensional nor symbolic"), nullptr);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr *AffineMapParser::parseIntegerExpr() {
  // No need to handle negative numbers separately here. They are naturally
  // handled via the unary negation operator, although (FIXME) MININT_64 still
  // not correctly handled.
  if (getToken().isNot(Token::integer))
    return (emitError("expected integer"), nullptr);

  auto val = getToken().getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0) {
    return (emitError("constant too large for affineint"), nullptr);
  }
  consumeToken(Token::integer);
  return builder.getConstantExpr((int64_t)val.getValue());
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and -l
//  are valid operands that will be parsed by this function.
AffineExpr *AffineMapParser::parseAffineOperandExpr(AffineExpr *lhs) {
  switch (getToken().getKind()) {
  case Token::bare_identifier:
    return parseBareIdExpr();
  case Token::integer:
    return parseIntegerExpr();
  case Token::l_paren:
    return parseParentheticalExpr();
  case Token::minus:
    return parseNegateExpression(lhs);
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::kw_mod:
  case Token::plus:
  case Token::star:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("expected affine expression");
    return nullptr;
  }
}

/// Parse affine expressions that are bare-id's, integer constants,
/// parenthetical affine expressions, and affine op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. Mul, floordiv,
/// ceildiv, and mod are at the same higher precedence level. Negation has
/// higher precedence than any binary op.
///
/// llhs: the affine expression appearing on the left of the one being parsed.
/// This function will return ((llhs llhsOp lhs) op rhs) if llhs is non null,
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned if
/// llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4, where (e2*e3)
/// will be parsed using parseAffineHighPrecOpExpr().
AffineExpr *AffineMapParser::parseAffineLowPrecOpExpr(AffineExpr *llhs,
                                                      AffineLowPrecOp llhsOp) {
  AffineExpr *lhs;
  if (!(lhs = parseAffineOperandExpr(llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      AffineExpr *sum = getBinaryAffineOpExpr(llhsOp, llhs, lhs);
      return parseAffineLowPrecOpExpr(sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(lhs, lOp);
  }
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr *highRes = parseAffineHighPrecOpExpr(lhs, hOp);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr *expr =
        llhs ? getBinaryAffineOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getBinaryAffineOpExpr(llhsOp, llhs, lhs);
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression.
///  affine-expr ::= `(` affine-expr `)`
///                | `-` affine-expr
///                | affine-expr `+` affine-expr
///                | affine-expr `-` affine-expr
///                | affine-expr `*` affine-expr
///                | affine-expr `floordiv` affine-expr
///                | affine-expr `ceildiv` affine-expr
///                | affine-expr `mod` affine-expr
///                | bare-id
///                | integer-literal
///
/// Additional conditions are checked depending on the production. For eg., one
/// of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
AffineExpr *AffineMapParser::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual expressions
/// of the affine map. Update our state to store the dimensional/symbolic
/// identifier. 'dim': whether it's the dim list or symbol list that is being
/// parsed.
ParseResult AffineMapParser::parseDimOrSymbolId(bool isDim) {
  if (getToken().isNot(Token::bare_identifier))
    return emitError("expected bare identifier");
  auto sRef = getTokenSpelling();
  consumeToken(Token::bare_identifier);
  if (dims.count(sRef))
    return emitError("dimensional identifier name reused");
  if (symbols.count(sRef))
    return emitError("symbolic identifier name reused");
  if (isDim)
    dims.insert({sRef, dims.size()});
  else
    symbols.insert({sRef, symbols.size()});
  return ParseSuccess;
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult AffineMapParser::parseSymbolIdList() {
  if (!consumeIf(Token::l_bracket))
    return emitError("expected '['");

  auto parseElt = [&]() -> ParseResult { return parseDimOrSymbolId(false); };
  return parseCommaSeparatedList(Token::r_bracket, parseElt);
}

/// Parse the list of dimensional identifiers to an affine map.
ParseResult AffineMapParser::parseDimIdList() {
  if (!consumeIf(Token::l_paren))
    return emitError("expected '(' at start of dimensional identifiers list");

  auto parseElt = [&]() -> ParseResult { return parseDimOrSymbolId(true); };
  return parseCommaSeparatedList(Token::r_paren, parseElt);
}

/// Parse an affine map definition.
///
///  affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        (`size` `(` dim-size (`,` dim-size)* `)`)?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
//  TODO(bondhugula): parse range size information.
AffineMap *AffineMapParser::parseAffineMapInline() {
  // List of dimensional identifiers.
  if (parseDimIdList())
    return nullptr;

  // Symbols are optional.
  if (getToken().is(Token::l_bracket)) {
    if (parseSymbolIdList())
      return nullptr;
  }
  if (!consumeIf(Token::arrow)) {
    return (emitError("expected '->' or '['"), nullptr);
  }
  if (!consumeIf(Token::l_paren)) {
    emitError("expected '(' at start of affine map range");
    return nullptr;
  }

  SmallVector<AffineExpr *, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto *elt = parseAffineExpr();
    ParseResult res = elt ? ParseSuccess : ParseFailure;
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of 1-d
  // affine expressions); the list cannot be empty.
  // Grammar: multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
  if (parseCommaSeparatedList(Token::r_paren, parseElt, false))
    return nullptr;

  // Parsed a valid affine map.
  return builder.getAffineMap(dims.size(), symbols.size(), exprs);
}

AffineMap *Parser::parseAffineMapInline() {
  return AffineMapParser(state).parseAffineMapInline();
}

//===----------------------------------------------------------------------===//
// SSA
//===----------------------------------------------------------------------===//

/// Parse a SSA operand for an instruction or statement.
///
///   ssa-use ::= ssa-id | ssa-constant
///
ParseResult Parser::parseSSAUse() {
  if (getToken().is(Token::percent_identifier)) {
    StringRef name = getTokenSpelling().drop_front();
    consumeToken(Token::percent_identifier);
    // TODO: Return this use.
    (void)name;
    return ParseSuccess;
  }

  // TODO: Parse SSA constants.

  return emitError("expected SSA operand");
}

/// Parse a (possibly empty) list of SSA operands.
///
///   ssa-use-list ::= ssa-use (`,` ssa-use)*
///   ssa-use-list-opt ::= ssa-use-list?
///
ParseResult Parser::parseOptionalSSAUseList(Token::Kind endToken) {
  // TODO: Build and return this.
  return parseCommaSeparatedList(
      endToken, [&]() -> ParseResult { return parseSSAUse(); });
}

/// Parse an SSA use with an associated type.
///
///   ssa-use-and-type ::= ssa-use `:` type
ParseResult Parser::parseSSAUseAndType() {
  if (parseSSAUse())
    return ParseFailure;

  if (!consumeIf(Token::colon))
    return emitError("expected ':' and type for SSA operand");

  if (!parseType())
    return ParseFailure;

  return ParseSuccess;
}

/// Parse a (possibly empty) list of SSA operands with types.
///
///   ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
///
ParseResult Parser::parseOptionalSSAUseAndTypeList(Token::Kind endToken) {
  // TODO: Build and return this.
  return parseCommaSeparatedList(
      endToken, [&]() -> ParseResult { return parseSSAUseAndType(); });
}


//===----------------------------------------------------------------------===//
// CFG Functions
//===----------------------------------------------------------------------===//

namespace {
/// This is a specialized parser for CFGFunction's, maintaining the state
/// transient to their bodies.
class CFGFunctionParser : public Parser {
public:
  CFGFunctionParser(ParserState &state, CFGFunction *function)
      : Parser(state), function(function), builder(function) {}

  ParseResult parseFunctionBody();

private:
  CFGFunction *function;
  llvm::StringMap<std::pair<BasicBlock*, SMLoc>> blocksByName;

  /// This builder intentionally shadows the builder in the base class, with a
  /// more specific builder type.
  CFGFuncBuilder builder;

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

  ParseResult parseBasicBlock();
  OperationInst *parseCFGOperation();
  TerminatorInst *parseTerminator();
};
} // end anonymous namespace

ParseResult CFGFunctionParser::parseFunctionBody() {
  if (!consumeIf(Token::l_brace))
    return emitError("expected '{' in CFG function");

  // Make sure we have at least one block.
  if (getToken().is(Token::r_brace))
    return emitError("CFG functions must have at least one basic block");

  // Parse the list of blocks.
  while (!consumeIf(Token::r_brace))
    if (parseBasicBlock())
      return ParseFailure;

  // Verify that all referenced blocks were defined.  Iteration over a
  // StringMap isn't determinstic, but this is good enough for our purposes.
  for (auto &elt : blocksByName) {
    auto *bb = elt.second.first;
    if (!bb->getFunction())
      return emitError(elt.second.second,
                       "reference to an undefined basic block '" +
                       elt.first() + "'");
  }

  getModule()->functionList.push_back(function);
  return ParseSuccess;
}

/// Basic block declaration.
///
///   basic-block ::= bb-label instruction* terminator-stmt
///   bb-label    ::= bb-id bb-arg-list? `:`
///   bb-id       ::= bare-id
///   bb-arg-list ::= `(` ssa-id-and-type-list? `)`
///
ParseResult CFGFunctionParser::parseBasicBlock() {
  SMLoc nameLoc = getToken().getLoc();
  auto name = getTokenSpelling();
  if (!consumeIf(Token::bare_identifier))
    return emitError("expected basic block name");

  auto *block = getBlockNamed(name, nameLoc);

  // If this block has already been parsed, then this is a redefinition with the
  // same block name.
  if (block->getFunction())
    return emitError(nameLoc, "redefinition of block '" + name.str() + "'");

  // Add the block to the function.
  function->push_back(block);

  // If an argument list is present, parse it.
  if (consumeIf(Token::l_paren)) {
    if (parseOptionalSSAUseAndTypeList(Token::r_paren))
      return ParseFailure;

    // TODO: attach it.
  }

  if (!consumeIf(Token::colon))
    return emitError("expected ':' after basic block name");

  // Set the insertion point to the block we want to insert new operations into.
  builder.setInsertionPoint(block);

  // Parse the list of operations that make up the body of the block.
  while (getToken().isNot(Token::kw_return, Token::kw_br)) {
    auto loc = getToken().getLoc();
    auto *inst = parseCFGOperation();
    if (!inst)
      return ParseFailure;

    // We just parsed an operation.  If it is a recognized one, verify that it
    // is structurally as we expect.  If not, produce an error with a reasonable
    // source location.
    if (auto *opInfo = inst->getAbstractOperation(builder.getContext()))
      if (auto error = opInfo->verifyInvariants(inst))
        return emitError(loc, error);
  }

  auto *term = parseTerminator();
  if (!term)
    return ParseFailure;

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
OperationInst *CFGFunctionParser::parseCFGOperation() {
  StringRef resultID;
  if (getToken().is(Token::percent_identifier)) {
    resultID = getTokenSpelling().drop_front();
    consumeToken();
    if (!consumeIf(Token::equal))
      return (emitError("expected '=' after SSA name"), nullptr);
  }

  if (getToken().isNot(Token::string))
    return (emitError("expected operation name in quotes"), nullptr);

  auto name = getToken().getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);

  consumeToken(Token::string);

  if (!consumeIf(Token::l_paren))
    return (emitError("expected '(' to start operand list"), nullptr);

  // Parse the operand list.
  parseOptionalSSAUseList(Token::r_paren);

  SmallVector<NamedAttribute, 4> attributes;
  if (getToken().is(Token::l_brace)) {
    if (parseAttributeDict(attributes))
      return nullptr;
  }

  // TODO: Don't drop result name and operand names on the floor.
  auto nameId = builder.getIdentifier(name);
  return builder.createOperation(nameId, attributes);
}

/// Parse the terminator instruction for a basic block.
///
///   terminator-stmt ::= `br` bb-id branch-use-list?
///   branch-use-list ::= `(` ssa-use-and-type-list? `)`
///   terminator-stmt ::=
///     `cond_br` ssa-use `,` bb-id branch-use-list? `,` bb-id branch-use-list?
///   terminator-stmt ::= `return` ssa-use-and-type-list?
///
TerminatorInst *CFGFunctionParser::parseTerminator() {
  switch (getToken().getKind()) {
  default:
    return (emitError("expected terminator at end of basic block"), nullptr);

  case Token::kw_return:
    consumeToken(Token::kw_return);
    return builder.createReturnInst();

  case Token::kw_br: {
    consumeToken(Token::kw_br);
    auto destBB = getBlockNamed(getTokenSpelling(), getToken().getLoc());
    if (!consumeIf(Token::bare_identifier))
      return (emitError("expected basic block name"), nullptr);
    return builder.createBranchInst(destBB);
  }
    // TODO: cond_br.
  }
}

//===----------------------------------------------------------------------===//
// ML Functions
//===----------------------------------------------------------------------===//

namespace {
/// Refined parser for MLFunction bodies.
class MLFunctionParser : public Parser {
public:
  MLFunction *function;

  /// This builder intentionally shadows the builder in the base class, with a
  /// more specific builder type.
  // TODO: MLFuncBuilder builder;

  MLFunctionParser(ParserState &state, MLFunction *function)
      : Parser(state), function(function) {}

  ParseResult parseFunctionBody();
  Statement *parseStatement(ParentType parent);
  ForStmt *parseForStmt(ParentType parent);
  IfStmt *parseIfStmt(ParentType parent);
  ParseResult parseNestedStatements(NodeStmt *parent);
};
} // end anonymous namespace

ParseResult MLFunctionParser::parseFunctionBody() {
  if (!consumeIf(Token::l_brace))
    return emitError("expected '{' in ML function");

  // Make sure we have at least one statement.
  if (getToken().is(Token::r_brace))
    return emitError("ML function must end with return statement");

  // Parse the list of instructions.
  while (!consumeIf(Token::kw_return)) {
    auto *stmt = parseStatement(function);
    if (!stmt)
      return ParseFailure;
    function->stmtList.push_back(stmt);
  }

  // TODO: parse return statement operands
  if (!consumeIf(Token::r_brace))
    emitError("expected '}' in ML function");

  getModule()->functionList.push_back(function);

  return ParseSuccess;
}

/// Statement.
///
///    ml-stmt ::= instruction | ml-for-stmt | ml-if-stmt
///
/// TODO: fix terminology in MLSpec document. ML functions
/// contain operation statements, not instructions.
///
Statement *MLFunctionParser::parseStatement(ParentType parent) {
  switch (getToken().getKind()) {
  default:
    //TODO: parse OperationStmt
    return (emitError("expected statement"), nullptr);

  case Token::kw_for:
    return parseForStmt(parent);

  case Token::kw_if:
    return parseIfStmt(parent);
  }
}

/// For statement.
///
///    ml-for-stmt ::= `for` ssa-id `=` lower-bound `to` upper-bound
///                   (`step` integer-literal)? `{` ml-stmt* `}`
///
ForStmt *MLFunctionParser::parseForStmt(ParentType parent) {
  consumeToken(Token::kw_for);

  //TODO: parse loop header
  ForStmt *stmt = new ForStmt(parent);
  if (parseNestedStatements(stmt)) {
    delete stmt;
    return nullptr;
  }
  return stmt;
}

/// If statement.
///
///   ml-if-head ::= `if` ml-if-cond `{` ml-stmt* `}`
///               | ml-if-head `else` `if` ml-if-cond `{` ml-stmt* `}`
///   ml-if-stmt ::= ml-if-head
///               | ml-if-head `else` `{` ml-stmt* `}`
///
IfStmt *
MLFunctionParser::parseIfStmt(PointerUnion<MLFunction *, NodeStmt *> parent) {
  consumeToken(Token::kw_if);

  //TODO: parse condition
  IfStmt *stmt = new IfStmt(parent);
  if (parseNestedStatements(stmt)) {
    delete stmt;
    return nullptr;
  }

  int clauseNum = 0;
  while (consumeIf(Token::kw_else)) {
    if (consumeIf(Token::kw_if)) {
       //TODO: parse condition
    }
    ElseClause * clause = new ElseClause(stmt, clauseNum);
    ++clauseNum;
    if (parseNestedStatements(clause)) {
      delete clause;
      return nullptr;
    }
  }

  return stmt;
}

///
/// Parse `{` ml-stmt* `}`
///
ParseResult MLFunctionParser::parseNestedStatements(NodeStmt *parent) {
  if (!consumeIf(Token::l_brace))
    return emitError("expected '{' before statement list");

  if (consumeIf(Token::r_brace)) {
    // TODO: parse OperationStmt
    return ParseSuccess;
  }

  while (!consumeIf(Token::r_brace)) {
    auto *stmt = parseStatement(parent);
    if (!stmt)
      return ParseFailure;
    parent->children.push_back(stmt);
  }

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Top-level entity parsing.
//===----------------------------------------------------------------------===//

namespace {
/// This parser handles entities that are only valid at the top level of the
/// file.
class ModuleParser : public Parser {
public:
  explicit ModuleParser(ParserState &state) : Parser(state) {}

  ParseResult parseModule();

private:
  ParseResult parseAffineMapDef();

  // Functions.
  ParseResult parseFunctionSignature(StringRef &name, FunctionType *&type);
  ParseResult parseExtFunc();
  ParseResult parseCFGFunc();
  ParseResult parseMLFunc();
};
} // end anonymous namespace

/// Affine map declaration.
///
///   affine-map-def ::= affine-map-id `=` affine-map-inline
///
ParseResult ModuleParser::parseAffineMapDef() {
  assert(getToken().is(Token::hash_identifier));

  StringRef affineMapId = getTokenSpelling().drop_front();

  // Check for redefinitions.
  auto *&entry = getState().affineMapDefinitions[affineMapId];
  if (entry)
    return emitError("redefinition of affine map id '" + affineMapId + "'");

  consumeToken(Token::hash_identifier);

  // Parse the '='
  if (!consumeIf(Token::equal))
    return emitError("expected '=' in affine map outlined definition");

  entry = parseAffineMapInline();
  if (!entry)
    return ParseFailure;

  getModule()->affineMapList.push_back(entry);
  return ParseSuccess;
}

/// Parse a function signature, starting with a name and including the parameter
/// list.
///
///   argument-list ::= type (`,` type)* | /*empty*/
///   function-signature ::= function-id `(` argument-list `)` (`->` type-list)?
///
ParseResult ModuleParser::parseFunctionSignature(StringRef &name,
                                                 FunctionType *&type) {
  if (getToken().isNot(Token::at_identifier))
    return emitError("expected a function identifier like '@foo'");

  name = getTokenSpelling().drop_front();
  consumeToken(Token::at_identifier);

  if (getToken().isNot(Token::l_paren))
    return emitError("expected '(' in function signature");

  SmallVector<Type *, 4> arguments;
  if (parseTypeList(arguments))
    return ParseFailure;

  // Parse the return type if present.
  SmallVector<Type *, 4> results;
  if (consumeIf(Token::arrow)) {
    if (parseTypeList(results))
      return ParseFailure;
  }
  type = builder.getFunctionType(arguments, results);
  return ParseSuccess;
}

/// External function declarations.
///
///   ext-func ::= `extfunc` function-signature
///
ParseResult ModuleParser::parseExtFunc() {
  consumeToken(Token::kw_extfunc);

  StringRef name;
  FunctionType *type = nullptr;
  if (parseFunctionSignature(name, type))
    return ParseFailure;

  // Okay, the external function definition was parsed correctly.
  getModule()->functionList.push_back(new ExtFunction(name, type));
  return ParseSuccess;
}

/// CFG function declarations.
///
///   cfg-func ::= `cfgfunc` function-signature `{` basic-block+ `}`
///
ParseResult ModuleParser::parseCFGFunc() {
  consumeToken(Token::kw_cfgfunc);

  StringRef name;
  FunctionType *type = nullptr;
  if (parseFunctionSignature(name, type))
    return ParseFailure;

  // Okay, the CFG function signature was parsed correctly, create the function.
  auto function = new CFGFunction(name, type);

  return CFGFunctionParser(getState(), function).parseFunctionBody();
}

/// ML function declarations.
///
///   ml-func ::= `mlfunc` ml-func-signature `{` ml-stmt* ml-return-stmt `}`
///
ParseResult ModuleParser::parseMLFunc() {
  consumeToken(Token::kw_mlfunc);

  StringRef name;
  FunctionType *type = nullptr;

  // FIXME: Parse ML function signature (args + types)
  // by passing pointer to SmallVector<identifier> into parseFunctionSignature
  if (parseFunctionSignature(name, type))
    return ParseFailure;

  // Okay, the ML function signature was parsed correctly, create the function.
  auto function = new MLFunction(name, type);

  return MLFunctionParser(getState(), function).parseFunctionBody();
}

/// This is the top-level module parser.
ParseResult ModuleParser::parseModule() {
  while (1) {
    switch (getToken().getKind()) {
    default:
      emitError("expected a top level entity");
      return ParseFailure;

      // If we got to the end of the file, then we're done.
    case Token::eof:
      return ParseSuccess;

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  Someday we could introduce error recovery if there was demand for
    // it.
    case Token::error:
      return ParseFailure;

    case Token::hash_identifier:
      if (parseAffineMapDef())
        return ParseFailure;
      break;

    case Token::kw_extfunc:
      if (parseExtFunc())
        return ParseFailure;
      break;

    case Token::kw_cfgfunc:
      if (parseCFGFunc())
        return ParseFailure;
      break;

    case Token::kw_mlfunc:
      if (parseMLFunc())
        return ParseFailure;
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
  // This is the result module we are parsing into.
  std::unique_ptr<Module> module(new Module(context));

  ParserState state(sourceMgr, module.get(),
                    errorReporter ? std::move(errorReporter)
                                  : defaultErrorReporter);
  if (ModuleParser(state).parseModule())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by the
  // verifier.
  module->verify();
  return module.release();
}
