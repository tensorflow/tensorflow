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
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSet.h"
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
  llvm::StringMap<AffineMap*> affineMapDefinitions;

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

  // Binary affine op parsing
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

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

  // Parsing identifiers' lists for polyhedral structures.
  ParseResult parseDimIdList(AffineMapParserState &state);
  ParseResult parseSymbolIdList(AffineMapParserState &state);
  ParseResult parseDimOrSymbolId(AffineMapParserState &state, bool dim);

  // Polyhedral structures.
  ParseResult parseAffineMapDef();
  AffineMap *parseAffineMapInline(StringRef mapId);
  AffineExpr *parseAffineExpr(const AffineMapParserState &state);

  AffineExpr *parseParentheticalExpr(const AffineMapParserState &state);
  AffineExpr *parseIntegerExpr(const AffineMapParserState &state);
  AffineExpr *parseBareIdExpr(const AffineMapParserState &state);

  static AffineBinaryOpExpr *getBinaryAffineOpExpr(AffineHighPrecOp op,
                                                   AffineExpr *lhs,
                                                   AffineExpr *rhs,
                                                   MLIRContext *context);
  static AffineBinaryOpExpr *getBinaryAffineOpExpr(AffineLowPrecOp op,
                                                   AffineExpr *lhs,
                                                   AffineExpr *rhs,
                                                   MLIRContext *context);
  ParseResult parseAffineOperandExpr(const AffineMapParserState &state,
                                     AffineExpr *&result);
  ParseResult parseAffineLowPrecOpExpr(AffineExpr *llhs, AffineLowPrecOp llhsOp,
                                       const AffineMapParserState &state,
                                       AffineExpr *&result);
  ParseResult parseAffineHighPrecOpExpr(AffineExpr *llhs,
                                        AffineHighPrecOp llhsOp,
                                        const AffineMapParserState &state,
                                        AffineExpr *&result);

  // SSA
  ParseResult parseSSAUse();
  ParseResult parseOptionalSSAUseList(Token::Kind endToken);
  ParseResult parseSSAUseAndType();
  ParseResult parseOptionalSSAUseAndTypeList(Token::Kind endToken);

  // Functions.
  ParseResult parseFunctionSignature(StringRef &name, FunctionType *&type);
  ParseResult parseExtFunc();
  ParseResult parseCFGFunc();
  ParseResult parseBasicBlock(CFGFunctionParserState &functionState);
  Statement *parseStatement(ParentType parent);

  OperationInst *parseCFGOperation(CFGFunctionParserState &functionState);
  TerminatorInst *parseTerminator(CFGFunctionParserState &functionState);

  ParseResult parseMLFunc();
  ForStmt *parseForStmt(ParentType parent);
  IfStmt *parseIfStmt(ParentType parent);
  ParseResult parseNestedStatements(NodeStmt *parent);
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
   explicit AffineMapParserState() {}

   void addDim(StringRef sRef) { dims.insert({sRef, dims.size()}); }
   void addSymbol(StringRef sRef) { symbols.insert({sRef, symbols.size()}); }

   unsigned getNumDims() const { return dims.size(); }
   unsigned getNumSymbols() const { return symbols.size(); }

   // TODO(bondhugula): could just use an vector/ArrayRef and scan the numbers.
   const llvm::StringMap<unsigned> &getDims() const { return dims; }
   const llvm::StringMap<unsigned> &getSymbols() const { return symbols; }

 private:
   llvm::StringMap<unsigned> dims;
   llvm::StringMap<unsigned> symbols;
};
}  // end anonymous namespace

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
  switch (curToken.getKind()) {
  case Token::kw_true:
    consumeToken(Token::kw_true);
    return BoolAttr::get(true, context);
  case Token::kw_false:
    consumeToken(Token::kw_false);
    return BoolAttr::get(false, context);

  case Token::integer: {
    auto val = curToken.getUInt64IntegerValue();
    if (!val.hasValue() || (int64_t)val.getValue() < 0)
      return (emitError("integer too large for attribute"), nullptr);
    consumeToken(Token::integer);
    return IntegerAttr::get((int64_t)val.getValue(), context);
  }

  case Token::minus: {
    consumeToken(Token::minus);
    if (curToken.is(Token::integer)) {
      auto val = curToken.getUInt64IntegerValue();
      if (!val.hasValue() || (int64_t)-val.getValue() >= 0)
        return (emitError("integer too large for attribute"), nullptr);
      consumeToken(Token::integer);
      return IntegerAttr::get((int64_t)-val.getValue(), context);
    }

    return (emitError("expected constant integer or floating point value"),
            nullptr);
  }

  case Token::string: {
    auto val = curToken.getStringValue();
    consumeToken(Token::string);
    return StringAttr::get(val, context);
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
    return ArrayAttr::get(elements, context);
  }
  default:
    // TODO: Handle floating point.
    return (emitError("expected constant attribute value"), nullptr);
  }
}


/// Attribute dictionary.
///
///  attribute-dict ::= `{` `}`
///                   | `{` attribute-entry (`,` attribute-entry)* `}`
///  attribute-entry ::= bare-id `:` attribute-value
///
ParseResult Parser::parseAttributeDict(
    SmallVectorImpl<NamedAttribute> &attributes) {
  consumeToken(Token::l_brace);

  auto parseElt = [&]() -> ParseResult {
    // We allow keywords as attribute names.
    if (curToken.isNot(Token::bare_identifier, Token::inttype) &&
        !curToken.isKeyword())
      return emitError("expected attribute name");
    auto nameId = Identifier::get(curToken.getSpelling(), context);
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

/// Affine map declaration.
///
///  affine-map-def ::= affine-map-id `=` affine-map-inline
///
ParseResult Parser::parseAffineMapDef() {
  assert(curToken.is(Token::hash_identifier));

  StringRef affineMapId = curToken.getSpelling().drop_front();

  // Check for redefinitions.
  auto *&entry = affineMapDefinitions[affineMapId];
  if (entry)
    return emitError("redefinition of affine map id '" + affineMapId + "'");

  consumeToken(Token::hash_identifier);

  // Parse the '='
  if (!consumeIf(Token::equal))
    return emitError("expected '=' in affine map outlined definition");

  entry = parseAffineMapInline(affineMapId);
  if (!entry)
    return ParseFailure;

  module->affineMapList.push_back(entry);
  return ParseSuccess;
}

/// Create an affine op expression
AffineBinaryOpExpr *Parser::getBinaryAffineOpExpr(AffineHighPrecOp op,
                                                  AffineExpr *lhs,
                                                  AffineExpr *rhs,
                                                  MLIRContext *context) {
  switch (op) {
  case Mul:
    return AffineMulExpr::get(lhs, rhs, context);
  case FloorDiv:
    return AffineFloorDivExpr::get(lhs, rhs, context);
  case CeilDiv:
    return AffineCeilDivExpr::get(lhs, rhs, context);
  case Mod:
    return AffineModExpr::get(lhs, rhs, context);
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
}

AffineBinaryOpExpr *Parser::getBinaryAffineOpExpr(AffineLowPrecOp op,
                                                  AffineExpr *lhs,
                                                  AffineExpr *rhs,
                                                  MLIRContext *context) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return AffineAddExpr::get(lhs, rhs, context);
  case AffineLowPrecOp::Sub:
    return AffineSubExpr::get(lhs, rhs, context);
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
}

/// Parses an expression that can be a valid operand of an affine expression
/// (where associativity may not have been specified through parentheses).
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For: i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineLowPrecOpExpression().
ParseResult Parser::parseAffineOperandExpr(const AffineMapParserState &state,
                                           AffineExpr *&result) {
  result = parseParentheticalExpr(state);
  if (!result)
    result = parseBareIdExpr(state);
  if (!result)
    result = parseIntegerExpr(state);
  return result ? ParseSuccess : ParseFailure;
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a AffineHighPrecOp (mul, div, mod).
/// All affine binary ops are left associative.
/// Given llhs, returns (llhs * lhs) * rhs, or (lhs * rhs) if llhs is null. If
/// no rhs can be found, returns (llhs * lhs) or lhs if llhs is null.
//  TODO(bondhugula): check whether mul is w.r.t. a constant - otherwise, the
/// map is semi-affine.
ParseResult Parser::parseAffineHighPrecOpExpr(AffineExpr *llhs,
                                              AffineHighPrecOp llhsOp,
                                              const AffineMapParserState &state,
                                              AffineExpr *&result) {
  // FIXME: Assume for now that llhsOp is mul.
  AffineExpr *lhs = nullptr;
  if (parseAffineOperandExpr(state, lhs)) {
    return ParseFailure;
  }
  AffineHighPrecOp op = HNoOp;
  // Found an LHS. Parse the remaining expression.
  if ((op = consumeIfHighPrecOp())) {
    if (llhs) {
      // TODO(bondhugula): check whether 'lhs' here is a constant (for affine
      // maps); semi-affine maps allow symbols.
      AffineExpr *expr =
          Parser::getBinaryAffineOpExpr(llhsOp, llhs, lhs, context);
      AffineExpr *subRes = nullptr;
      if (parseAffineHighPrecOpExpr(expr, op, state, subRes)) {
        if (!subRes)
          emitError("missing right operand of multiply op");
        // In spite of the error, setting result to prevent duplicate errors
        // messages as the call stack unwinds. All of this due to left
        // associativity.
        result = expr;
        return ParseFailure;
      }
      result = subRes ? subRes : expr;
      return ParseSuccess;
    }
    // No LLHS, get RHS
    AffineExpr *subRes = nullptr;
    if (parseAffineHighPrecOpExpr(lhs, op, state, subRes)) {
      // 'product' needs to be checked to prevent duplicate errors messages as
      // the call stack unwinds. All of this due to left associativity.
      if (!subRes)
        emitError("missing right operand of multiply op");
      return ParseFailure;
    }
    result = subRes;
    return ParseSuccess;
  }

  // This is the last operand in this expression.
  if (llhs) {
    // TODO(bondhugula): check whether lhs here is a constant (for affine
    // maps); semi-affine maps allow symbols.
    result = Parser::getBinaryAffineOpExpr(llhsOp, llhs, lhs, context);
    return ParseSuccess;
  }

  // No llhs, 'lhs' itself is the expression.
  result = lhs;
  return ParseSuccess;
}

/// Consume this token if it is a lower precedence affine op (there are only two
/// precedence levels)
AffineLowPrecOp Parser::consumeIfLowPrecOp() {
  switch (curToken.getKind()) {
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
AffineHighPrecOp Parser::consumeIfHighPrecOp() {
  switch (curToken.getKind()) {
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

/// Parse affine expressions that are bare-id's, integer constants,
/// parenthetical affine expressions, and affine op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. mul, div, and mod are
/// at the same higher precedence level.
///
/// llhs: the affine expression appearing on the left of the one being parsed.
/// This function will return ((llhs + lhs) + rhs) if llhs is non null, and
/// lhs + rhs otherwise; if there is no rhs, llhs + lhs is returned if llhs is
/// non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4.
///
//  TODO(bondhugula): add support for unary op negation. Assuming for now that
//  the op to associate with llhs is add.
ParseResult Parser::parseAffineLowPrecOpExpr(AffineExpr *llhs,
                                             AffineLowPrecOp llhsOp,
                                             const AffineMapParserState &state,
                                             AffineExpr *&result) {
  AffineExpr *lhs = nullptr;
  if (parseAffineOperandExpr(state, lhs))
    return ParseFailure;

  // Found an LHS. Deal with the ops.
  AffineLowPrecOp lOp;
  AffineHighPrecOp rOp;
  if ((lOp = consumeIfLowPrecOp())) {
    if (llhs) {
      AffineExpr *sum =
          Parser::getBinaryAffineOpExpr(llhsOp, llhs, lhs, context);
      AffineExpr *recSum = nullptr;
      parseAffineLowPrecOpExpr(sum, lOp, state, recSum);
      result = recSum ? recSum : sum;
      return ParseSuccess;
    }
    // No LLHS, get RHS and form the expression.
    if (parseAffineLowPrecOpExpr(lhs, lOp, state, result)) {
      if (!result)
        emitError("missing right operand of add op");
      return ParseFailure;
    }
    return ParseSuccess;
  } else if ((rOp = consumeIfHighPrecOp())) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr *highRes = nullptr;
    if (parseAffineHighPrecOpExpr(lhs, rOp, state, highRes)) {
      // 'product' needs to be checked to prevent duplicate errors messages as
      // the call stack unwinds. All of this due to left associativity.
      if (!highRes)
        emitError("missing right operand of binary op");
      return ParseFailure;
    }
    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, assume for now that the op to associate
    // with llhs is add.
    AffineExpr *expr =
        llhs ? getBinaryAffineOpExpr(llhsOp, llhs, highRes, context) : highRes;
    // Recurse for subsequent add's after the affine mul expression
    AffineLowPrecOp nextOp = consumeIfLowPrecOp();
    if (nextOp) {
      AffineExpr *sumProd = nullptr;
      parseAffineLowPrecOpExpr(expr, nextOp, state, sumProd);
      result = sumProd ? sumProd : expr;
    } else {
      result = expr;
    }
    return ParseSuccess;
  } else {
    // Last operand in the expression list.
    if (llhs) {
      result = Parser::getBinaryAffineOpExpr(llhsOp, llhs, lhs, context);
      return ParseSuccess;
    }
    // No llhs, 'lhs' itself is the expression.
    result = lhs;
    return ParseSuccess;
  }
}

/// Parse an affine expression inside parentheses.
/// affine-expr ::= `(` affine-expr `)`
AffineExpr *Parser::parseParentheticalExpr(const AffineMapParserState &state) {
  if (!consumeIf(Token::l_paren)) {
    return nullptr;
  }
  auto *expr = parseAffineExpr(state);
  if (!consumeIf(Token::r_paren)) {
    emitError("expected ')'");
    return nullptr;
  }
  if (!expr)
    emitError("no expression inside parentheses");
  return expr;
}

/// Parse a bare id that may appear in an affine expression.
/// affine-expr ::= bare-id
AffineExpr *Parser::parseBareIdExpr(const AffineMapParserState &state) {
  if (curToken.is(Token::bare_identifier)) {
    StringRef sRef = curToken.getSpelling();
    const auto &dims = state.getDims();
    const auto &symbols = state.getSymbols();
    if (dims.count(sRef)) {
      consumeToken(Token::bare_identifier);
      return AffineDimExpr::get(dims.lookup(sRef), context);
    }
    if (symbols.count(sRef)) {
      consumeToken(Token::bare_identifier);
      return AffineSymbolExpr::get(symbols.lookup(sRef), context);
    }
    return emitError("identifier is neither dimensional nor symbolic"), nullptr;
  }
  return nullptr;
}

/// Parse an integral constant appearing in an affine expression.
/// affine-expr ::= `-`? integer-literal
/// TODO(bondhugula): handle negative numbers.
AffineExpr *Parser::parseIntegerExpr(const AffineMapParserState &state) {
  if (curToken.is(Token::integer)) {
    auto *expr = AffineConstantExpr::get(
        curToken.getUnsignedIntegerValue().getValue(), context);
    consumeToken(Token::integer);
    return expr;
  }
  return nullptr;
}

/// Parse an affine expression.
/// affine-expr ::= `(` affine-expr `)`
///              | affine-expr `+` affine-expr
///              | affine-expr `-` affine-expr
///              | `-`? integer-literal `*` affine-expr
///              | `ceildiv` `(` affine-expr `,` integer-literal `)`
///              | `floordiv` `(` affine-expr `,` integer-literal `)`
///              | affine-expr `mod` integer-literal
///              | bare-id
///              | `-`? integer-literal
/// Use 'state' to check if valid identifiers appear.
//  TODO(bondhugula): check if mul, mod, div take integral constants
AffineExpr *Parser::parseAffineExpr(const AffineMapParserState &state) {
  switch (curToken.getKind()) {
  case Token::l_paren:
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::bare_identifier:
  case Token::integer: {
    AffineExpr *result = nullptr;
    parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp, state, result);
    return result;
  }

  case Token::plus:
  case Token::minus:
  case Token::star:
    emitError("left operand of binary op missing");
    return nullptr;

  default:
    return nullptr;
  }
}

/// Parse a dim or symbol from the lists appearing before the actual expressions
/// of the affine map. Update state to store the dimensional/symbolic
/// identifier. 'dim': whether it's the dim list or symbol list that is being
/// parsed.
ParseResult Parser::parseDimOrSymbolId(AffineMapParserState &state, bool dim) {
  if (curToken.isNot(Token::bare_identifier))
    return emitError("expected bare identifier");
  auto sRef = curToken.getSpelling();
  consumeToken(Token::bare_identifier);
  if (state.getDims().count(sRef) == 1)
    return emitError("dimensional identifier name reused");
  if (state.getSymbols().count(sRef) == 1)
    return emitError("symbolic identifier name reused");
  if (dim)
    state.addDim(sRef);
  else
    state.addSymbol(sRef);
  return ParseSuccess;
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult Parser::parseSymbolIdList(AffineMapParserState &state) {
  if (!consumeIf(Token::l_bracket)) return emitError("expected '['");

  auto parseElt = [&]() -> ParseResult {
    return parseDimOrSymbolId(state, false);
  };
  return parseCommaSeparatedList(Token::r_bracket, parseElt);
}

/// Parse the list of dimensional identifiers to an affine map.
ParseResult Parser::parseDimIdList(AffineMapParserState &state) {
  if (!consumeIf(Token::l_paren))
    return emitError("expected '(' at start of dimensional identifiers list");

  auto parseElt = [&]() -> ParseResult {
    return parseDimOrSymbolId(state, true);
  };
  return parseCommaSeparatedList(Token::r_paren, parseElt);
}

/// Parse an affine map definition.
///
/// affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        ( `size` `(` dim-size (`,` dim-size)* `)` )?
/// dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
/// multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
AffineMap *Parser::parseAffineMapInline(StringRef mapId) {
  AffineMapParserState state;

  // List of dimensional identifiers.
  if (parseDimIdList(state))
    return nullptr;

  // Symbols are optional.
  if (curToken.is(Token::l_bracket)) {
    if (parseSymbolIdList(state))
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
    auto *elt = parseAffineExpr(state);
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
  return AffineMap::get(state.getNumDims(), state.getNumSymbols(), exprs,
                        context);
}

//===----------------------------------------------------------------------===//
// SSA
//===----------------------------------------------------------------------===//

/// Parse a SSA operand for an instruction or statement.
///
///   ssa-use ::= ssa-id | ssa-constant
///
ParseResult Parser::parseSSAUse() {
  if (curToken.is(Token::percent_identifier)) {
    StringRef name = curToken.getSpelling().drop_front();
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

  // If an argument list is present, parse it.
  if (consumeIf(Token::l_paren)) {
    if (parseOptionalSSAUseAndTypeList(Token::r_paren))
      return ParseFailure;

    // TODO: attach it.
  }

  if (!consumeIf(Token::colon))
    return emitError("expected ':' after basic block name");

  // Parse the list of operations that make up the body of the block.
  while (curToken.isNot(Token::kw_return, Token::kw_br)) {
    auto loc = curToken.getLoc();
    auto *inst = parseCFGOperation(functionState);
    if (!inst)
      return ParseFailure;

    // We just parsed an operation.  If it is a recognized one, verify that it
    // is structurally as we expect.  If not, produce an error with a reasonable
    // source location.
    if (auto *opInfo = inst->getAbstractOperation(context))
      if (auto error = opInfo->verifyInvariants(inst))
        return emitError(loc, error);

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

  StringRef resultID;
  if (curToken.is(Token::percent_identifier)) {
    resultID = curToken.getSpelling().drop_front();
    consumeToken();
    if (!consumeIf(Token::equal))
      return (emitError("expected '=' after SSA name"), nullptr);
  }

  if (curToken.isNot(Token::string))
    return (emitError("expected operation name in quotes"), nullptr);

  auto name = curToken.getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);

  consumeToken(Token::string);

  if (!consumeIf(Token::l_paren))
    return (emitError("expected '(' to start operand list"), nullptr);

  // Parse the operand list.
  parseOptionalSSAUseList(Token::r_paren);

  SmallVector<NamedAttribute, 4> attributes;
  if (curToken.is(Token::l_brace)) {
    if (parseAttributeDict(attributes))
      return nullptr;
  }

  // TODO: Don't drop result name and operand names on the floor.
  auto nameId = Identifier::get(name, context);
  return new OperationInst(nameId, attributes, context);
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
    // TODO: cond_br.
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
    auto *stmt = parseStatement(function);
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

/// Statement.
///
/// ml-stmt ::= instruction | ml-for-stmt | ml-if-stmt
/// TODO: fix terminology in MLSpec document. ML functions
/// contain operation statements, not instructions.
///
Statement * Parser::parseStatement(ParentType parent) {
  switch (curToken.getKind()) {
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
/// ml-for-stmt ::= `for` ssa-id `=` lower-bound `to` upper-bound
///                (`step` integer-literal)? `{` ml-stmt* `}`
///
ForStmt * Parser::parseForStmt(ParentType parent) {
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
/// ml-if-head ::= `if` ml-if-cond `{` ml-stmt* `}`
///             | ml-if-head `else` `if` ml-if-cond `{` ml-stmt* `}`
/// ml-if-stmt ::= ml-if-head
///             | ml-if-head `else` `{` ml-stmt* `}`
///
IfStmt * Parser::parseIfStmt(PointerUnion<MLFunction *, NodeStmt *> parent) {
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
ParseResult Parser::parseNestedStatements(NodeStmt *parent) {
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

    case Token::hash_identifier:
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
  auto *result =
      Parser(sourceMgr, context,
             errorReporter ? std::move(errorReporter) : defaultErrorReporter)
          .parseModule();

  // Make sure the parse module has no other structural problems detected by the
  // verifier.
  if (result)
    result->verify();
  return result;
}
