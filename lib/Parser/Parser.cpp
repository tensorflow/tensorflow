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
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
using namespace mlir;
using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;

namespace {
class Parser;

/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position etc.  The Parser base class provides
/// methods to access this.
class ParserState {
public:
  ParserState(const llvm::SourceMgr &sourceMgr, Module *module)
      : context(module->getContext()), module(module), lex(sourceMgr, context),
        curToken(lex.lexToken()) {}

  ~ParserState() {
    // Destroy the forward references upon error.
    for (auto forwardRef : functionForwardRefs)
      delete forwardRef.second;
    functionForwardRefs.clear();
  }

  // A map from attribute alias identifier to Attribute.
  llvm::StringMap<Attribute> attributeAliasDefinitions;

  // A map from type alias identifier to Type.
  llvm::StringMap<Type> typeAliasDefinitions;

  // This keeps track of all forward references to functions along with the
  // temporary function used to represent them.
  llvm::DenseMap<Identifier, Function *> functionForwardRefs;

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
  const llvm::SourceMgr &getSourceMgr() { return state.lex.getSourceMgr(); }

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location getEncodedSourceLocation(llvm::SMLoc loc) {
    return state.lex.getEncodedSourceLocation(loc);
  }

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(state.curToken.getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

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

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(Token::Kind expectedToken, const Twine &message);

  /// Parse a comma-separated list of elements up until the specified end token.
  ParseResult
  parseCommaSeparatedListUntil(Token::Kind rightToken,
                               const std::function<ParseResult()> &parseElement,
                               bool allowEmptyList = true);

  /// Parse a comma separated list of elements that must have at least one entry
  /// in it.
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement);

  // We have two forms of parsing methods - those that return a non-null
  // pointer on success, and those that return a ParseResult to indicate whether
  // they returned a failure.  The second class fills in by-reference arguments
  // as the results of their action.

  // Type parsing.
  VectorType parseVectorType();
  ParseResult parseXInDimensionList();
  ParseResult parseDimensionListRanked(SmallVectorImpl<int64_t> &dimensions,
                                       bool allowDynamic);
  Type parseExtendedType();
  ParseResult parsePrettyDialectTypeName(StringRef &prettyName);
  Type parseTensorType();
  Type parseComplexType();
  Type parseTupleType();
  Type parseMemRefType();
  Type parseFunctionType();
  Type parseNonFunctionType();
  Type parseType();
  ParseResult parseTypeListNoParens(SmallVectorImpl<Type> &elements);
  ParseResult parseTypeListParens(SmallVectorImpl<Type> &elements);
  ParseResult parseFunctionResultTypes(SmallVectorImpl<Type> &elements);

  // Attribute parsing.
  Function *resolveFunctionReference(StringRef nameStr, SMLoc nameLoc,
                                     FunctionType type);
  Attribute parseAttribute(Type type = {});

  ParseResult parseAttributeDict(SmallVectorImpl<NamedAttribute> &attributes);

  // Polyhedral structures.
  ParseResult parseAffineMapOrIntegerSetReference(AffineMap &map,
                                                  IntegerSet &set);
  DenseElementsAttr parseDenseElementsAttr(VectorOrTensorType type);
  DenseElementsAttr parseDenseElementsAttrAsTensor(Type eltType);
  VectorOrTensorType parseVectorOrTensorType();

  // Location Parsing.

  /// Trailing locations.
  ///
  ///   trailing-location     ::= location?
  ///
  template <typename Owner>
  ParseResult parseOptionalTrailingLocation(Owner *owner) {
    // If there is a 'loc' we parse a trailing location.
    if (!getToken().is(Token::kw_loc))
      return success();

    // Parse the location.
    llvm::Optional<Location> directLoc;
    if (parseLocation(&directLoc))
      return failure();
    owner->setLoc(*directLoc);
    return success();
  }

  /// Parse an inline location.
  ParseResult parseLocation(llvm::Optional<Location> *loc);

  /// Parse a raw location instance.
  ParseResult parseLocationInstance(llvm::Optional<Location> *loc);

private:
  // The Parser is subclassed and reinstantiated.  Do not add additional
  // non-trivial state here, add it to the ParserState class.
  ParserState &state;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

InFlightDiagnostic Parser::emitError(SMLoc loc, const Twine &message) {
  auto diag = getContext()->emitError(getEncodedSourceLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(Token::error))
    diag.abandon();
  return diag;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult Parser::parseToken(Token::Kind expectedToken,
                               const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a comma separated list of elements that must have at least one entry
/// in it.
ParseResult Parser::parseCommaSeparatedList(
    const std::function<ParseResult()> &parseElement) {
  // Non-empty case starts with an element.
  if (parseElement())
    return failure();

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElement())
      return failure();
  }
  return success();
}

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
///
///   abstract-list ::= rightToken                  // if allowEmptyList == true
///   abstract-list ::= element (',' element)* rightToken
///
ParseResult Parser::parseCommaSeparatedListUntil(
    Token::Kind rightToken, const std::function<ParseResult()> &parseElement,
    bool allowEmptyList) {
  // Handle the empty case.
  if (getToken().is(rightToken)) {
    if (!allowEmptyList)
      return emitError("expected list element");
    consumeToken(rightToken);
    return success();
  }

  if (parseCommaSeparatedList(parseElement) ||
      parseToken(rightToken, "expected ',' or '" +
                                 Token::getTokenSpelling(rightToken) + "'"))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

/// Parse any type except the function type.
///
///   non-function-type ::= integer-type
///                       | index-type
///                       | float-type
///                       | extended-type
///                       | vector-type
///                       | tensor-type
///                       | memref-type
///                       | complex-type
///                       | tuple-type
///                       | none-type
///
///   index-type ::= `index`
///   float-type ::= `f16` | `bf16` | `f32` | `f64`
///   none-type ::= `none`
///
Type Parser::parseNonFunctionType() {
  switch (getToken().getKind()) {
  default:
    return (emitError("expected non-function type"), nullptr);
  case Token::kw_memref:
    return parseMemRefType();
  case Token::kw_tensor:
    return parseTensorType();
  case Token::kw_complex:
    return parseComplexType();
  case Token::kw_tuple:
    return parseTupleType();
  case Token::kw_vector:
    return parseVectorType();
  // integer-type
  case Token::inttype: {
    auto width = getToken().getIntTypeBitwidth();
    if (!width.hasValue())
      return (emitError("invalid integer width"), nullptr);
    auto loc = getEncodedSourceLocation(getToken().getLoc());
    consumeToken(Token::inttype);
    return IntegerType::getChecked(width.getValue(), builder.getContext(), loc);
  }

  // float-type
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

  // index-type
  case Token::kw_index:
    consumeToken(Token::kw_index);
    return builder.getIndexType();

  // none-type
  case Token::kw_none:
    consumeToken(Token::kw_none);
    return builder.getNoneType();

  // extended type
  case Token::exclamation_identifier:
    return parseExtendedType();
  }
}

/// Parse an arbitrary type.
///
///   type ::= function-type
///          | non-function-type
///
Type Parser::parseType() {
  if (getToken().is(Token::l_paren))
    return parseFunctionType();
  return parseNonFunctionType();
}

/// Parse a vector type.
///
///   vector-type ::= `vector` `<` static-dimension-list primitive-type `>`
///   static-dimension-list ::= (decimal-literal `x`)+
///
VectorType Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (parseToken(Token::less, "expected '<' in vector type"))
    return nullptr;

  SmallVector<int64_t, 4> dimensions;
  if (parseDimensionListRanked(dimensions, /*allowDynamic=*/false))
    return nullptr;
  if (dimensions.empty())
    return (emitError("expected dimension size in vector type"), nullptr);

  // Parse the element type.
  auto typeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType || parseToken(Token::greater, "expected '>' in vector type"))
    return nullptr;

  return VectorType::getChecked(dimensions, elementType,
                                getEncodedSourceLocation(typeLoc));
}

/// Parse an 'x' token in a dimension list, handling the case where the x is
/// juxtaposed with an element type, as in "xf32", leaving the "f32" as the next
/// token.
ParseResult Parser::parseXInDimensionList() {
  if (getToken().isNot(Token::bare_identifier) || getTokenSpelling()[0] != 'x')
    return emitError("expected 'x' in dimension list");

  // If we had a prefix of 'x', lex the next token immediately after the 'x'.
  if (getTokenSpelling().size() != 1)
    state.lex.resetPointer(getTokenSpelling().data() + 1);

  // Consume the 'x'.
  consumeToken(Token::bare_identifier);

  return success();
}

/// Parse a dimension list of a tensor or memref type.  This populates the
/// dimension list, using -1 for the `?` dimensions if `allowDynamic` is set and
/// errors out on `?` otherwise.
///
///   dimension-list-ranked ::= (dimension `x`)*
///   dimension ::= `?` | decimal-literal
///
/// When `allowDynamic` is not set, this can be also used to parse
///
///   static-dimension-list ::= (decimal-literal `x`)*
ParseResult
Parser::parseDimensionListRanked(SmallVectorImpl<int64_t> &dimensions,
                                 bool allowDynamic = true) {
  while (getToken().isAny(Token::integer, Token::question)) {
    if (consumeIf(Token::question)) {
      if (!allowDynamic)
        return emitError("expected static shape");
      dimensions.push_back(-1);
    } else {
      // Hexadecimal integer literals (starting with `0x`) are not allowed in
      // aggregate type declarations.  Therefore, `0xf32` should be processed as
      // a sequence of separate elements `0`, `x`, `f32`.
      if (getTokenSpelling().size() > 1 && getTokenSpelling()[1] == 'x') {
        // We can get here only if the token is an integer literal.  Hexadecimal
        // integer literals can only start with `0x` (`1x` wouldn't lex as a
        // literal, just `1` would, at which point we don't get into this
        // branch).
        assert(getTokenSpelling()[0] == '0' && "invalid integer literal");
        dimensions.push_back(0);
        state.lex.resetPointer(getTokenSpelling().data() + 1);
        consumeToken();
      } else {
        // Make sure this integer value is in bound and valid.
        auto dimension = getToken().getUnsignedIntegerValue();
        if (!dimension.hasValue())
          return emitError("invalid dimension");
        dimensions.push_back((int64_t)dimension.getValue());
        consumeToken(Token::integer);
      }
    }

    // Make sure we have an 'x' or something like 'xbf32'.
    if (parseXInDimensionList())
      return failure();
  }

  return success();
}

/// Parse the body of a pretty dialect type, which starts and ends with <>'s,
/// and may be recursive.  Return with the 'prettyName' StringRef encompasing
/// the entire pretty name.
///
///   pretty-dialect-type-body ::= '<' pretty-dialect-type-contents+ '>'
///   pretty-dialect-type-contents ::= pretty-dialect-type-body
///                                  | '(' pretty-dialect-type-contents+ ')'
///                                  | '[' pretty-dialect-type-contents+ ']'
///                                  | '{' pretty-dialect-type-contents+ '}'
///                                  | '[^[<({>\])}\0]+'
///
ParseResult Parser::parsePrettyDialectTypeName(StringRef &prettyName) {
  // Pretty type names are a relatively unstructured format that contains a
  // series of properly nested punctuation, with anything else in the middle.
  // Scan ahead to find it and consume it if successful, otherwise emit an
  // error.
  auto *curPtr = getTokenSpelling().data();

  SmallVector<char, 8> nestedPunctuation;

  // Scan over the nested punctuation, bailing out on error and consuming until
  // we find the end.  We know that we're currently looking at the '<', so we
  // can go until we find the matching '>' character.
  assert(*curPtr == '<');
  do {
    char c = *curPtr++;
    switch (c) {
    case '\0':
      // This also handles the EOF case.
      return emitError("unexpected nul or EOF in pretty dialect name");
    case '<':
    case '[':
    case '(':
    case '{':
      nestedPunctuation.push_back(c);
      continue;

    case '>':
      if (nestedPunctuation.pop_back_val() != '<')
        return emitError("unbalanced '>' character in pretty dialect name");
      break;
    case ']':
      if (nestedPunctuation.pop_back_val() != '[')
        return emitError("unbalanced ']' character in pretty dialect name");
      break;
    case ')':
      if (nestedPunctuation.pop_back_val() != '(')
        return emitError("unbalanced ')' character in pretty dialect name");
      break;
    case '}':
      if (nestedPunctuation.pop_back_val() != '{')
        return emitError("unbalanced '}' character in pretty dialect name");
      break;

    default:
      continue;
    }
  } while (!nestedPunctuation.empty());

  // Ok, we succeeded, remember where we stopped, reset the lexer to know it is
  // consuming all this stuff, and return.
  state.lex.resetPointer(curPtr);

  unsigned length = curPtr - prettyName.begin();
  prettyName = StringRef(prettyName.begin(), length);
  consumeToken();
  return success();
}

/// Parse an extended type.
///
///   extended-type ::= (dialect-type | type-alias)
///   dialect-type  ::= `!` dialect-namespace `<` '"' type-data '"' `>`
///   dialect-type  ::= `!` alias-name pretty-dialect-type-body?
///   type-alias    ::= `!` alias-name
///
Type Parser::parseExtendedType() {
  assert(getToken().is(Token::exclamation_identifier));

  // Parse the dialect namespace.
  StringRef identifier = getTokenSpelling().drop_front();
  auto loc = getToken().getLoc();
  consumeToken(Token::exclamation_identifier);

  // If there is no '<' token following this, and if the typename contains no
  // dot, then we are parsing a type alias.
  if (getToken().isNot(Token::less) && !identifier.contains('.')) {
    // Check for an alias for this type.
    auto aliasIt = state.typeAliasDefinitions.find(identifier);
    if (aliasIt == state.typeAliasDefinitions.end())
      return (emitError("undefined type alias id '" + identifier + "'"),
              nullptr);
    return aliasIt->second;
  }

  // Otherwise, we are parsing a dialect-specific type.  If the name contains a
  // dot, then this is the "pretty" form.  If not, it is the verbose form that
  // looks like <"...">.
  std::string typeData;
  auto dialectName = identifier;

  // Handle the verbose form, where "identifier" is a simple dialect name.
  if (!identifier.contains('.')) {
    // Consume the '<'.
    if (parseToken(Token::less, "expected '<' in dialect type"))
      return nullptr;

    // Parse the type specific data.
    if (getToken().isNot(Token::string))
      return (emitError("expected string literal type data in dialect type"),
              nullptr);
    typeData = getToken().getStringValue();
    loc = getToken().getLoc();
    consumeToken(Token::string);

    // Consume the '>'.
    if (parseToken(Token::greater, "expected '>' in dialect type"))
      return nullptr;
  } else {
    // Ok, the dialect name is the part of the identifier before the dot, the
    // part after the dot is the dialect's type, or the start thereof.
    auto dotHalves = identifier.split('.');
    dialectName = dotHalves.first;
    auto prettyName = dotHalves.second;

    // If the dialect's type is followed immediately by a <, then lex the body
    // of it into prettyName.
    if (getToken().is(Token::less) &&
        prettyName.bytes_end() == getTokenSpelling().bytes_begin()) {
      if (parsePrettyDialectTypeName(prettyName))
        return nullptr;
    }

    typeData = prettyName.str();
  }

  auto encodedLoc = getEncodedSourceLocation(loc);

  // If we found a registered dialect, then ask it to parse the type.
  if (auto *dialect = state.context->getRegisteredDialect(dialectName))
    return dialect->parseType(typeData, encodedLoc);

  // Otherwise, form a new opaque type.
  return OpaqueType::getChecked(Identifier::get(dialectName, state.context),
                                typeData, state.context, encodedLoc);
}

/// Parse a tensor type.
///
///   tensor-type ::= `tensor` `<` dimension-list element-type `>`
///   dimension-list ::= dimension-list-ranked | `*x`
///
Type Parser::parseTensorType() {
  consumeToken(Token::kw_tensor);

  if (parseToken(Token::less, "expected '<' in tensor type"))
    return nullptr;

  bool isUnranked;
  SmallVector<int64_t, 4> dimensions;

  if (consumeIf(Token::star)) {
    // This is an unranked tensor type.
    isUnranked = true;

    if (parseXInDimensionList())
      return nullptr;

  } else {
    isUnranked = false;
    if (parseDimensionListRanked(dimensions))
      return nullptr;
  }

  // Parse the element type.
  auto typeLocation = getEncodedSourceLocation(getToken().getLoc());
  auto elementType = parseType();
  if (!elementType || parseToken(Token::greater, "expected '>' in tensor type"))
    return nullptr;

  if (isUnranked)
    return UnrankedTensorType::getChecked(elementType, typeLocation);
  return RankedTensorType::getChecked(dimensions, elementType, typeLocation);
}

/// Parse a complex type.
///
///   complex-type ::= `complex` `<` type `>`
///
Type Parser::parseComplexType() {
  consumeToken(Token::kw_complex);

  // Parse the '<'.
  if (parseToken(Token::less, "expected '<' in complex type"))
    return nullptr;

  auto typeLocation = getEncodedSourceLocation(getToken().getLoc());
  auto elementType = parseType();
  if (!elementType ||
      parseToken(Token::greater, "expected '>' in complex type"))
    return nullptr;

  return ComplexType::getChecked(elementType, typeLocation);
}

/// Parse a tuple type.
///
///   tuple-type ::= `tuple` `<` (type (`,` type)*)? `>`
///
Type Parser::parseTupleType() {
  consumeToken(Token::kw_tuple);

  // Parse the '<'.
  if (parseToken(Token::less, "expected '<' in tuple type"))
    return nullptr;

  // Check for an empty tuple by directly parsing '>'.
  if (consumeIf(Token::greater))
    return TupleType::get(getContext());

  // Parse the element types and the '>'.
  SmallVector<Type, 4> types;
  if (parseTypeListNoParens(types) ||
      parseToken(Token::greater, "expected '>' in tuple type"))
    return nullptr;

  return TupleType::get(types, getContext());
}

/// Parse a memref type.
///
///   memref-type ::= `memref` `<` dimension-list-ranked element-type
///                   (`,` semi-affine-map-composition)? (`,` memory-space)? `>`
///
///   semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
///   memory-space ::= integer-literal /* | TODO: address-space-id */
///
Type Parser::parseMemRefType() {
  consumeToken(Token::kw_memref);

  if (parseToken(Token::less, "expected '<' in memref type"))
    return nullptr;

  SmallVector<int64_t, 4> dimensions;
  if (parseDimensionListRanked(dimensions))
    return nullptr;

  // Parse the element type.
  auto typeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType)
    return nullptr;

  // Parse semi-affine-map-composition.
  SmallVector<AffineMap, 2> affineMapComposition;
  unsigned memorySpace = 0;
  bool parsedMemorySpace = false;

  auto parseElt = [&]() -> ParseResult {
    if (getToken().is(Token::integer)) {
      // Parse memory space.
      if (parsedMemorySpace)
        return emitError("multiple memory spaces specified in memref type");
      auto v = getToken().getUnsignedIntegerValue();
      if (!v.hasValue())
        return emitError("invalid memory space in memref type");
      memorySpace = v.getValue();
      consumeToken(Token::integer);
      parsedMemorySpace = true;
    } else {
      // Parse affine map.
      if (parsedMemorySpace)
        return emitError("affine map after memory space in memref type");
      auto affineMap = parseAttribute();
      if (!affineMap)
        return failure();

      // Verify that the parsed attribute is an affine map.
      if (auto affineMapAttr = affineMap.dyn_cast<AffineMapAttr>())
        affineMapComposition.push_back(affineMapAttr.getValue());
      else
        return emitError("expected affine map in memref type");
    }
    return success();
  };

  // Parse a list of mappings and address space if present.
  if (consumeIf(Token::comma)) {
    // Parse comma separated list of affine maps, followed by memory space.
    if (parseCommaSeparatedListUntil(Token::greater, parseElt,
                                     /*allowEmptyList=*/false)) {
      return nullptr;
    }
  } else {
    if (parseToken(Token::greater, "expected ',' or '>' in memref type"))
      return nullptr;
  }

  return MemRefType::getChecked(dimensions, elementType, affineMapComposition,
                                memorySpace, getEncodedSourceLocation(typeLoc));
}

/// Parse a function type.
///
///   function-type ::= type-list-parens `->` type-list
///
Type Parser::parseFunctionType() {
  assert(getToken().is(Token::l_paren));

  SmallVector<Type, 4> arguments, results;
  if (parseTypeListParens(arguments) ||
      parseToken(Token::arrow, "expected '->' in function type") ||
      parseFunctionResultTypes(results))
    return nullptr;

  return builder.getFunctionType(arguments, results);
}

/// Parse a list of types without an enclosing parenthesis.  The list must have
/// at least one member.
///
///   type-list-no-parens ::=  type (`,` type)*
///
ParseResult Parser::parseTypeListNoParens(SmallVectorImpl<Type> &elements) {
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseType();
    elements.push_back(elt);
    return elt ? success() : failure();
  };

  return parseCommaSeparatedList(parseElt);
}

/// Parse a parenthesized list of types.
///
///   type-list-parens ::= `(` `)`
///                      | `(` type-list-no-parens `)`
///
ParseResult Parser::parseTypeListParens(SmallVectorImpl<Type> &elements) {
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();

  // Handle empty lists.
  if (getToken().is(Token::r_paren))
    return consumeToken(), success();

  if (parseTypeListNoParens(elements) ||
      parseToken(Token::r_paren, "expected ')'"))
    return failure();
  return success();
}

/// Parse a function result type.
///
///   function-result-type ::= type-list-parens
///                          | non-function-type
///
ParseResult Parser::parseFunctionResultTypes(SmallVectorImpl<Type> &elements) {
  if (getToken().is(Token::l_paren))
    return parseTypeListParens(elements);

  Type t = parseNonFunctionType();
  if (!t)
    return failure();
  elements.push_back(t);
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute parsing.
//===----------------------------------------------------------------------===//

namespace {
class TensorLiteralParser {
public:
  TensorLiteralParser(Parser &p, Type eltTy) : p(p), eltTy(eltTy) {}

  ParseResult parse() {
    if (p.getToken().is(Token::l_square)) {
      return parseList(shape);
    }
    return parseElement();
  }

  ArrayRef<Attribute> getValues() const { return storage; }

  ArrayRef<int64_t> getShape() const { return shape; }

private:
  /// Parse a single element, returning failure if it isn't a valid element
  /// literal. For example:
  /// parseElement(1) -> Success, 1
  /// parseElement([1]) -> Failure
  ParseResult parseElement();

  /// Parse a list of either lists or elements, returning the dimensions of the
  /// parsed sub-tensors in dims. For example:
  ///   parseList([1, 2, 3]) -> Success, [3]
  ///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
  ///   parseList([[1, 2], 3]) -> Failure
  ///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
  ParseResult parseList(llvm::SmallVectorImpl<int64_t> &dims);

  Parser &p;
  Type eltTy;
  SmallVector<int64_t, 4> shape;
  std::vector<Attribute> storage;
};
} // namespace

ParseResult TensorLiteralParser::parseElement() {
  switch (p.getToken().getKind()) {
  case Token::floatliteral:
  case Token::integer:
  case Token::minus: {
    auto result = p.parseAttribute(eltTy);
    if (!result)
      return failure();
    // check result matches the element type.
    switch (eltTy.getKind()) {
    case StandardTypes::BF16:
    case StandardTypes::F16:
    case StandardTypes::F32:
    case StandardTypes::F64: {
      // Bitcast the APFloat value to APInt and store the bit representation.
      auto fpAttrResult = result.dyn_cast<FloatAttr>();
      if (!fpAttrResult)
        return p.emitError(
            "expected tensor literal element with floating point type");
      auto apInt = fpAttrResult.getValue().bitcastToAPInt();

      // FIXME: using 64 bits and double semantics for BF16 because APFloat does
      // not support BF16 directly.
      size_t bitWidth = eltTy.isBF16() ? 64 : eltTy.getIntOrFloatBitWidth();
      assert(apInt.getBitWidth() == bitWidth);
      (void)bitWidth;
      (void)apInt;
      break;
    }
    case StandardTypes::Integer: {
      if (!result.isa<IntegerAttr>())
        return p.emitError("expected tensor literal element has integer type");
      auto value = result.cast<IntegerAttr>().getValue();
      if (value.getMinSignedBits() > eltTy.getIntOrFloatBitWidth())
        return p.emitError("tensor literal element has more bits than that "
                           "specified in the type");
      break;
    }
    default:
      return p.emitError("expected integer or float tensor element");
    }
    storage.push_back(result);
    break;
  }
  default:
    return p.emitError("expected element literal of primitive type");
  }
  return success();
}

/// Parse a list of either lists or elements, returning the dimensions of the
/// parsed sub-tensors in dims. For example:
///   parseList([1, 2, 3]) -> Success, [3]
///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
///   parseList([[1, 2], 3]) -> Failure
///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
ParseResult
TensorLiteralParser::parseList(llvm::SmallVectorImpl<int64_t> &dims) {
  p.consumeToken(Token::l_square);

  auto checkDims =
      [&](const llvm::SmallVectorImpl<int64_t> &prevDims,
          const llvm::SmallVectorImpl<int64_t> &newDims) -> ParseResult {
    if (prevDims == newDims)
      return success();
    return p.emitError("tensor literal is invalid; ranks are not consistent "
                       "between elements");
  };

  bool first = true;
  llvm::SmallVector<int64_t, 4> newDims;
  unsigned size = 0;
  auto parseCommaSeparatedList = [&]() -> ParseResult {
    llvm::SmallVector<int64_t, 4> thisDims;
    if (p.getToken().getKind() == Token::l_square) {
      if (parseList(thisDims))
        return failure();
    } else if (parseElement()) {
      return failure();
    }
    ++size;
    if (!first)
      return checkDims(newDims, thisDims);
    newDims = thisDims;
    first = false;
    return success();
  };
  if (p.parseCommaSeparatedListUntil(Token::r_square, parseCommaSeparatedList))
    return failure();

  // Return the sublists' dimensions with 'size' prepended.
  dims.clear();
  dims.push_back(size);
  dims.append(newDims.begin(), newDims.end());
  return success();
}

/// Given a parsed reference to a function name like @foo and a type that it
/// corresponds to, resolve it to a concrete function object (possibly
/// synthesizing a forward reference) or emit an error and return null on
/// failure.
Function *Parser::resolveFunctionReference(StringRef nameStr, SMLoc nameLoc,
                                           FunctionType type) {
  Identifier name = builder.getIdentifier(nameStr.drop_front());

  // See if the function has already been defined in the module.
  Function *function = getModule()->getNamedFunction(name);

  // If not, get or create a forward reference to one.
  if (!function) {
    auto &entry = state.functionForwardRefs[name];
    if (!entry)
      entry = new Function(getEncodedSourceLocation(nameLoc), name, type,
                           /*attrs=*/{});
    function = entry;
  }

  if (function->getType() != type)
    return (emitError(nameLoc, "reference to function with mismatched type"),
            nullptr);
  return function;
}

/// Attribute parsing.
///
///  attribute-value ::= `unit`
///                    | bool-literal
///                    | integer-literal (`:` (index-type | integer-type))?
///                    | float-literal (`:` float-type)?
///                    | string-literal
///                    | type
///                    | `[` (attribute-value (`,` attribute-value)*)? `]`
///                    | function-id `:` function-type
///                    | (`splat` | `dense`) `<` (tensor-type | vector-type) `,`
///                      attribute-value `>`
///                    | `sparse` `<` (tensor-type | vector-type)`,`
///                          attribute-value `,` attribute-value `>`
///                    | `opaque` `<` dialect-namespace  `,`
///                      (tensor-type | vector-type) `,` hex-string-literal `>`
///
Attribute Parser::parseAttribute(Type type) {
  // If this is a hash_identifier, we are parsing an attribute alias.
  if (getToken().is(Token::hash_identifier)) {
    StringRef id = getTokenSpelling().drop_front();
    consumeToken(Token::hash_identifier);

    // Check for an alias for this attribute.
    auto aliasIt = state.attributeAliasDefinitions.find(id);
    if (aliasIt == state.attributeAliasDefinitions.end())
      return (emitError("undefined attribute alias id '" + id + "'"), nullptr);

    // Ensure that the attribute alias has the same type as requested.
    if (type && aliasIt->second.getType() != type) {
      emitError("requested attribute type different then alias attribute type");
      return nullptr;
    }

    return aliasIt->second;
  }

  switch (getToken().getKind()) {
  case Token::kw_unit:
    consumeToken(Token::kw_unit);
    return builder.getUnitAttr();

  case Token::kw_true:
    consumeToken(Token::kw_true);
    return builder.getBoolAttr(true);
  case Token::kw_false:
    consumeToken(Token::kw_false);
    return builder.getBoolAttr(false);

  case Token::floatliteral: {
    auto val = getToken().getFloatingPointValue();
    if (!val.hasValue())
      return (emitError("floating point value too large for attribute"),
              nullptr);
    auto valTok = getToken().getLoc();
    consumeToken(Token::floatliteral);
    if (!type) {
      if (consumeIf(Token::colon)) {
        if (!(type = parseType()))
          return nullptr;
      } else {
        // Default to F64 when no type is specified.
        type = builder.getF64Type();
      }
    }
    if (!type.isa<FloatType>())
      return (emitError("floating point value not valid for specified type"),
              nullptr);
    return FloatAttr::getChecked(type, val.getValue(),
                                 getEncodedSourceLocation(valTok));
  }
  case Token::integer: {
    auto val = getToken().getUInt64IntegerValue();
    if (!val.hasValue() || (int64_t)val.getValue() < 0)
      return (emitError("integer constant out of range for attribute"),
              nullptr);
    consumeToken(Token::integer);
    if (!type) {
      if (consumeIf(Token::colon)) {
        if (!(type = parseType()))
          return nullptr;
      } else {
        // Default to i64 if not type is specified.
        type = builder.getIntegerType(64);
      }
    }
    if (!type.isIntOrIndex())
      return (emitError("integer value not valid for specified type"), nullptr);
    int width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
    APInt apInt(width, val.getValue());
    if (apInt != *val)
      return emitError("integer constant out of range for attribute"), nullptr;
    return builder.getIntegerAttr(type, apInt);
  }

  case Token::minus: {
    consumeToken(Token::minus);
    if (getToken().is(Token::integer)) {
      auto val = getToken().getUInt64IntegerValue();
      if (!val.hasValue() || (int64_t)-val.getValue() >= 0)
        return (emitError("integer constant out of range for attribute"),
                nullptr);
      consumeToken(Token::integer);
      if (!type) {
        if (consumeIf(Token::colon)) {
          if (!(type = parseType()))
            return nullptr;
        } else {
          // Default to i64 if not type is specified.
          type = builder.getIntegerType(64);
        }
      }
      if (!type.isIntOrIndex())
        return (emitError("integer value not valid for type"), nullptr);
      int width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
      APInt apInt(width, *val, /*isSigned=*/true);
      if (apInt != *val)
        return (emitError("integer constant out of range for attribute"),
                nullptr);
      return builder.getIntegerAttr(type, -apInt);
    }
    if (getToken().is(Token::floatliteral)) {
      auto val = getToken().getFloatingPointValue();
      if (!val.hasValue())
        return (emitError("floating point value too large for attribute"),
                nullptr);
      auto valTok = getToken().getLoc();
      consumeToken(Token::floatliteral);
      if (!type) {
        if (consumeIf(Token::colon)) {
          if (!(type = parseType()))
            return nullptr;
        } else {
          // Default to F64 when no type is specified.
          type = builder.getF64Type();
        }
      }
      if (!type.isa<FloatType>())
        return (emitError("floating point value not valid for type"), nullptr);
      return FloatAttr::getChecked(type, -val.getValue(),
                                   getEncodedSourceLocation(valTok));
    }

    return (emitError("expected constant integer or floating point value"),
            nullptr);
  }

  case Token::string: {
    auto val = getToken().getStringValue();
    consumeToken(Token::string);
    return builder.getStringAttr(val);
  }

  case Token::l_square: {
    consumeToken(Token::l_square);
    SmallVector<Attribute, 4> elements;

    auto parseElt = [&]() -> ParseResult {
      elements.push_back(parseAttribute());
      return elements.back() ? success() : failure();
    };

    if (parseCommaSeparatedListUntil(Token::r_square, parseElt))
      return nullptr;
    return builder.getArrayAttr(elements);
  }
  case Token::l_paren: {
    // Try to parse an affine map or an integer set reference.
    AffineMap map;
    IntegerSet set;
    if (parseAffineMapOrIntegerSetReference(map, set))
      return nullptr;
    if (map)
      return builder.getAffineMapAttr(map);
    assert(set);
    return builder.getIntegerSetAttr(set);
  }

  case Token::at_identifier: {
    auto nameLoc = getToken().getLoc();
    auto nameStr = getTokenSpelling();
    consumeToken(Token::at_identifier);

    if (parseToken(Token::colon, "expected ':' and function type"))
      return nullptr;
    auto typeLoc = getToken().getLoc();
    Type type = parseType();
    if (!type)
      return nullptr;
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType)
      return (emitError(typeLoc, "expected function type"), nullptr);

    auto *function = resolveFunctionReference(nameStr, nameLoc, fnType);
    return function ? builder.getFunctionAttr(function) : nullptr;
  }
  case Token::kw_opaque: {
    consumeToken(Token::kw_opaque);
    if (parseToken(Token::less, "expected '<' after 'opaque'"))
      return nullptr;

    if (getToken().getKind() != Token::string)
      return (emitError("expected dialect namespace"), nullptr);
    auto name = getToken().getStringValue();
    auto *dialect = builder.getContext()->getRegisteredDialect(name);
    // TODO(shpeisman): Allow for having an unknown dialect on an opaque
    // attribute. Otherwise, it can't be roundtripped without having the dialect
    // registered.
    if (!dialect)
      return (emitError("no registered dialect with namespace '" + name + "'"),
              nullptr);

    consumeToken(Token::string);
    if (parseToken(Token::comma, "expected ','"))
      return nullptr;

    auto type = parseVectorOrTensorType();
    if (!type)
      return nullptr;

    if (getToken().getKind() != Token::string)
      return (emitError("opaque string should start with '0x'"), nullptr);
    auto val = getToken().getStringValue();
    if (val.size() < 2 || val[0] != '0' || val[1] != 'x')
      return (emitError("opaque string should start with '0x'"), nullptr);
    val = val.substr(2);
    if (!std::all_of(val.begin(), val.end(),
                     [](char c) { return llvm::isHexDigit(c); })) {
      return (emitError("opaque string only contains hex digits"), nullptr);
    }
    consumeToken(Token::string);
    if (parseToken(Token::greater, "expected '>'"))
      return nullptr;
    return builder.getOpaqueElementsAttr(dialect, type, llvm::fromHex(val));
  }
  case Token::kw_splat: {
    consumeToken(Token::kw_splat);
    if (parseToken(Token::less, "expected '<' after 'splat'"))
      return nullptr;

    auto type = parseVectorOrTensorType();
    if (!type)
      return nullptr;
    switch (getToken().getKind()) {
    case Token::floatliteral:
    case Token::integer:
    case Token::kw_false:
    case Token::kw_true:
    case Token::minus: {
      auto scalar = parseAttribute(type.getElementType());
      if (!scalar)
        return nullptr;
      if (parseToken(Token::greater, "expected '>'"))
        return nullptr;
      return builder.getSplatElementsAttr(type, scalar);
    }
    default:
      return (emitError("expected scalar constant inside tensor literal"),
              nullptr);
    }
  }
  case Token::kw_dense: {
    consumeToken(Token::kw_dense);
    if (parseToken(Token::less, "expected '<' after 'dense'"))
      return nullptr;

    auto type = parseVectorOrTensorType();
    if (!type)
      return nullptr;

    auto attr = parseDenseElementsAttr(type);
    if (!attr)
      return nullptr;

    if (parseToken(Token::greater, "expected '>'"))
      return nullptr;

    return attr;
  }
  case Token::kw_sparse: {
    consumeToken(Token::kw_sparse);
    if (parseToken(Token::less, "Expected '<' after 'sparse'"))
      return nullptr;

    auto type = parseVectorOrTensorType();
    if (!type)
      return nullptr;

    switch (getToken().getKind()) {
    case Token::l_square: {
      /// Parse indices
      auto indicesEltType = builder.getIntegerType(64);
      auto indices = parseDenseElementsAttrAsTensor(indicesEltType);
      if (!indices)
        return nullptr;

      if (parseToken(Token::comma, "expected ','"))
        return nullptr;

      /// Parse values.
      auto valuesEltType = type.getElementType();
      auto values = parseDenseElementsAttrAsTensor(valuesEltType);
      if (!values)
        return nullptr;

      /// Sanity check.
      auto valuesType = values.getType();
      if (valuesType.getRank() != 1) {
        return (emitError("expected 1-d tensor for values"), nullptr);
      }
      auto indicesType = indices.getType();
      auto sameShape = (indicesType.getRank() == 1) ||
                       (type.getRank() == indicesType.getDimSize(1));
      auto sameElementNum =
          indicesType.getDimSize(0) == valuesType.getDimSize(0);
      if (!sameShape || !sameElementNum) {
        emitError() << "expected shape ([" << type.getShape()
                    << "]); inferred shape of indices literal (["
                    << indicesType.getShape()
                    << "]); inferred shape of values literal (["
                    << valuesType.getShape() << "])";
        return nullptr;
      }

      if (parseToken(Token::greater, "expected '>'"))
        return nullptr;

      // Build the sparse elements attribute by the indices and values.
      return builder.getSparseElementsAttr(
          type, indices.cast<DenseIntElementsAttr>(), values);
    }
    default:
      return (emitError("expected '[' to start sparse tensor literal"),
              nullptr);
    }
    return (emitError("expected elements literal has a tensor or vector type"),
            nullptr);
  }
  default: {
    if (Type type = parseType())
      return builder.getTypeAttr(type);
    return nullptr;
  }
  }
}

/// Dense elements attribute.
///
///   dense-attr-list ::= `[` attribute-value `]`
///   attribute-value ::= integer-literal
///                     | float-literal
///                     | `[` (attribute-value (`,` attribute-value)*)? `]`
///
/// This method returns a constructed dense elements attribute of tensor type
/// with the shape from the parsing result.
DenseElementsAttr Parser::parseDenseElementsAttrAsTensor(Type eltType) {
  TensorLiteralParser literalParser(*this, eltType);
  if (literalParser.parse())
    return nullptr;

  auto type = builder.getTensorType(literalParser.getShape(), eltType);
  return builder.getDenseElementsAttr(type, literalParser.getValues())
      .cast<DenseElementsAttr>();
}

/// Dense elements attribute.
///
///   dense-attr-list ::= `[` attribute-value `]`
///   attribute-value ::= integer-literal
///                     | float-literal
///                     | `[` (attribute-value (`,` attribute-value)*)? `]`
///
/// This method compares the shapes from the parsing result and that from the
/// input argument. It returns a constructed dense elements attribute if both
/// match.
DenseElementsAttr Parser::parseDenseElementsAttr(VectorOrTensorType type) {
  auto eltTy = type.getElementType();
  TensorLiteralParser literalParser(*this, eltTy);
  if (literalParser.parse())
    return nullptr;

  if (literalParser.getShape() != type.getShape()) {
    emitError() << "inferred shape of elements literal (["
                << literalParser.getShape() << "]) does not match type (["
                << type.getShape() << "])";
    return nullptr;
  }

  return builder.getDenseElementsAttr(type, literalParser.getValues())
      .cast<DenseElementsAttr>();
}

/// Vector or tensor type for elements attribute.
///
///   vector-or-tensor-type ::= vector-type | tensor-type
///
/// This method also checks the type has static shape and ranked.
VectorOrTensorType Parser::parseVectorOrTensorType() {
  auto elementType = parseType();
  if (!elementType)
    return nullptr;

  auto type = elementType.dyn_cast<VectorOrTensorType>();
  if (!type) {
    return (emitError("expected elements literal has a tensor or vector type"),
            nullptr);
  }

  if (parseToken(Token::comma, "expected ','"))
    return nullptr;

  if (!type.hasStaticShape() || type.getRank() == -1) {
    return (emitError("tensor literals must be ranked and have static shape"),
            nullptr);
  }
  return type;
}

/// Debug Location.
///
///   location           ::= `loc` inline-location
///   inline-location    ::= '(' location-inst ')'
///
ParseResult Parser::parseLocation(llvm::Optional<Location> *loc) {
  assert(loc && "loc is expected to be non-null");

  // Check for 'loc' identifier.
  if (getToken().isNot(Token::kw_loc))
    return emitError("expected location keyword");
  consumeToken(Token::kw_loc);

  // Parse the inline-location.
  if (parseToken(Token::l_paren, "expected '(' in inline location") ||
      parseLocationInstance(loc) ||
      parseToken(Token::r_paren, "expected ')' in inline location"))
    return failure();
  return success();
}

/// Specific location instances.
///
/// location-inst ::= filelinecol-location |
///                   name-location |
///                   callsite-location |
///                   fused-location |
///                   unknown-location
/// filelinecol-location ::= string-literal ':' integer-literal
///                                         ':' integer-literal
/// name-location ::= string-literal
/// callsite-location ::= 'callsite' '(' location-inst 'at' location-inst ')'
/// fused-location ::= fused ('<' attribute-value '>')?
///                    '[' location-inst (location-inst ',')* ']'
/// unknown-location ::= 'unknown'
///
ParseResult Parser::parseLocationInstance(llvm::Optional<Location> *loc) {
  auto *ctx = getContext();

  // Handle either name or filelinecol locations.
  if (getToken().is(Token::string)) {
    auto str = getToken().getStringValue();
    consumeToken(Token::string);

    // If the next token is ':' this is a filelinecol location.
    if (consumeIf(Token::colon)) {
      // Parse the line number.
      if (getToken().isNot(Token::integer))
        return emitError("expected integer line number in FileLineColLoc");
      auto line = getToken().getUnsignedIntegerValue();
      if (!line.hasValue())
        return emitError("expected integer line number in FileLineColLoc");
      consumeToken(Token::integer);

      // Parse the ':'.
      if (parseToken(Token::colon, "expected ':' in FileLineColLoc"))
        return failure();

      // Parse the column number.
      if (getToken().isNot(Token::integer))
        return emitError("expected integer column number in FileLineColLoc");
      auto column = getToken().getUnsignedIntegerValue();
      if (!column.hasValue())
        return emitError("expected integer column number in FileLineColLoc");
      consumeToken(Token::integer);

      auto file = UniquedFilename::get(str, ctx);
      *loc = FileLineColLoc::get(file, line.getValue(), column.getValue(), ctx);
      return success();
    }

    // Otherwise, this is a NameLoc.
    *loc = NameLoc::get(Identifier::get(str, ctx), ctx);
    return success();
  }

  // Check for a 'unknown' for an unknown location.
  if (getToken().is(Token::bare_identifier) &&
      getToken().getSpelling() == "unknown") {
    consumeToken(Token::bare_identifier);
    *loc = UnknownLoc::get(ctx);
    return success();
  }

  // If the token is 'fused', then this is a fused location.
  if (getToken().is(Token::bare_identifier) &&
      getToken().getSpelling() == "fused") {
    consumeToken(Token::bare_identifier);

    // Try to parse the optional metadata.
    Attribute metadata;
    if (consumeIf(Token::less)) {
      metadata = parseAttribute();
      if (!metadata)
        return emitError("expected valid attribute metadata");
      // Parse the '>' token.
      if (parseToken(Token::greater,
                     "expected '>' after fused location metadata"))
        return failure();
    }

    // Parse the '['.
    if (parseToken(Token::l_square, "expected '[' in fused location"))
      return failure();

    // Parse the internal locations.
    llvm::SmallVector<Location, 4> locations;
    do {
      llvm::Optional<Location> newLoc;
      if (parseLocationInstance(&newLoc))
        return failure();
      locations.push_back(*newLoc);

      // Parse the ','.
    } while (consumeIf(Token::comma));

    // Parse the ']'.
    if (parseToken(Token::r_square, "expected ']' in fused location"))
      return failure();

    // Return the fused location.
    if (metadata)
      *loc = FusedLoc::get(locations, metadata, getContext());
    else
      *loc = FusedLoc::get(locations, ctx);
    return success();
  }

  // Check for the 'callsite' signifying a callsite location.
  if (getToken().is(Token::bare_identifier) &&
      getToken().getSpelling() == "callsite") {
    consumeToken(Token::bare_identifier);

    // Parse the '('.
    if (parseToken(Token::l_paren, "expected '(' in callsite location"))
      return failure();

    // Parse the callee location.
    llvm::Optional<Location> calleeLoc;
    if (parseLocationInstance(&calleeLoc))
      return failure();

    // Parse the 'at'.
    if (getToken().isNot(Token::bare_identifier) ||
        getToken().getSpelling() != "at")
      return emitError("expected 'at' in callsite location");
    consumeToken(Token::bare_identifier);

    // Parse the caller location.
    llvm::Optional<Location> callerLoc;
    if (parseLocationInstance(&callerLoc))
      return failure();

    // Parse the ')'.
    if (parseToken(Token::r_paren, "expected ')' in callsite location"))
      return failure();

    // Return the callsite location.
    *loc = CallSiteLoc::get(*calleeLoc, *callerLoc, ctx);
    return success();
  }

  return emitError("expected location instance");
}

/// Attribute dictionary.
///
///   attribute-dict ::= `{` `}`
///                    | `{` attribute-entry (`,` attribute-entry)* `}`
///   attribute-entry ::= bare-id `:` attribute-value
///
ParseResult
Parser::parseAttributeDict(SmallVectorImpl<NamedAttribute> &attributes) {
  if (!consumeIf(Token::l_brace))
    return failure();

  auto parseElt = [&]() -> ParseResult {
    // We allow keywords as attribute names.
    if (getToken().isNot(Token::bare_identifier, Token::inttype) &&
        !getToken().isKeyword())
      return emitError("expected attribute name");
    Identifier nameId = builder.getIdentifier(getTokenSpelling());
    consumeToken();

    // Try to parse the ':' for the attribute value.
    if (!consumeIf(Token::colon)) {
      // If there is no ':', we treat this as a unit attribute.
      attributes.push_back({nameId, builder.getUnitAttr()});
      return success();
    }

    auto attr = parseAttribute();
    if (!attr)
      return failure();

    attributes.push_back({nameId, attr});
    return success();
  };

  if (parseCommaSeparatedListUntil(Token::r_brace, parseElt))
    return failure();

  return success();
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

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

namespace {
/// This is a specialized parser for affine structures (affine maps, affine
/// expressions, and integer sets), maintaining the state transient to their
/// bodies.
class AffineParser : public Parser {
public:
  explicit AffineParser(ParserState &state) : Parser(state) {}

  AffineMap parseAffineMapRange(unsigned numDims, unsigned numSymbols);
  ParseResult parseAffineMapOrIntegerSetInline(AffineMap &map, IntegerSet &set);
  IntegerSet parseIntegerSetConstraints(unsigned numDims, unsigned numSymbols);

private:
  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList(unsigned &numDims);
  ParseResult parseSymbolIdList(unsigned &numSymbols);
  ParseResult parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols);
  ParseResult parseIdentifierDefinition(AffineExpr idExpr);

  AffineExpr parseAffineExpr();
  AffineExpr parseParentheticalExpr();
  AffineExpr parseNegateExpression(AffineExpr lhs);
  AffineExpr parseIntegerExpr();
  AffineExpr parseBareIdExpr();

  AffineExpr getAffineBinaryOpExpr(AffineHighPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs, SMLoc opLoc);
  AffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs);
  AffineExpr parseAffineOperandExpr(AffineExpr lhs);
  AffineExpr parseAffineLowPrecOpExpr(AffineExpr llhs, AffineLowPrecOp llhsOp);
  AffineExpr parseAffineHighPrecOpExpr(AffineExpr llhs, AffineHighPrecOp llhsOp,
                                       SMLoc llhsOpLoc);
  AffineExpr parseAffineConstraint(bool *isEq);

private:
  SmallVector<std::pair<StringRef, AffineExpr>, 4> dimsAndSymbols;
};
} // end anonymous namespace

/// Create an affine binary high precedence op expression (mul's, div's, mod).
/// opLoc is the location of the op token to be used to report errors
/// for non-conforming expressions.
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineHighPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs,
                                               SMLoc opLoc) {
  // TODO: make the error location info accurate.
  switch (op) {
  case Mul:
    if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: at least one of the multiply "
                       "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of floordiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of ceildiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of mod "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs % rhs;
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
}

/// Create an affine binary low precedence op expression (add, sub).
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineLowPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return lhs + rhs;
  case AffineLowPrecOp::Sub:
    return lhs - rhs;
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
}

/// Consume this token if it is a lower precedence affine op (there are only
/// two precedence levels).
AffineLowPrecOp AffineParser::consumeIfLowPrecOp() {
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
AffineHighPrecOp AffineParser::consumeIfHighPrecOp() {
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
/// null. llhsOpLoc is the location of the llhsOp token that will be used to
/// report an error for non-conforming expressions.
AffineExpr AffineParser::parseAffineHighPrecOpExpr(AffineExpr llhs,
                                                   AffineHighPrecOp llhsOp,
                                                   SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = getToken().getLoc();
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      AffineExpr expr = getAffineBinaryOpExpr(llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs, llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
AffineExpr AffineParser::parseParentheticalExpr() {
  if (parseToken(Token::l_paren, "expected '('"))
    return nullptr;
  if (getToken().is(Token::r_paren))
    return (emitError("no expression inside parentheses"), nullptr);

  auto expr = parseAffineExpr();
  if (!expr)
    return nullptr;
  if (parseToken(Token::r_paren, "expected ')'"))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
AffineExpr AffineParser::parseNegateExpression(AffineExpr lhs) {
  if (parseToken(Token::minus, "expected '-'"))
    return nullptr;

  AffineExpr operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand)
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return (emitError("missing operand of negation"), nullptr);
  return (-1) * operand;
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr AffineParser::parseBareIdExpr() {
  if (getToken().isNot(Token::bare_identifier))
    return (emitError("expected bare identifier"), nullptr);

  StringRef sRef = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == sRef) {
      consumeToken(Token::bare_identifier);
      return entry.second;
    }
  }

  return (emitError("use of undeclared identifier"), nullptr);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr AffineParser::parseIntegerExpr() {
  auto val = getToken().getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0)
    return (emitError("constant too large for index"), nullptr);

  consumeToken(Token::integer);
  return builder.getAffineConstantExpr((int64_t)val.getValue());
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
AffineExpr AffineParser::parseAffineOperandExpr(AffineExpr lhs) {
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
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned
/// if llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4, where
/// (e2*e3) will be parsed using parseAffineHighPrecOpExpr().
AffineExpr AffineParser::parseAffineLowPrecOpExpr(AffineExpr llhs,
                                                  AffineLowPrecOp llhsOp) {
  AffineExpr lhs;
  if (!(lhs = parseAffineOperandExpr(llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      AffineExpr sum = getAffineBinaryOpExpr(llhsOp, llhs, lhs);
      return parseAffineLowPrecOpExpr(sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(lhs, lOp);
  }
  auto opLoc = getToken().getLoc();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes = parseAffineHighPrecOpExpr(lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr =
        llhs ? getAffineBinaryOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs);
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
/// Additional conditions are checked depending on the production. For eg.,
/// one of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
AffineExpr AffineParser::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual
/// expressions of the affine map. Update our state to store the
/// dimensional/symbolic identifier.
ParseResult AffineParser::parseIdentifierDefinition(AffineExpr idExpr) {
  if (getToken().isNot(Token::bare_identifier))
    return emitError("expected bare identifier");

  auto name = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == name)
      return emitError("redefinition of identifier '" + name + "'");
  }
  consumeToken(Token::bare_identifier);

  dimsAndSymbols.push_back({name, idExpr});
  return success();
}

/// Parse the list of dimensional identifiers to an affine map.
ParseResult AffineParser::parseDimIdList(unsigned &numDims) {
  if (parseToken(Token::l_paren,
                 "expected '(' at start of dimensional identifiers list")) {
    return failure();
  }

  auto parseElt = [&]() -> ParseResult {
    auto dimension = getAffineDimExpr(numDims++, getContext());
    return parseIdentifierDefinition(dimension);
  };
  return parseCommaSeparatedListUntil(Token::r_paren, parseElt);
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult AffineParser::parseSymbolIdList(unsigned &numSymbols) {
  consumeToken(Token::l_square);
  auto parseElt = [&]() -> ParseResult {
    auto symbol = getAffineSymbolExpr(numSymbols++, getContext());
    return parseIdentifierDefinition(symbol);
  };
  return parseCommaSeparatedListUntil(Token::r_square, parseElt);
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult
AffineParser::parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols) {
  if (parseDimIdList(numDims)) {
    return failure();
  }
  if (!getToken().is(Token::l_square)) {
    numSymbols = 0;
    return success();
  }
  return parseSymbolIdList(numSymbols);
}

/// Parses an ambiguous affine map or integer set definition inline.
ParseResult AffineParser::parseAffineMapOrIntegerSetInline(AffineMap &map,
                                                           IntegerSet &set) {
  unsigned numDims = 0, numSymbols = 0;

  // List of dimensional and optional symbol identifiers.
  if (parseDimAndOptionalSymbolIdList(numDims, numSymbols)) {
    return failure();
  }

  // This is needed for parsing attributes as we wouldn't know whether we would
  // be parsing an integer set attribute or an affine map attribute.
  bool isArrow = getToken().is(Token::arrow);
  bool isColon = getToken().is(Token::colon);
  if (!isArrow && !isColon) {
    return emitError("expected '->' or ':'");
  } else if (isArrow) {
    parseToken(Token::arrow, "expected '->' or '['");
    map = parseAffineMapRange(numDims, numSymbols);
    return map ? success() : failure();
  } else if (parseToken(Token::colon, "expected ':' or '['")) {
    return failure();
  }

  if ((set = parseIntegerSetConstraints(numDims, numSymbols)))
    return success();

  return failure();
}

/// Parse the range and sizes affine map definition inline.
///
///  affine-map ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                 (`size` `(` dim-size (`,` dim-size)* `)`)?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
AffineMap AffineParser::parseAffineMapRange(unsigned numDims,
                                            unsigned numSymbols) {
  parseToken(Token::l_paren, "expected '(' at start of affine map range");

  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseAffineExpr();
    ParseResult res = elt ? success() : failure();
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of
  // 1-d affine expressions); the list cannot be empty. Grammar:
  // multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt, false))
    return AffineMap();

  // Parse optional range sizes.
  //  range-sizes ::= (`size` `(` dim-size (`,` dim-size)* `)`)?
  //  dim-size ::= affine-expr | `min` `(` affine-expr (`,` affine-expr)+ `)`
  // TODO(bondhugula): support for min of several affine expressions.
  // TODO: check if sizes are non-negative whenever they are constant.
  SmallVector<AffineExpr, 4> rangeSizes;
  if (consumeIf(Token::kw_size)) {
    // Location of the l_paren token (if it exists) for error reporting later.
    auto loc = getToken().getLoc();
    if (parseToken(Token::l_paren, "expected '(' at start of affine map range"))
      return AffineMap();

    auto parseRangeSize = [&]() -> ParseResult {
      auto loc = getToken().getLoc();
      auto elt = parseAffineExpr();
      if (!elt)
        return failure();

      if (!elt.isSymbolicOrConstant())
        return emitError(loc,
                         "size expressions cannot refer to dimension values");

      rangeSizes.push_back(elt);
      return success();
    };

    if (parseCommaSeparatedListUntil(Token::r_paren, parseRangeSize, false))
      return AffineMap();
    if (exprs.size() > rangeSizes.size())
      return (emitError(loc, "fewer range sizes than range expressions"),
              AffineMap());
    if (exprs.size() < rangeSizes.size())
      return (emitError(loc, "more range sizes than range expressions"),
              AffineMap());
  }

  // Parsed a valid affine map.
  return builder.getAffineMap(numDims, numSymbols, exprs, rangeSizes);
}

/// Parse an ambiguous reference to either and affine map or an integer set.
ParseResult Parser::parseAffineMapOrIntegerSetReference(AffineMap &map,
                                                        IntegerSet &set) {
  return AffineParser(state).parseAffineMapOrIntegerSetInline(map, set);
}

//===----------------------------------------------------------------------===//
// FunctionParser
//===----------------------------------------------------------------------===//

namespace {
/// This class contains parser state that is common across CFG and ML
/// functions, notably for dealing with operations and SSA values.
class FunctionParser : public Parser {
public:
  /// This builder intentionally shadows the builder in the base class, with a
  /// more specific builder type.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow-field"
  FuncBuilder builder;
#pragma clang diagnostic pop

  FunctionParser(ParserState &state, Function *function)
      : Parser(state), builder(function), function(function) {}

  ~FunctionParser();

  ParseResult parseFunctionBody(bool hadNamedArguments);

  /// Parse a single operation successor and it's operand list.
  ParseResult parseSuccessorAndUseList(Block *&dest,
                                       SmallVectorImpl<Value *> &operands);

  /// Parse a comma-separated list of operation successors in brackets.
  ParseResult
  parseSuccessors(SmallVectorImpl<Block *> &destinations,
                  SmallVectorImpl<SmallVector<Value *, 4>> &operands);

  /// After the function is finished parsing, this function checks to see if
  /// there are any remaining issues.
  ParseResult finalizeFunction(SMLoc loc);

  /// This represents a use of an SSA value in the program.  The first two
  /// entries in the tuple are the name and result number of a reference.  The
  /// third is the location of the reference, which is used in case this ends
  /// up being a use of an undefined value.
  struct SSAUseInfo {
    StringRef name;  // Value name, e.g. %42 or %abc
    unsigned number; // Number, specified with #12
    SMLoc loc;       // Location of first definition or use.
  };

  /// Given a reference to an SSA value and its type, return a reference. This
  /// returns null on failure.
  Value *resolveSSAUse(SSAUseInfo useInfo, Type type);

  /// Register a definition of a value with the symbol table.
  ParseResult addDefinition(SSAUseInfo useInfo, Value *value);

  // SSA parsing productions.
  ParseResult parseSSAUse(SSAUseInfo &result);
  ParseResult parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results);

  template <typename ResultType>
  ResultType parseSSADefOrUseAndType(
      const std::function<ResultType(SSAUseInfo, Type)> &action);

  Value *parseSSAUseAndType() {
    return parseSSADefOrUseAndType<Value *>(
        [&](SSAUseInfo useInfo, Type type) -> Value * {
          return resolveSSAUse(useInfo, type);
        });
  }

  template <typename ValueTy>
  ParseResult
  parseOptionalSSAUseAndTypeList(SmallVectorImpl<ValueTy *> &results);

  // Block references.

  ParseResult
  parseOperationRegion(Region &region,
                       ArrayRef<std::pair<SSAUseInfo, Type>> entryArguments);
  ParseResult parseRegionBody(Region &region);
  ParseResult parseBlock(Block *&block);
  ParseResult parseBlockBody(Block *block);

  ParseResult
  parseOptionalBlockArgList(SmallVectorImpl<BlockArgument *> &results,
                            Block *owner);

  /// Cleans up the memory for allocated blocks when a parser error occurs.
  void cleanupInvalidBlocks(ArrayRef<Block *> invalidBlocks) {
    // Add the referenced blocks to the function so that they can be properly
    // cleaned up when the function is destroyed.
    for (auto *block : invalidBlocks)
      function->push_back(block);
  }

  /// Get the block with the specified name, creating it if it doesn't
  /// already exist.  The location specified is the point of use, which allows
  /// us to diagnose references to blocks that are not defined precisely.
  Block *getBlockNamed(StringRef name, SMLoc loc);

  // Define the block with the specified name. Returns the Block* or
  // nullptr in the case of redefinition.
  Block *defineBlockNamed(StringRef name, SMLoc loc, Block *existing);

  // Operations
  ParseResult parseOperation();
  Operation *parseGenericOperation();
  Operation *parseCustomOperation();

  ParseResult parseOperations(Block *block);

  /// Return the location of the value identified by its name and number if it
  /// has been already defined.  Placeholder values are considered undefined.
  llvm::Optional<SMLoc> getDefinitionLoc(StringRef name, unsigned number) {
    if (!values.count(name) || number >= values[name].size())
      return {};
    Value *value = values[name][number].first;
    if (value && !isForwardReferencePlaceholder(value))
      return values[name][number].second;
    return {};
  }

private:
  Function *function;

  // This keeps track of the block names as well as the location of the first
  // reference, used to diagnose invalid block references and memoize them.
  llvm::StringMap<std::pair<Block *, SMLoc>> blocksByName;
  DenseMap<Block *, SMLoc> forwardRef;

  /// This keeps track of all of the SSA values we are tracking, indexed by
  /// their name.  This has one entry per result number.
  llvm::StringMap<SmallVector<std::pair<Value *, SMLoc>, 1>> values;

  /// These are all of the placeholders we've made along with the location of
  /// their first reference, to allow checking for use of undefined values.
  DenseMap<Value *, SMLoc> forwardReferencePlaceholders;

  Value *createForwardReferencePlaceholder(SMLoc loc, Type type);

  /// Return true if this is a forward reference.
  bool isForwardReferencePlaceholder(Value *value) {
    return forwardReferencePlaceholders.count(value);
  }
};
} // end anonymous namespace

ParseResult FunctionParser::parseFunctionBody(bool hadNamedArguments) {
  auto braceLoc = getToken().getLoc();
  if (parseToken(Token::l_brace, "expected '{' in function"))
    return failure();

  // Make sure we have at least one block.
  if (getToken().is(Token::r_brace))
    return emitError("function must have a body");

  // If we had named arguments, then we don't allow a block name.
  if (hadNamedArguments) {
    if (getToken().is(Token::caret_identifier))
      return emitError("invalid block name in function with named arguments");
  }

  // The first block is already created and should be filled in.
  auto firstBlock = &function->front();

  // Parse the first block.
  if (parseBlock(firstBlock))
    return failure();

  // Parse the remaining list of blocks.
  if (parseRegionBody(function->getBody()))
    return failure();

  // Verify that all referenced blocks were defined.
  if (!forwardRef.empty()) {
    SmallVector<std::pair<const char *, Block *>, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardRef) {
      errors.push_back({entry.second.getPointer(), entry.first});
      cleanupInvalidBlocks(entry.first);
    }
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry.first);
      emitError(loc, "reference to an undefined block");
    }
    return failure();
  }

  return finalizeFunction(braceLoc);
}

/// Block list.
///
///   block-list ::= '{' block-list-body
///
ParseResult FunctionParser::parseOperationRegion(
    Region &region,
    ArrayRef<std::pair<FunctionParser::SSAUseInfo, Type>> entryArguments) {
  // Parse the '{'.
  if (parseToken(Token::l_brace, "expected '{' to begin a region"))
    return failure();

  // Check for an empty region.
  if (entryArguments.empty() && consumeIf(Token::r_brace))
    return success();
  Block *currentBlock = builder.getInsertionBlock();

  // Parse the first block directly to allow for it to be unnamed.
  Block *block = new Block();

  // Add arguments to the entry block.
  for (auto &placeholderArgPair : entryArguments)
    if (addDefinition(placeholderArgPair.first,
                      block->addArgument(placeholderArgPair.second))) {
      delete block;
      return failure();
    }

  if (parseBlock(block)) {
    delete block;
    return failure();
  }

  // Verify that no other arguments were parsed.
  if (!entryArguments.empty() &&
      block->getNumArguments() > entryArguments.size()) {
    delete block;
    return emitError("entry block arguments were already defined");
  }

  // Parse the rest of the region.
  region.push_back(block);
  if (parseRegionBody(region))
    return failure();

  // Reset insertion point to the current block.
  builder.setInsertionPointToEnd(currentBlock);
  return success();
}

/// Region.
///
///   region-body ::= block* '}'
///
ParseResult FunctionParser::parseRegionBody(Region &region) {
  // Parse the list of blocks.
  while (!consumeIf(Token::r_brace)) {
    Block *newBlock = nullptr;
    if (parseBlock(newBlock))
      return failure();
    region.push_back(newBlock);
  }
  return success();
}

/// Block declaration.
///
///   block ::= block-label? operation* terminator-op
///   block-label    ::= block-id block-arg-list? `:`
///   block-id       ::= caret-id
///   block-arg-list ::= `(` ssa-id-and-type-list? `)`
///
ParseResult FunctionParser::parseBlock(Block *&block) {
  // The first block for a function is already created.
  if (block) {
    // The name for a first block is optional.
    if (getToken().isNot(Token::caret_identifier))
      return parseBlockBody(block);
  }

  SMLoc nameLoc = getToken().getLoc();
  auto name = getTokenSpelling();
  if (parseToken(Token::caret_identifier, "expected block name"))
    return failure();

  block = defineBlockNamed(name, nameLoc, block);

  // Fail if redefinition.
  if (!block)
    return emitError(nameLoc, "redefinition of block '" + name.str() + "'");

  // If an argument list is present, parse it.
  if (consumeIf(Token::l_paren)) {
    SmallVector<BlockArgument *, 8> bbArgs;
    if (parseOptionalBlockArgList(bbArgs, block) ||
        parseToken(Token::r_paren, "expected ')' to end argument list"))
      return failure();
  }

  if (parseToken(Token::colon, "expected ':' after block name"))
    return failure();

  return parseBlockBody(block);
}

ParseResult FunctionParser::parseBlockBody(Block *block) {

  // Set the insertion point to the block we want to insert new operations
  // into.
  builder.setInsertionPointToEnd(block);

  // Parse the list of operations that make up the body of the block.
  while (getToken().isNot(Token::caret_identifier, Token::r_brace)) {
    switch (getToken().getKind()) {
    default:
      if (parseOperation())
        return failure();
      break;
    }
  }

  return success();
}

/// Create and remember a new placeholder for a forward reference.
Value *FunctionParser::createForwardReferencePlaceholder(SMLoc loc, Type type) {
  // Forward references are always created as operations, even in ML
  // functions, because we just need something with a def/use chain.
  //
  // We create these placeholders as having an empty name, which we know
  // cannot be created through normal user input, allowing us to distinguish
  // them.
  auto name = OperationName("placeholder", getContext());
  auto *op = Operation::create(
      getEncodedSourceLocation(loc), name, /*operands=*/{}, type,
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, getContext());
  forwardReferencePlaceholders[op->getResult(0)] = loc;
  return op->getResult(0);
}

/// Given an unbound reference to an SSA value and its type, return the value
/// it specifies.  This returns null on failure.
Value *FunctionParser::resolveSSAUse(SSAUseInfo useInfo, Type type) {
  auto &entries = values[useInfo.name];

  // If we have already seen a value of this name, return it.
  if (useInfo.number < entries.size() && entries[useInfo.number].first) {
    auto *result = entries[useInfo.number].first;
    // Check that the type matches the other uses.
    if (result->getType() == type)
      return result;

    emitError(useInfo.loc, "use of value '" + useInfo.name.str() +
                               "' expects different type than prior uses");
    emitError(entries[useInfo.number].second, "prior use here");
    return nullptr;
  }

  // Make sure we have enough slots for this.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If the value has already been defined and this is an overly large result
  // number, diagnose that.
  if (entries[0].first && !isForwardReferencePlaceholder(entries[0].first))
    return (emitError(useInfo.loc, "reference to invalid result number"),
            nullptr);

  // Otherwise, this is a forward reference.  Create a placeholder and remember
  // that we did so.
  auto *result = createForwardReferencePlaceholder(useInfo.loc, type);
  entries[useInfo.number].first = result;
  entries[useInfo.number].second = useInfo.loc;
  return result;
}

/// After the function is finished parsing, this function checks to see if
/// there are any remaining issues.
ParseResult FunctionParser::finalizeFunction(SMLoc loc) {
  // Check for any forward references that are left.  If we find any, error
  // out.
  if (!forwardReferencePlaceholders.empty()) {
    SmallVector<std::pair<const char *, Value *>, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardReferencePlaceholders)
      errors.push_back({entry.second.getPointer(), entry.first});
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry.first);
      emitError(loc, "use of undeclared SSA value name");
    }
    return failure();
  }

  return success();
}

FunctionParser::~FunctionParser() {
  for (auto &fwd : forwardReferencePlaceholders) {
    // Drop all uses of undefined forward declared reference and destroy
    // defining operation.
    fwd.first->dropAllUses();
    fwd.first->getDefiningOp()->destroy();
  }
}

/// Register a definition of a value with the symbol table.
ParseResult FunctionParser::addDefinition(SSAUseInfo useInfo, Value *value) {
  auto &entries = values[useInfo.name];

  // Make sure there is a slot for this value.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If we already have an entry for this, check to see if it was a definition
  // or a forward reference.
  if (auto *existing = entries[useInfo.number].first) {
    if (!isForwardReferencePlaceholder(existing)) {
      emitError(useInfo.loc,
                "redefinition of SSA value '" + useInfo.name + "'");
      return emitError(entries[useInfo.number].second,
                       "previously defined here");
    }

    // If it was a forward reference, update everything that used it to use
    // the actual definition instead, delete the forward ref, and remove it
    // from our set of forward references we track.
    existing->replaceAllUsesWith(value);
    existing->getDefiningOp()->destroy();
    forwardReferencePlaceholders.erase(existing);
  }

  entries[useInfo.number].first = value;
  entries[useInfo.number].second = useInfo.loc;
  return success();
}

/// Parse a SSA operand for an operation.
///
///   ssa-use ::= ssa-id
///
ParseResult FunctionParser::parseSSAUse(SSAUseInfo &result) {
  result.name = getTokenSpelling();
  result.number = 0;
  result.loc = getToken().getLoc();
  if (parseToken(Token::percent_identifier, "expected SSA operand"))
    return failure();

  // If we have an affine map ID, it is a result number.
  if (getToken().is(Token::hash_identifier)) {
    if (auto value = getToken().getHashIdentifierNumber())
      result.number = value.getValue();
    else
      return emitError("invalid SSA value result number");
    consumeToken(Token::hash_identifier);
  }

  return success();
}

/// Parse a (possibly empty) list of SSA operands.
///
///   ssa-use-list ::= ssa-use (`,` ssa-use)*
///   ssa-use-list-opt ::= ssa-use-list?
///
ParseResult
FunctionParser::parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results) {
  if (getToken().isNot(Token::percent_identifier))
    return success();
  return parseCommaSeparatedList([&]() -> ParseResult {
    SSAUseInfo result;
    if (parseSSAUse(result))
      return failure();
    results.push_back(result);
    return success();
  });
}

/// Parse an SSA use with an associated type.
///
///   ssa-use-and-type ::= ssa-use `:` type
template <typename ResultType>
ResultType FunctionParser::parseSSADefOrUseAndType(
    const std::function<ResultType(SSAUseInfo, Type)> &action) {

  SSAUseInfo useInfo;
  if (parseSSAUse(useInfo) ||
      parseToken(Token::colon, "expected ':' and type for SSA operand"))
    return nullptr;

  auto type = parseType();
  if (!type)
    return nullptr;

  return action(useInfo, type);
}

/// Parse a (possibly empty) list of SSA operands, followed by a colon, then
/// followed by a type list.
///
///   ssa-use-and-type-list
///     ::= ssa-use-list ':' type-list-no-parens
///
template <typename ValueTy>
ParseResult FunctionParser::parseOptionalSSAUseAndTypeList(
    SmallVectorImpl<ValueTy *> &results) {
  SmallVector<SSAUseInfo, 4> valueIDs;
  if (parseOptionalSSAUseList(valueIDs))
    return failure();

  // If there were no operands, then there is no colon or type lists.
  if (valueIDs.empty())
    return success();

  SmallVector<Type, 4> types;
  if (parseToken(Token::colon, "expected ':' in operand list") ||
      parseTypeListNoParens(types))
    return failure();

  if (valueIDs.size() != types.size())
    return emitError("expected ")
           << valueIDs.size() << " types to match operand list";

  results.reserve(valueIDs.size());
  for (unsigned i = 0, e = valueIDs.size(); i != e; ++i) {
    if (auto *value = resolveSSAUse(valueIDs[i], types[i]))
      results.push_back(cast<ValueTy>(value));
    else
      return failure();
  }

  return success();
}

/// Get the block with the specified name, creating it if it doesn't already
/// exist.  The location specified is the point of use, which allows
/// us to diagnose references to blocks that are not defined precisely.
Block *FunctionParser::getBlockNamed(StringRef name, SMLoc loc) {
  auto &blockAndLoc = blocksByName[name];
  if (!blockAndLoc.first) {
    blockAndLoc.first = new Block();
    forwardRef[blockAndLoc.first] = loc;
    blockAndLoc.second = loc;
  }

  return blockAndLoc.first;
}

/// Define the block with the specified name. Returns the Block* or nullptr in
/// the case of redefinition.
Block *FunctionParser::defineBlockNamed(StringRef name, SMLoc loc,
                                        Block *existing) {
  auto &blockAndLoc = blocksByName[name];
  if (!blockAndLoc.first) {
    // If the caller provided a block, use it.  Otherwise create a new one.
    if (!existing)
      existing = new Block();
    blockAndLoc.first = existing;
    blockAndLoc.second = loc;
    return blockAndLoc.first;
  }

  // Forward declarations are removed once defined, so if we are defining a
  // existing block and it is not a forward declaration, then it is a
  // redeclaration.
  if (!forwardRef.erase(blockAndLoc.first))
    return nullptr;
  return blockAndLoc.first;
}

/// Parse a single operation successor and it's operand list.
///
///   successor ::= block-id branch-use-list?
///   branch-use-list ::= `(` ssa-use-list ':' type-list-no-parens `)`
///
ParseResult
FunctionParser::parseSuccessorAndUseList(Block *&dest,
                                         SmallVectorImpl<Value *> &operands) {
  // Verify branch is identifier and get the matching block.
  if (!getToken().is(Token::caret_identifier))
    return emitError("expected block name");
  dest = getBlockNamed(getTokenSpelling(), getToken().getLoc());
  consumeToken();

  // Handle optional arguments.
  if (consumeIf(Token::l_paren) &&
      (parseOptionalSSAUseAndTypeList(operands) ||
       parseToken(Token::r_paren, "expected ')' to close argument list"))) {
    return failure();
  }

  return success();
}

/// Parse a comma-separated list of operation successors in brackets.
///
///   successor-list ::= `[` successor (`,` successor )* `]`
///
ParseResult FunctionParser::parseSuccessors(
    SmallVectorImpl<Block *> &destinations,
    SmallVectorImpl<SmallVector<Value *, 4>> &operands) {
  if (parseToken(Token::l_square, "expected '['"))
    return failure();

  auto parseElt = [this, &destinations, &operands]() {
    Block *dest;
    SmallVector<Value *, 4> destOperands;
    auto res = parseSuccessorAndUseList(dest, destOperands);
    destinations.push_back(dest);
    operands.push_back(destOperands);
    return res;
  };
  return parseCommaSeparatedListUntil(Token::r_square, parseElt,
                                      /*allowEmptyList=*/false);
}

/// Parse a (possibly empty) list of SSA operands with types as block arguments.
///
///   ssa-id-and-type-list ::= ssa-id-and-type (`,` ssa-id-and-type)*
///
ParseResult FunctionParser::parseOptionalBlockArgList(
    SmallVectorImpl<BlockArgument *> &results, Block *owner) {
  if (getToken().is(Token::r_brace))
    return success();

  // If the block already has arguments, then we're handling the entry block.
  // Parse and register the names for the arguments, but do not add them.
  bool definingExistingArgs = owner->getNumArguments() != 0;
  unsigned nextArgument = 0;

  return parseCommaSeparatedList([&]() -> ParseResult {
    auto type = parseSSADefOrUseAndType<Type>(
        [&](SSAUseInfo useInfo, Type type) -> Type {
          BlockArgument *arg;
          if (!definingExistingArgs) {
            arg = owner->addArgument(type);
          } else if (nextArgument >= owner->getNumArguments()) {
            emitError("too many arguments specified in argument list");
            return {};
          } else {
            arg = owner->getArgument(nextArgument++);
            if (arg->getType() != type) {
              emitError("argument and block argument type mismatch");
              return {};
            }
          }

          if (addDefinition(useInfo, arg))
            return {};
          return type;
        });
    return type ? success() : failure();
  });
}

/// Parse an operation.
///
///  operation ::=
///    operation-result? string '(' ssa-use-list? ')' attribute-dict?
///    `:` function-type trailing-location?
///  operation-result ::= ssa-id ((`:` integer-literal) | (`,` ssa-id)*) `=`
///
ParseResult FunctionParser::parseOperation() {
  auto loc = getToken().getLoc();
  SmallVector<std::pair<StringRef, SMLoc>, 1> resultIDs;
  size_t numExpectedResults;
  if (getToken().is(Token::percent_identifier)) {
    // Parse the first result id.
    resultIDs.emplace_back(getTokenSpelling(), loc);
    consumeToken(Token::percent_identifier);

    // If the next token is a ':', we parse the expected result count.
    if (consumeIf(Token::colon)) {
      // Check that the next token is an integer.
      if (!getToken().is(Token::integer))
        return emitError("expected integer number of results");

      // Check that number of results is > 0.
      auto val = getToken().getUInt64IntegerValue();
      if (!val.hasValue() || val.getValue() < 1)
        return emitError("expected named operation to have atleast 1 result");
      consumeToken(Token::integer);
      numExpectedResults = *val;
    } else {
      // Otherwise, this is a comma separated list of result ids.
      if (consumeIf(Token::comma)) {
        auto parseNextResult = [&]() -> ParseResult {
          // Parse the next result id.
          if (!getToken().is(Token::percent_identifier))
            return emitError("expected valid ssa identifier");

          resultIDs.emplace_back(getTokenSpelling(), getToken().getLoc());
          consumeToken(Token::percent_identifier);
          return success();
        };

        if (parseCommaSeparatedList(parseNextResult))
          return failure();
      }
      numExpectedResults = resultIDs.size();
    }

    if (parseToken(Token::equal, "expected '=' after SSA name"))
      return failure();
  }

  Operation *op;
  if (getToken().is(Token::bare_identifier) || getToken().isKeyword())
    op = parseCustomOperation();
  else if (getToken().is(Token::string))
    op = parseGenericOperation();
  else
    return emitError("expected operation name in quotes");

  // If parsing of the basic operation failed, then this whole thing fails.
  if (!op)
    return failure();

  // If the operation had a name, register it.
  if (!resultIDs.empty()) {
    if (op->getNumResults() == 0)
      return emitError(loc, "cannot name an operation with no results");
    if (numExpectedResults != op->getNumResults())
      return emitError(loc, "operation defines ")
             << op->getNumResults() << " results but was provided "
             << numExpectedResults << " to bind";

    // If the number of result names matches the number of operation results, we
    // can directly use the provided names.
    if (resultIDs.size() == op->getNumResults()) {
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i)
        if (addDefinition({resultIDs[i].first, 0, resultIDs[i].second},
                          op->getResult(i)))
          return failure();
    } else {
      // Otherwise, we use the same name for all results.
      StringRef name = resultIDs.front().first;
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i)
        if (addDefinition({name, i, loc}, op->getResult(i)))
          return failure();
    }
  }

  // Try to parse the optional trailing location.
  if (parseOptionalTrailingLocation(op))
    return failure();

  return success();
}

namespace {
// RAII-style guard for cleaning up the regions in the operation state before
// deleting them.  Within the parser, regions may get deleted if parsing failed,
// and other errors may be present, in praticular undominated uses.  This makes
// sure such uses are deleted.
struct CleanupOpStateRegions {
  ~CleanupOpStateRegions() {
    SmallVector<Region *, 4> regionsToClean;
    regionsToClean.reserve(state.regions.size());
    for (auto &region : state.regions)
      if (region)
        for (auto &block : *region)
          block.dropAllDefinedValueUses();
  }
  OperationState &state;
};
} // namespace

Operation *FunctionParser::parseGenericOperation() {
  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(getToken().getLoc());

  auto name = getToken().getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);
  if (name.find('\0') != StringRef::npos)
    return (emitError("null character not allowed in operation name"), nullptr);

  consumeToken(Token::string);

  OperationState result(builder.getContext(), srcLocation, name);

  // Generic operations have a resizable operation list.
  result.setOperandListToResizable();

  // Parse the operand list.
  SmallVector<SSAUseInfo, 8> operandInfos;

  if (parseToken(Token::l_paren, "expected '(' to start operand list") ||
      parseOptionalSSAUseList(operandInfos) ||
      parseToken(Token::r_paren, "expected ')' to end operand list")) {
    return nullptr;
  }

  // Parse the successor list but don't add successors to the result yet to
  // avoid messing up with the argument order.
  SmallVector<Block *, 2> successors;
  SmallVector<SmallVector<Value *, 4>, 2> successorOperands;
  if (getToken().is(Token::l_square)) {
    // Check if the operation is a known terminator.
    const AbstractOperation *abstractOp = result.name.getAbstractOperation();
    if (abstractOp && !abstractOp->hasProperty(OperationProperty::Terminator))
      return emitError("successors in non-terminator"), nullptr;
    if (parseSuccessors(successors, successorOperands))
      return nullptr;
  }

  // Parse the region list.
  CleanupOpStateRegions guard{result};
  if (consumeIf(Token::l_paren)) {
    do {
      // Create temporary regions with function as parent.
      result.regions.emplace_back(new Region(function));
      if (parseOperationRegion(*result.regions.back(),
                               /*entryArguments*/ {}))
        return nullptr;
    } while (consumeIf(Token::comma));
    if (parseToken(Token::r_paren, "expected ')' to end region list"))
      return nullptr;
  }

  if (getToken().is(Token::l_brace)) {
    if (parseAttributeDict(result.attributes))
      return nullptr;
  }

  if (parseToken(Token::colon, "expected ':' followed by operation type"))
    return nullptr;

  auto typeLoc = getToken().getLoc();
  auto type = parseType();
  if (!type)
    return nullptr;
  auto fnType = type.dyn_cast<FunctionType>();
  if (!fnType)
    return (emitError(typeLoc, "expected function type"), nullptr);

  result.addTypes(fnType.getResults());

  // Check that we have the right number of types for the operands.
  auto operandTypes = fnType.getInputs();
  if (operandTypes.size() != operandInfos.size()) {
    auto plural = "s"[operandInfos.size() == 1];
    return (emitError(typeLoc, "expected ")
                << operandInfos.size() << " operand type" << plural
                << " but had " << operandTypes.size(),
            nullptr);
  }

  // Resolve all of the operands.
  for (unsigned i = 0, e = operandInfos.size(); i != e; ++i) {
    result.operands.push_back(resolveSSAUse(operandInfos[i], operandTypes[i]));
    if (!result.operands.back())
      return nullptr;
  }

  // Add the sucessors, and their operands after the proper operands.
  for (const auto &succ : llvm::zip(successors, successorOperands)) {
    Block *successor = std::get<0>(succ);
    const SmallVector<Value *, 4> &operands = std::get<1>(succ);
    result.addSuccessor(successor, operands);
  }

  return builder.createOperation(result);
}

namespace {
class CustomOpAsmParser : public OpAsmParser {
public:
  CustomOpAsmParser(SMLoc nameLoc, StringRef opName, FunctionParser &parser)
      : nameLoc(nameLoc), opName(opName), parser(parser) {}

  ParseResult parseOperation(const AbstractOperation *opDefinition,
                             OperationState *opState) {
    if (opDefinition->parseAssembly(this, opState))
      return failure();

    // Check that none of the operands of the current operation reference an
    // entry block argument for any of the region.
    for (auto *entryArg : parsedRegionEntryArgumentPlaceholders)
      if (llvm::is_contained(opState->operands, entryArg))
        return emitError(nameLoc, "operand use before it's defined");

    return success();
  }

  //===--------------------------------------------------------------------===//
  // High level parsing methods.
  //===--------------------------------------------------------------------===//

  ParseResult getCurrentLocation(llvm::SMLoc *loc) override {
    *loc = parser.getToken().getLoc();
    return success();
  }
  ParseResult parseComma() override {
    return parser.parseToken(Token::comma, "expected ','");
  }
  ParseResult parseColon() override {
    return parser.parseToken(Token::colon, "expected ':'");
  }
  ParseResult parseEqual() override {
    return parser.parseToken(Token::equal, "expected '='");
  }

  ParseResult parseType(Type &result) override {
    return failure(!(result = parser.parseType()));
  }

  ParseResult parseColonType(Type &result) override {
    return failure(parser.parseToken(Token::colon, "expected ':'") ||
                   !(result = parser.parseType()));
  }

  ParseResult parseColonTypeList(SmallVectorImpl<Type> &result) override {
    if (parser.parseToken(Token::colon, "expected ':'"))
      return failure();

    do {
      if (auto type = parser.parseType())
        result.push_back(type);
      else
        return failure();

    } while (parser.consumeIf(Token::comma));
    return success();
  }

  ParseResult parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                                       int requiredOperandCount,
                                       Delimiter delimiter) override {
    if (parser.getToken().is(Token::comma)) {
      parseComma();
      return parseOperandList(result, requiredOperandCount, delimiter);
    }
    if (requiredOperandCount != -1)
      return emitError(parser.getToken().getLoc(), "expected ")
             << requiredOperandCount << " operands";
    return success();
  }

  ParseResult parseOptionalComma() override {
    return success(parser.consumeIf(Token::comma));
  }

  /// Parse an optional keyword.
  ParseResult parseOptionalKeyword(const char *keyword) override {
    // Check that the current token is a bare identifier or keyword.
    if (parser.getToken().isNot(Token::bare_identifier) &&
        !parser.getToken().isKeyword())
      return failure();

    if (parser.getTokenSpelling() == keyword) {
      parser.consumeToken();
      return success();
    }
    return failure();
  }

  /// Parse an arbitrary attribute of a given type and return it in result. This
  /// also adds the attribute to the specified attribute list with the specified
  /// name.
  ParseResult parseAttribute(Attribute &result, Type type, StringRef attrName,
                             SmallVectorImpl<NamedAttribute> &attrs) override {
    result = parser.parseAttribute(type);
    if (!result)
      return failure();

    attrs.push_back(parser.builder.getNamedAttr(attrName, result));
    return success();
  }

  /// Parse an arbitrary attribute and return it in result.  This also adds
  /// the attribute to the specified attribute list with the specified name.
  ParseResult parseAttribute(Attribute &result, StringRef attrName,
                             SmallVectorImpl<NamedAttribute> &attrs) override {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// If a named attribute list is present, parse is into result.
  ParseResult
  parseOptionalAttributeDict(SmallVectorImpl<NamedAttribute> &result) override {
    if (parser.getToken().isNot(Token::l_brace))
      return success();
    return parser.parseAttributeDict(result);
  }

  /// Parse a function name like '@foo' and return the name in a form that can
  /// be passed to resolveFunctionName when a function type is available.
  ParseResult parseFunctionName(StringRef &result, llvm::SMLoc &loc) override {
    if (parseOptionalFunctionName(result, loc))
      return emitError(loc, "expected function name");
    return success();
  }

  /// Parse a function name like '@foo` if present and return the name without
  /// the sigil in `result`.  Return true if the next token is not a function
  /// name and keep `result` unchanged.
  ParseResult parseOptionalFunctionName(StringRef &result,
                                        llvm::SMLoc &loc) override {
    loc = parser.getToken().getLoc();

    if (parser.getToken().isNot(Token::at_identifier))
      return failure();

    result = parser.getTokenSpelling();
    parser.consumeToken(Token::at_identifier);
    return success();
  }

  ParseResult parseOperand(OperandType &result) override {
    FunctionParser::SSAUseInfo useInfo;
    if (parser.parseSSAUse(useInfo))
      return failure();

    result = {useInfo.loc, useInfo.name, useInfo.number};
    return success();
  }

  ParseResult
  parseSuccessorAndUseList(Block *&dest,
                           SmallVectorImpl<Value *> &operands) override {
    // Defer successor parsing to the function parsers.
    return parser.parseSuccessorAndUseList(dest, operands);
  }

  ParseResult parseLParen() override {
    return parser.parseToken(Token::l_paren, "expected '('");
  }

  ParseResult parseRParen() override {
    return parser.parseToken(Token::r_paren, "expected ')'");
  }

  ParseResult parseOperandList(SmallVectorImpl<OperandType> &result,
                               int requiredOperandCount = -1,
                               Delimiter delimiter = Delimiter::None) override {
    auto startLoc = parser.getToken().getLoc();

    // Handle delimiters.
    switch (delimiter) {
    case Delimiter::None:
      // Don't check for the absence of a delimiter if the number of operands
      // is unknown (and hence the operand list could be empty).
      if (requiredOperandCount == -1)
        break;
      // Token already matches an identifier and so can't be a delimiter.
      if (parser.getToken().is(Token::percent_identifier))
        break;
      // Test against known delimiters.
      if (parser.getToken().is(Token::l_paren) ||
          parser.getToken().is(Token::l_square))
        return emitError(startLoc, "unexpected delimiter");
      return emitError(startLoc, "invalid operand");
    case Delimiter::OptionalParen:
      if (parser.getToken().isNot(Token::l_paren))
        return success();
      LLVM_FALLTHROUGH;
    case Delimiter::Paren:
      if (parser.parseToken(Token::l_paren, "expected '(' in operand list"))
        return failure();
      break;
    case Delimiter::OptionalSquare:
      if (parser.getToken().isNot(Token::l_square))
        return success();
      LLVM_FALLTHROUGH;
    case Delimiter::Square:
      if (parser.parseToken(Token::l_square, "expected '[' in operand list"))
        return failure();
      break;
    }

    // Check for zero operands.
    if (parser.getToken().is(Token::percent_identifier)) {
      do {
        OperandType operand;
        if (parseOperand(operand))
          return failure();
        result.push_back(operand);
      } while (parser.consumeIf(Token::comma));
    }

    // Handle delimiters.   If we reach here, the optional delimiters were
    // present, so we need to parse their closing one.
    switch (delimiter) {
    case Delimiter::None:
      break;
    case Delimiter::OptionalParen:
    case Delimiter::Paren:
      if (parser.parseToken(Token::r_paren, "expected ')' in operand list"))
        return failure();
      break;
    case Delimiter::OptionalSquare:
    case Delimiter::Square:
      if (parser.parseToken(Token::r_square, "expected ']' in operand list"))
        return failure();
      break;
    }

    if (requiredOperandCount != -1 && result.size() != requiredOperandCount)
      return emitError(startLoc, "expected ")
             << requiredOperandCount << " operands";
    return success();
  }

  /// Resolve a parse function name and a type into a function reference.
  ParseResult resolveFunctionName(StringRef name, FunctionType type,
                                  llvm::SMLoc loc, Function *&result) override {
    result = parser.resolveFunctionReference(name, loc, type);
    return failure(result == nullptr);
  }

  /// Parse a region that takes `arguments` of `argTypes` types.  This
  /// effectively defines the SSA values of `arguments` and assignes their type.
  ParseResult parseRegion(Region &region, ArrayRef<OperandType> arguments,
                          ArrayRef<Type> argTypes) override {
    assert(arguments.size() == argTypes.size() &&
           "mismatching number of arguments and types");

    SmallVector<std::pair<FunctionParser::SSAUseInfo, Type>, 2> regionArguments;
    for (const auto &pair : llvm::zip(arguments, argTypes)) {
      const OperandType &operand = std::get<0>(pair);
      Type type = std::get<1>(pair);
      FunctionParser::SSAUseInfo operandInfo = {operand.name, operand.number,
                                                operand.location};
      regionArguments.emplace_back(operandInfo, type);

      // Create a placeholder for this argument so that we can detect invalid
      // references to region arguments.
      Value *value = parser.resolveSSAUse(operandInfo, type);
      if (!value)
        return failure();
      parsedRegionEntryArgumentPlaceholders.emplace_back(value);
    }

    return parser.parseOperationRegion(region, regionArguments);
  }

  /// Parse a region argument.  Region arguments define new values, so this also
  /// checks if the values with the same name has not been defined yet.  The
  /// type of the argument will be resolved later by a call to `parseRegion`.
  ParseResult parseRegionArgument(OperandType &argument) override {
    // Use parseOperand to fill in the OperandType structure.
    if (parseOperand(argument))
      return failure();
    if (auto defLoc = parser.getDefinitionLoc(argument.name, argument.number)) {
      parser.emitError(argument.location,
                       "redefinition of SSA value '" + argument.name + "'")
              .attachNote(parser.getEncodedSourceLocation(*defLoc))
          << "previously defined here";
      return failure();
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Methods for interacting with the parser
  //===--------------------------------------------------------------------===//

  Builder &getBuilder() const override { return parser.builder; }

  llvm::SMLoc getNameLoc() const override { return nameLoc; }

  ParseResult resolveOperand(const OperandType &operand, Type type,
                             SmallVectorImpl<Value *> &result) override {
    FunctionParser::SSAUseInfo operandInfo = {operand.name, operand.number,
                                              operand.location};
    if (auto *value = parser.resolveSSAUse(operandInfo, type)) {
      result.push_back(value);
      return success();
    }
    return failure();
  }

  /// Emit a diagnostic at the specified location and return failure.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) override {
    emittedError = true;
    return parser.emitError(loc, "custom op '" + opName + "' " + message);
  }

  bool didEmitError() const { return emittedError; }

private:
  SmallVector<Value *, 2> parsedRegionEntryArgumentPlaceholders;
  SMLoc nameLoc;
  StringRef opName;
  FunctionParser &parser;
  bool emittedError = false;
};
} // end anonymous namespace.

Operation *FunctionParser::parseCustomOperation() {
  auto opLoc = getToken().getLoc();
  auto opName = getTokenSpelling();
  CustomOpAsmParser opAsmParser(opLoc, opName, *this);

  auto *opDefinition = AbstractOperation::lookup(opName, getContext());
  if (!opDefinition && !opName.contains('.')) {
    // If the operation name has no namespace prefix we treat it as a standard
    // operation and prefix it with "std".
    // TODO: Would it be better to just build a mapping of the registered
    // operations in the standard dialect?
    opDefinition =
        AbstractOperation::lookup(Twine("std." + opName).str(), getContext());
  }

  if (!opDefinition) {
    opAsmParser.emitError(opLoc, "is unknown");
    return nullptr;
  }

  consumeToken();

  // If the custom op parser crashes, produce some indication to help
  // debugging.
  std::string opNameStr = opName.str();
  llvm::PrettyStackTraceFormat fmt("MLIR Parser: custom op parser '%s'",
                                   opNameStr.c_str());

  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(opLoc);

  // Have the op implementation take a crack and parsing this.
  OperationState opState(builder.getContext(), srcLocation, opDefinition->name);
  CleanupOpStateRegions guard{opState};
  if (opAsmParser.parseOperation(opDefinition, &opState))
    return nullptr;

  // If it emitted an error, we failed.
  if (opAsmParser.didEmitError())
    return nullptr;

  // Otherwise, we succeeded.  Use the state it parsed as our op information.
  return builder.createOperation(opState);
}

/// Parse an affine constraint.
///  affine-constraint ::= affine-expr `>=` `0`
///                      | affine-expr `==` `0`
///
/// isEq is set to true if the parsed constraint is an equality, false if it
/// is an inequality (greater than or equal).
///
AffineExpr AffineParser::parseAffineConstraint(bool *isEq) {
  AffineExpr expr = parseAffineExpr();
  if (!expr)
    return nullptr;

  if (consumeIf(Token::greater) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim.hasValue() && dim.getValue() == 0) {
      consumeToken(Token::integer);
      *isEq = false;
      return expr;
    }
    return (emitError("expected '0' after '>='"), nullptr);
  }

  if (consumeIf(Token::equal) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim.hasValue() && dim.getValue() == 0) {
      consumeToken(Token::integer);
      *isEq = true;
      return expr;
    }
    return (emitError("expected '0' after '=='"), nullptr);
  }

  return (emitError("expected '== 0' or '>= 0' at end of affine constraint"),
          nullptr);
}

/// Parse the constraints that are part of an integer set definition.
///  integer-set-inline
///                ::= dim-and-symbol-id-lists `:`
///                '(' affine-constraint-conjunction? ')'
///  affine-constraint-conjunction ::= affine-constraint (`,`
///                                       affine-constraint)*
///
IntegerSet AffineParser::parseIntegerSetConstraints(unsigned numDims,
                                                    unsigned numSymbols) {
  if (parseToken(Token::l_paren,
                 "expected '(' at start of integer set constraint list"))
    return IntegerSet();

  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;
  auto parseElt = [&]() -> ParseResult {
    bool isEq;
    auto elt = parseAffineConstraint(&isEq);
    ParseResult res = elt ? success() : failure();
    if (elt) {
      constraints.push_back(elt);
      isEqs.push_back(isEq);
    }
    return res;
  };

  // Parse a list of affine constraints (comma-separated).
  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt, true))
    return IntegerSet();

  // If no constraints were parsed, then treat this as a degenerate 'true' case.
  if (constraints.empty()) {
    /* 0 == 0 */
    auto zero = getAffineConstantExpr(0, getContext());
    return builder.getIntegerSet(numDims, numSymbols, zero, true);
  }

  // Parsed a valid integer set.
  return builder.getIntegerSet(numDims, numSymbols, constraints, isEqs);
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
  ParseResult finalizeModule();

  ParseResult parseAttributeAliasDef();

  ParseResult parseTypeAliasDef();

  // Functions.
  ParseResult
  parseArgumentList(SmallVectorImpl<Type> &argTypes,
                    SmallVectorImpl<StringRef> &argNames,
                    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs);
  ParseResult parseFunctionSignature(
      StringRef &name, FunctionType &type, SmallVectorImpl<StringRef> &argNames,
      SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs);
  ParseResult parseFunc();
};
} // end anonymous namespace

/// Parses an attribute alias declaration.
///
///   attribute-alias-def ::= '#' alias-name `=` attribute-value
///
ParseResult ModuleParser::parseAttributeAliasDef() {
  assert(getToken().is(Token::hash_identifier));

  StringRef attrId = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (getState().attributeAliasDefinitions.count(attrId) > 0)
    return emitError("redefinition of attribute alias id '" + attrId + "'");

  consumeToken(Token::hash_identifier);

  // Parse the '='
  if (parseToken(Token::equal, "expected '=' in attribute alias definition"))
    return failure();

  // Parse the attribute value.
  Attribute attr = parseAttribute();
  if (!attr)
    return failure();

  getState().attributeAliasDefinitions[attrId] = attr;
  return success();
}

/// Parse a type alias declaration.
///
///   type-alias-def ::= '!' alias-name `=` 'type' type
///
ParseResult ModuleParser::parseTypeAliasDef() {
  assert(getToken().is(Token::exclamation_identifier));

  StringRef aliasName = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (getState().typeAliasDefinitions.count(aliasName) > 0)
    return emitError("redefinition of type alias id '" + aliasName + "'");

  // Make sure this isn't invading the dialect type namespace.
  if (aliasName.contains('.'))
    return emitError("type names with a '.' are reserved for "
                     "dialect-defined names");

  consumeToken(Token::exclamation_identifier);

  // Parse the '=' and 'type'.
  if (parseToken(Token::equal, "expected '=' in type alias definition") ||
      parseToken(Token::kw_type, "expected 'type' in type alias definition"))
    return failure();

  // Parse the type.
  Type aliasedType = parseType();
  if (!aliasedType)
    return failure();

  // Register this alias with the parser state.
  getState().typeAliasDefinitions.try_emplace(aliasName, aliasedType);

  return success();
}

/// Parse a (possibly empty) list of Function arguments with types.
///
///   named-argument ::= ssa-id `:` type attribute-dict?
///   argument-list  ::= named-argument (`,` named-argument)* | /*empty*/
///   argument-list ::= type attribute-dict? (`,` type attribute-dict?)*
///                     | /*empty*/
///
ParseResult ModuleParser::parseArgumentList(
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<StringRef> &argNames,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs) {
  consumeToken(Token::l_paren);

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's and
  // sometimes not.
  auto parseElt = [&]() -> ParseResult {
    // Parse argument name if present.
    auto loc = getToken().getLoc();
    StringRef name = getTokenSpelling();
    if (consumeIf(Token::percent_identifier)) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return emitError(loc, "expected type instead of SSA identifier");

      argNames.push_back(name);

      if (parseToken(Token::colon, "expected ':'"))
        return failure();
    } else {
      // Reject this if the preceding argument had a name.
      if (!argNames.empty())
        return emitError("expected SSA identifier");
    }

    // Parse argument type
    auto elt = parseType();
    if (!elt)
      return failure();
    argTypes.push_back(elt);

    // Parse the attribute dict.
    SmallVector<NamedAttribute, 2> attrs;
    if (getToken().is(Token::l_brace)) {
      if (parseAttributeDict(attrs))
        return failure();
    }
    argAttrs.push_back(attrs);
    return success();
  };

  return parseCommaSeparatedListUntil(Token::r_paren, parseElt);
}

/// Parse a function signature, starting with a name and including the
/// parameter list.
///
///   function-signature ::=
///      function-id `(` argument-list `)` (`->` type-list)?
///
ParseResult ModuleParser::parseFunctionSignature(
    StringRef &name, FunctionType &type, SmallVectorImpl<StringRef> &argNames,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs) {
  if (getToken().isNot(Token::at_identifier))
    return emitError("expected a function identifier like '@foo'");

  name = getTokenSpelling().drop_front();
  consumeToken(Token::at_identifier);

  if (getToken().isNot(Token::l_paren))
    return emitError("expected '(' in function signature");

  SmallVector<Type, 4> argTypes;
  if (parseArgumentList(argTypes, argNames, argAttrs))
    return failure();

  // Parse the return type if present.
  SmallVector<Type, 4> results;
  if (consumeIf(Token::arrow)) {
    if (parseFunctionResultTypes(results))
      return failure();
  }
  type = builder.getFunctionType(argTypes, results);
  return success();
}

/// Function declarations.
///
///   function ::= `func` function-signature function-attributes?
///                                          trailing-location? function-body?
///   function-body ::= `{` block+ `}`
///   function-attributes ::= `attributes` attribute-dict
///
ParseResult ModuleParser::parseFunc() {
  consumeToken();

  StringRef name;
  FunctionType type;
  SmallVector<StringRef, 4> argNames;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;

  auto loc = getToken().getLoc();
  if (parseFunctionSignature(name, type, argNames, argAttrs))
    return failure();

  // If function attributes are present, parse them.
  SmallVector<NamedAttribute, 8> attrs;
  if (consumeIf(Token::kw_attributes)) {
    if (parseAttributeDict(attrs))
      return failure();
  }

  // Okay, the function signature was parsed correctly, create the function now.
  auto *function =
      new Function(getEncodedSourceLocation(loc), name, type, attrs);
  getModule()->getFunctions().push_back(function);

  // Verify no name collision / redefinition.
  if (function->getName() != name)
    return emitError(loc, "redefinition of function named '") << name << "'";

  // Parse an optional trailing location.
  if (parseOptionalTrailingLocation(function))
    return failure();

  // Add the attributes to the function arguments.
  for (unsigned i = 0, e = function->getNumArguments(); i != e; ++i)
    function->setArgAttrs(i, argAttrs[i]);

  // External functions have no body.
  if (getToken().isNot(Token::l_brace))
    return success();

  // Create the parser.
  auto parser = FunctionParser(getState(), function);

  bool hadNamedArguments = !argNames.empty();

  // Add the entry block and argument list.
  function->addEntryBlock();

  // Add definitions of the function arguments.
  if (hadNamedArguments) {
    for (unsigned i = 0, e = function->getNumArguments(); i != e; ++i) {
      if (parser.addDefinition({argNames[i], 0, loc}, function->getArgument(i)))
        return failure();
    }
  }

  return parser.parseFunctionBody(hadNamedArguments);
}

/// Finish the end of module parsing - when the result is valid, do final
/// checking.
ParseResult ModuleParser::finalizeModule() {
  // Resolve all forward references, building a remapping table of attributes.
  DenseMap<Attribute, FunctionAttr> remappingTable;
  for (auto forwardRef : getState().functionForwardRefs) {
    auto name = forwardRef.first;

    // Resolve the reference.
    auto *resolvedFunction = getModule()->getNamedFunction(name);
    if (!resolvedFunction) {
      forwardRef.second->emitError("reference to undefined function '")
          << name << "'";
      return failure();
    }

    remappingTable[builder.getFunctionAttr(forwardRef.second)] =
        builder.getFunctionAttr(resolvedFunction);
  }

  // If there was nothing to remap, then we're done.
  if (remappingTable.empty())
    return success();

  // Otherwise, walk the entire module replacing uses of one attribute set
  // with the correct ones.
  remapFunctionAttrs(*getModule(), remappingTable);

  // Now that all references to the forward definition placeholders are
  // resolved, we can deallocate the placeholders.
  for (auto forwardRef : getState().functionForwardRefs)
    delete forwardRef.second;
  getState().functionForwardRefs.clear();
  return success();
}

/// This is the top-level module parser.
ParseResult ModuleParser::parseModule() {
  while (1) {
    switch (getToken().getKind()) {
    default:
      emitError("expected a top level entity");
      return failure();

      // If we got to the end of the file, then we're done.
    case Token::eof:
      return finalizeModule();

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  Someday we could introduce error recovery if there was demand
    // for it.
    case Token::error:
      return failure();

    case Token::hash_identifier:
      if (parseAttributeAliasDef())
        return failure();
      break;

    case Token::exclamation_identifier:
      if (parseTypeAliasDef())
        return failure();
      break;

    case Token::kw_func:
      if (parseFunc())
        return failure();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, it emits diagnostics and returns
/// null.
Module *mlir::parseSourceFile(const llvm::SourceMgr &sourceMgr,
                              MLIRContext *context) {

  // This is the result module we are parsing into.
  std::unique_ptr<Module> module(new Module(context));

  ParserState state(sourceMgr, module.get());
  if (ModuleParser(state).parseModule()) {
    return nullptr;
  }

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  if (failed(module->verify()))
    return nullptr;

  return module.release();
}

/// This parses the file specified by the indicated filename and returns an
/// MLIR module if it was valid.  If not, the error message is emitted through
/// the error handler registered in the context, and a null pointer is returned.
Module *mlir::parseSourceFile(StringRef filename, MLIRContext *context) {
  auto file_or_err = llvm::MemoryBuffer::getFile(filename);
  if (std::error_code error = file_or_err.getError()) {
    context->emitError(mlir::UnknownLoc::get(context),
                       "Could not open input file " + filename);
    return nullptr;
  }

  // Load the MLIR module.
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  return parseSourceFile(source_mgr, context);
}

/// This parses the program string to a MLIR module if it was valid. If not,
/// it emits diagnostics and returns null.
Module *mlir::parseSourceString(StringRef moduleStr, MLIRContext *context) {
  auto memBuffer = MemoryBuffer::getMemBuffer(moduleStr);
  if (!memBuffer)
    return nullptr;

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  return parseSourceFile(sourceMgr, context);
}
