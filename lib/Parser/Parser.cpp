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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/IR/Types.h"
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

/// Simple enum to make code read better in cases that would otherwise return a
/// bool value.  Failure is "true" in a boolean context.
enum ParseResult { ParseSuccess, ParseFailure };

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
      forwardRef.second->destroy();
    functionForwardRefs.clear();
  }

  // A map from affine map identifier to AffineMap.
  llvm::StringMap<AffineMap> affineMapDefinitions;

  // A map from integer set identifier to IntegerSet.
  llvm::StringMap<IntegerSet> integerSetDefinitions;

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

using CreateOperationFunction =
    std::function<Operation *(const OperationState &)>;

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
  ParseResult parseDimensionListRanked(SmallVectorImpl<int> &dimensions);
  Type parseTensorType();
  Type parseMemRefType();
  Type parseFunctionType();
  Type parseType();
  ParseResult parseTypeListNoParens(SmallVectorImpl<Type> &elements);
  ParseResult parseTypeList(SmallVectorImpl<Type> &elements);

  // Attribute parsing.
  Function *resolveFunctionReference(StringRef nameStr, SMLoc nameLoc,
                                     FunctionType type);
  Attribute parseAttribute(Type type = {});

  ParseResult parseAttributeDict(SmallVectorImpl<NamedAttribute> &attributes);

  // Polyhedral structures.
  void parseAffineStructureInline(AffineMap *map, IntegerSet *set);
  void parseAffineStructureReference(AffineMap *map, IntegerSet *set);
  AffineMap parseAffineMapInline();
  AffineMap parseAffineMapReference();
  IntegerSet parseIntegerSetInline();
  IntegerSet parseIntegerSetReference();
  DenseElementsAttr parseDenseElementsAttr(VectorOrTensorType type);
  DenseElementsAttr parseDenseElementsAttr(Type eltType, bool isVector);
  VectorOrTensorType parseVectorOrTensorType();

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

  getContext()->emitDiagnostic(getEncodedSourceLocation(loc), message,
                               MLIRContext::DiagnosticKind::Error);
  return ParseFailure;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult Parser::parseToken(Token::Kind expectedToken,
                               const Twine &message) {
  if (consumeIf(expectedToken))
    return ParseSuccess;
  return emitError(message);
}

/// Parse a comma separated list of elements that must have at least one entry
/// in it.
ParseResult Parser::parseCommaSeparatedList(
    const std::function<ParseResult()> &parseElement) {
  // Non-empty case starts with an element.
  if (parseElement())
    return ParseFailure;

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElement())
      return ParseFailure;
  }
  return ParseSuccess;
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
    return ParseSuccess;
  }

  if (parseCommaSeparatedList(parseElement) ||
      parseToken(rightToken, "expected ',' or '" +
                                 Token::getTokenSpelling(rightToken) + "'"))
    return ParseFailure;

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

/// Parse an arbitrary type.
///
///   type ::= integer-type
///          | index-type
///          | float-type
///          | other-type
///          | vector-type
///          | tensor-type
///          | memref-type
///          | function-type
///
///   index-type ::= `index`
///   float-type ::= `f16` | `bf16` | `f32` | `f64`
///   other-type ::= `tf_control`
///
Type Parser::parseType() {
  switch (getToken().getKind()) {
  default:
    return (emitError("expected type"), nullptr);
  case Token::kw_memref:
    return parseMemRefType();
  case Token::kw_tensor:
    return parseTensorType();
  case Token::kw_vector:
    return parseVectorType();
  case Token::l_paren:
    return parseFunctionType();
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

  // other-type
  case Token::kw_tf_control:
    consumeToken(Token::kw_tf_control);
    return builder.getTFControlType();
  case Token::kw_tf_resource:
    consumeToken(Token::kw_tf_resource);
    return builder.getTFResourceType();
  case Token::kw_tf_variant:
    consumeToken(Token::kw_tf_variant);
    return builder.getTFVariantType();
  case Token::kw_tf_complex64:
    consumeToken(Token::kw_tf_complex64);
    return builder.getTFComplex64Type();
  case Token::kw_tf_complex128:
    consumeToken(Token::kw_tf_complex128);
    return builder.getTFComplex128Type();
  case Token::kw_tf_string:
    consumeToken(Token::kw_tf_string);
    return builder.getTFStringType();
  }
}

/// Parse a vector type.
///
///   vector-type ::= `vector` `<` const-dimension-list primitive-type `>`
///   const-dimension-list ::= (integer-literal `x`)+
///
VectorType Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (parseToken(Token::less, "expected '<' in vector type"))
    return nullptr;

  if (getToken().isNot(Token::integer))
    return (emitError("expected dimension size in vector type"), nullptr);

  SmallVector<int, 4> dimensions;
  while (getToken().is(Token::integer)) {
    // Make sure this integer value is in bound and valid.
    auto dimension = getToken().getUnsignedIntegerValue();
    if (!dimension.hasValue())
      return (emitError("invalid dimension in vector type"), nullptr);
    dimensions.push_back((int)dimension.getValue());

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

  return ParseSuccess;
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
    if (parseXInDimensionList())
      return ParseFailure;
  }

  return ParseSuccess;
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
  SmallVector<int, 4> dimensions;

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

  SmallVector<int, 4> dimensions;
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
      auto affineMap = parseAffineMapReference();
      if (!affineMap)
        return ParseFailure;
      affineMapComposition.push_back(affineMap);
    }
    return ParseSuccess;
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
  if (parseTypeList(arguments) ||
      parseToken(Token::arrow, "expected '->' in function type") ||
      parseTypeList(results))
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
    return elt ? ParseSuccess : ParseFailure;
  };

  return parseCommaSeparatedList(parseElt);
}

/// Parse a "type list", which is a singular type, or a parenthesized list of
/// types.
///
///   type-list ::= type-list-parens | type
///   type-list-parens ::= `(` `)`
///                      | `(` type-list-no-parens `)`
///
ParseResult Parser::parseTypeList(SmallVectorImpl<Type> &elements) {
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseType();
    elements.push_back(elt);
    return elt ? ParseSuccess : ParseFailure;
  };

  // If there is no parens, then it must be a singular type.
  if (!consumeIf(Token::l_paren))
    return parseElt();

  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt))
    return ParseFailure;

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// Attribute parsing.
//===----------------------------------------------------------------------===//

namespace {
class TensorLiteralParser {
public:
  TensorLiteralParser(Parser &p, Type eltTy)
      : p(p), eltTy(eltTy), currBitPos(0), bitsWidth(eltTy.getBitWidth()) {}

  ParseResult parse() { return parseList(shape); }

  ArrayRef<char> getValues() const {
    return {reinterpret_cast<const char *>(storage.data()), storage.size() * 8};
  }

  ArrayRef<int> getShape() const { return shape; }

private:
  /// Parse either a single element or a list of elements. Return the dimensions
  /// of the parsed sub-tensor in dims.
  ParseResult parseElementOrList(llvm::SmallVectorImpl<int> &dims);

  /// Parse a list of either lists or elements, returning the dimensions of the
  /// parsed sub-tensors in dims. For example:
  ///   parseList([1, 2, 3]) -> Success, [3]
  ///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
  ///   parseList([[1, 2], 3]) -> Failure
  ///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
  ParseResult parseList(llvm::SmallVectorImpl<int> &dims);

  void addToStorage(uint64_t value) {
    if (bitsWidth == 64)
      storage.push_back(value);

    if (currBitPos + bitsWidth > storage.size() * 64)
      storage.push_back(0L);

    auto *rawData = reinterpret_cast<char *>(storage.data());
    DenseIntElementsAttr::writeBits(rawData, currBitPos, bitsWidth, value);
    currBitPos += bitsWidth;
  }

  Parser &p;
  Type eltTy;
  size_t currBitPos;
  size_t bitsWidth;
  SmallVector<int, 4> shape;
  std::vector<uint64_t> storage;
};
} // namespace

/// Parse either a single element or a list of elements. Return the dimensions
/// of the parsed sub-tensor in dims.
ParseResult
TensorLiteralParser::parseElementOrList(llvm::SmallVectorImpl<int> &dims) {
  switch (p.getToken().getKind()) {
  case Token::l_square:
    return parseList(dims);
  case Token::floatliteral:
  case Token::integer:
  case Token::minus: {
    auto result = p.parseAttribute();
    if (!result)
      return p.emitError("expected tensor element");
    // check result matches the element type.
    switch (eltTy.getKind()) {
    case Type::Kind::BF16:
    case Type::Kind::F16:
    case Type::Kind::F32:
    case Type::Kind::F64: {
      if (!result.isa<FloatAttr>())
        return p.emitError("expected tensor literal element has float type");
      double value = result.cast<FloatAttr>().getDouble();
      addToStorage(*(uint64_t *)(&value));
      break;
    }
    case Type::Kind::Integer: {
      if (!result.isa<IntegerAttr>())
        return p.emitError("expected tensor literal element has integer type");
      auto value = result.cast<IntegerAttr>().getValue();
      if (value.getMinSignedBits() > bitsWidth)
        return p.emitError("tensor literal element has more bits than that "
                           "specified in the type");
      addToStorage(value.getSExtValue());
      break;
    }
    default:
      return p.emitError("expected integer or float tensor element");
    }
    break;
  }
  default:
    return p.emitError("expected '[' or scalar constant inside tensor literal");
  }
  return ParseSuccess;
}

/// Parse a list of either lists or elements, returning the dimensions of the
/// parsed sub-tensors in dims. For example:
///   parseList([1, 2, 3]) -> Success, [3]
///   parseList([[1, 2], [3, 4]]) -> Success, [2, 2]
///   parseList([[1, 2], 3]) -> Failure
///   parseList([[1, [2, 3]], [4, [5]]]) -> Failure
ParseResult TensorLiteralParser::parseList(llvm::SmallVectorImpl<int> &dims) {
  p.consumeToken(Token::l_square);

  auto checkDims = [&](const llvm::SmallVectorImpl<int> &prevDims,
                       const llvm::SmallVectorImpl<int> &newDims) {
    if (prevDims == newDims)
      return ParseSuccess;
    return p.emitError("tensor literal is invalid; ranks are not consistent "
                       "between elements");
  };

  bool first = true;
  llvm::SmallVector<int, 4> newDims;
  unsigned size = 0;
  auto parseCommaSeparatedList = [&]() {
    llvm::SmallVector<int, 4> thisDims;
    if (parseElementOrList(thisDims))
      return ParseFailure;
    ++size;
    if (!first)
      return checkDims(newDims, thisDims);
    newDims = thisDims;
    first = false;
    return ParseSuccess;
  };
  if (p.parseCommaSeparatedListUntil(Token::r_square, parseCommaSeparatedList))
    return ParseFailure;

  // Return the sublists' dimensions with 'size' prepended.
  dims.clear();
  dims.push_back(size);
  dims.insert(dims.end(), newDims.begin(), newDims.end());
  return ParseSuccess;
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
      entry = new ExtFunction(getEncodedSourceLocation(nameLoc), name, type,
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
///  attribute-value ::= bool-literal
///                    | integer-literal (`:` integer-type)
///                    | float-literal (`:` float-type)
///                    | string-literal
///                    | type
///                    | `[` (attribute-value (`,` attribute-value)*)? `]`
///                    | function-id `:` function-type
///                    | (`splat<` | `dense<`) (tensor-type | vector-type)`,`
///                          attribute-value `>`
///                    | `sparse<` (tensor-type | vector-type)`,`
///                          attribute-value`, ` attribute-value `>`
///
Attribute Parser::parseAttribute(Type type) {
  switch (getToken().getKind()) {
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
    consumeToken(Token::floatliteral);
    if (!type) {
      if (consumeIf(Token::colon)) {
        if (!(type = parseType()))
          return nullptr;
      } else {
        // Default to F32 when no type is specified.
        type = builder.getF32Type();
      }
    }
    if (!type.isa<FloatType>())
      return (emitError("floating point value not valid for specified type"),
              nullptr);
    return builder.getFloatAttr(type, APFloat(val.getValue()));
  }
  case Token::integer: {
    auto val = getToken().getUInt64IntegerValue();
    if (!val.hasValue() || (int64_t)val.getValue() < 0)
      return (emitError("integer too large for attribute"), nullptr);
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
    int width = type.isIndex() ? 64 : type.getBitWidth();
    return builder.getIntegerAttr(type, APInt(width, val.getValue()));
  }

  case Token::minus: {
    consumeToken(Token::minus);
    if (getToken().is(Token::integer)) {
      auto val = getToken().getUInt64IntegerValue();
      if (!val.hasValue() || (int64_t)-val.getValue() >= 0)
        return (emitError("integer too large for attribute"), nullptr);
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
      int width = type.isIndex() ? 64 : type.getBitWidth();
      return builder.getIntegerAttr(type, -APInt(width, val.getValue()));
    }
    if (getToken().is(Token::floatliteral)) {
      auto val = getToken().getFloatingPointValue();
      if (!val.hasValue())
        return (emitError("floating point value too large for attribute"),
                nullptr);
      consumeToken(Token::floatliteral);
      if (!type) {
        if (consumeIf(Token::colon)) {
          if (!(type = parseType()))
            return nullptr;
        } else {
          // Default to F32 when no type is specified.
          type = builder.getF32Type();
        }
      }
      if (!type.isa<FloatType>())
        return (emitError("floating point value not valid for type"), nullptr);
      return builder.getFloatAttr(type, APFloat(-val.getValue()));
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
      return elements.back() ? ParseSuccess : ParseFailure;
    };

    if (parseCommaSeparatedListUntil(Token::r_square, parseElt))
      return nullptr;
    return builder.getArrayAttr(elements);
  }
  case Token::hash_identifier:
  case Token::l_paren: {
    // Try to parse an affine map or an integer set reference.
    AffineMap map;
    IntegerSet set;
    parseAffineStructureReference(&map, &set);
    if (map)
      return builder.getAffineMapAttr(map);
    if (set)
      return builder.getIntegerSetAttr(set);
    return (emitError("expected affine map or integer set attribute value"),
            nullptr);
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
    auto type = parseVectorOrTensorType();
    if (!type)
      return nullptr;
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
    return builder.getOpaqueElementsAttr(type, llvm::fromHex(val));
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
    case Token::minus: {
      auto scalar = parseAttribute(type.getElementType());
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

    switch (getToken().getKind()) {
    case Token::l_square: {
      auto attr = parseDenseElementsAttr(type);
      if (!attr)
        return nullptr;
      if (parseToken(Token::greater, "expected '>'"))
        return nullptr;
      return attr;
    }
    default:
      return (emitError("expected '[' to start dense tensor literal"), nullptr);
    }
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
      auto indicesEltType = builder.getIntegerType(32);
      auto indices =
          parseDenseElementsAttr(indicesEltType, type.isa<VectorType>());

      if (parseToken(Token::comma, "expected ','"))
        return nullptr;

      /// Parse values.
      auto valuesEltType = type.getElementType();
      auto values =
          parseDenseElementsAttr(valuesEltType, type.isa<VectorType>());

      /// Sanity check.
      auto indicesType = indices.getType();
      auto valuesType = values.getType();
      auto sameShape = (indicesType.getRank() == 1) ||
                       (type.getRank() == indicesType.getDimSize(1));
      auto sameElementNum =
          indicesType.getDimSize(0) == valuesType.getDimSize(0);
      if (!sameShape || !sameElementNum) {
        std::string str;
        llvm::raw_string_ostream s(str);
        s << "expected shape ([";
        interleaveComma(type.getShape(), s);
        s << "]); inferred shape of indices literal ([";
        interleaveComma(indicesType.getShape(), s);
        s << "]); inferred shape of values literal ([";
        interleaveComma(valuesType.getShape(), s);
        s << "])";
        return (emitError(s.str()), nullptr);
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
/// This method returns a constructed dense elements attribute with the shape
/// from the parsing result.
DenseElementsAttr Parser::parseDenseElementsAttr(Type eltType, bool isVector) {
  TensorLiteralParser literalParser(*this, eltType);
  if (literalParser.parse())
    return nullptr;

  VectorOrTensorType type;
  if (isVector) {
    type = builder.getVectorType(literalParser.getShape(), eltType);
  } else {
    type = builder.getTensorType(literalParser.getShape(), eltType);
  }
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
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "inferred shape of elements literal ([";
    interleaveComma(literalParser.getShape(), s);
    s << "]) does not match type ([";
    interleaveComma(type.getShape(), s);
    s << "])";
    return (emitError(s.str()), nullptr);
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

/// Attribute dictionary.
///
///   attribute-dict ::= `{` `}`
///                    | `{` attribute-entry (`,` attribute-entry)* `}`
///   attribute-entry ::= bare-id `:` attribute-value
///
ParseResult
Parser::parseAttributeDict(SmallVectorImpl<NamedAttribute> &attributes) {
  if (!consumeIf(Token::l_brace))
    return ParseFailure;

  auto parseElt = [&]() -> ParseResult {
    // We allow keywords as attribute names.
    if (getToken().isNot(Token::bare_identifier, Token::inttype) &&
        !getToken().isKeyword())
      return emitError("expected attribute name");
    auto nameId = builder.getIdentifier(getTokenSpelling());
    consumeToken();

    if (parseToken(Token::colon, "expected ':' in attribute list"))
      return ParseFailure;

    auto attr = parseAttribute();
    if (!attr)
      return ParseFailure;

    attributes.push_back({nameId, attr});
    return ParseSuccess;
  };

  if (parseCommaSeparatedListUntil(Token::r_brace, parseElt))
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

  void parseAffineStructureInline(AffineMap *map, IntegerSet *set);
  AffineMap parseAffineMapRange(unsigned numDims, unsigned numSymbols);
  IntegerSet parseIntegerSetConstraints(unsigned numDims, unsigned numSymbols);

private:
  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList(unsigned &numDims);
  ParseResult parseSymbolIdList(unsigned &numSymbols);
  ParseResult parseIdentifierDefinition(AffineExpr idExpr);

  AffineExpr parseAffineExpr();
  AffineExpr parseParentheticalExpr();
  AffineExpr parseNegateExpression(AffineExpr lhs);
  AffineExpr parseIntegerExpr();
  AffineExpr parseBareIdExpr();

  AffineExpr getBinaryAffineOpExpr(AffineHighPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs, SMLoc opLoc);
  AffineExpr getBinaryAffineOpExpr(AffineLowPrecOp op, AffineExpr lhs,
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
AffineExpr AffineParser::getBinaryAffineOpExpr(AffineHighPrecOp op,
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
AffineExpr AffineParser::getBinaryAffineOpExpr(AffineLowPrecOp op,
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
      AffineExpr expr = getBinaryAffineOpExpr(llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getBinaryAffineOpExpr(llhsOp, llhs, lhs, llhsOpLoc);

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
      AffineExpr sum = getBinaryAffineOpExpr(llhsOp, llhs, lhs);
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
      return emitError("redefinition of identifier '" + Twine(name) + "'");
  }
  consumeToken(Token::bare_identifier);

  dimsAndSymbols.push_back({name, idExpr});
  return ParseSuccess;
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

/// Parse the list of dimensional identifiers to an affine map.
ParseResult AffineParser::parseDimIdList(unsigned &numDims) {
  if (parseToken(Token::l_paren,
                 "expected '(' at start of dimensional identifiers list"))
    return ParseFailure;

  auto parseElt = [&]() -> ParseResult {
    auto dimension = getAffineDimExpr(numDims++, getContext());
    return parseIdentifierDefinition(dimension);
  };
  return parseCommaSeparatedListUntil(Token::r_paren, parseElt);
}

/// Parses either an affine map or an integer set definition inline. If both
/// 'map' and 'set' are non-null, parses either an affine map or an integer set.
/// If 'map' is set to nullptr, parses an integer set. If 'set' is set to
/// nullptr, parses an affine map.  'map'/'set' are set to the parsed structure.
///
///  affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        (`size` `(` dim-size (`,` dim-size)* `)`)?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
///
///
///  integer-set-inline
///                ::= dim-and-symbol-id-lists `:`
///                affine-constraint-conjunction
///  affine-constraint-conjunction ::= /*empty*/
///                                 | affine-constraint (`,`
///                                 affine-constraint)*
///
void AffineParser::parseAffineStructureInline(AffineMap *map, IntegerSet *set) {
  assert((map || set) && "one of map or set expected to be non-null");

  unsigned numDims = 0, numSymbols = 0;

  // List of dimensional identifiers.
  if (parseDimIdList(numDims)) {
    if (map)
      *map = AffineMap::Null();
    if (set)
      *set = IntegerSet::Null();
    return;
  }

  // Symbols are optional.
  if (getToken().is(Token::l_square)) {
    if (parseSymbolIdList(numSymbols)) {
      if (map)
        *map = AffineMap::Null();
      if (set)
        *set = IntegerSet::Null();
      return;
    }
  }

  // This is needed for parsing attributes as we wouldn't know whether we would
  // be parsing an integer set attribute or an affine map attribute.
  if (map && set && getToken().isNot(Token::arrow) &&
      getToken().isNot(Token::colon)) {
    emitError("expected '->' or ':' or '['");
    *map = AffineMap::Null();
    *set = IntegerSet::Null();
    return;
  }

  if (map && (!set || getToken().is(Token::arrow))) {
    // Parse an affine map.
    if (parseToken(Token::arrow, "expected '->' or '['")) {
      *map = AffineMap::Null();
      if (set)
        *set = IntegerSet::Null();
      return;
    }
    *map = parseAffineMapRange(numDims, numSymbols);
    if (set)
      *set = IntegerSet::Null();
    return;
  }

  if (set && (!map || getToken().is(Token::colon))) {
    // Parse an integer set.
    if (parseToken(Token::colon, "expected ':' or '['")) {
      *set = IntegerSet::Null();
      if (map)
        *map = AffineMap::Null();
      return;
    }
    *set = parseIntegerSetConstraints(numDims, numSymbols);
    if (map)
      *map = AffineMap::Null();
    return;
  }
}

/// Parse the range and sizes affine map definition inline.
///
///  affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        (`size` `(` dim-size (`,` dim-size)* `)`)?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
AffineMap AffineParser::parseAffineMapRange(unsigned numDims,
                                            unsigned numSymbols) {
  parseToken(Token::l_paren, "expected '(' at start of affine map range");

  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseAffineExpr();
    ParseResult res = elt ? ParseSuccess : ParseFailure;
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of
  // 1-d affine expressions); the list cannot be empty. Grammar:
  // multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)
  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt, false))
    return AffineMap::Null();

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
      return AffineMap::Null();

    auto parseRangeSize = [&]() -> ParseResult {
      auto loc = getToken().getLoc();
      auto elt = parseAffineExpr();
      if (!elt)
        return ParseFailure;

      if (!elt.isSymbolicOrConstant())
        return emitError(loc,
                         "size expressions cannot refer to dimension values");

      rangeSizes.push_back(elt);
      return ParseSuccess;
    };

    if (parseCommaSeparatedListUntil(Token::r_paren, parseRangeSize, false))
      return AffineMap::Null();
    if (exprs.size() > rangeSizes.size())
      return (emitError(loc, "fewer range sizes than range expressions"),
              AffineMap::Null());
    if (exprs.size() < rangeSizes.size())
      return (emitError(loc, "more range sizes than range expressions"),
              AffineMap::Null());
  }

  // Parsed a valid affine map.
  return builder.getAffineMap(numDims, numSymbols, exprs, rangeSizes);
}

void Parser::parseAffineStructureInline(AffineMap *map, IntegerSet *set) {
  AffineParser(state).parseAffineStructureInline(map, set);
}

AffineMap Parser::parseAffineMapInline() {
  AffineMap map;
  AffineParser(state).parseAffineStructureInline(&map, nullptr);
  return map;
}

/// Parse either an affine map reference or integer set reference.
///
///  affine-structure ::= affine-structure-id | affine-structure-inline
///  affine-structure-id ::= `#` suffix-id
///
///  affine-structure ::= affine-map | integer-set
///
void Parser::parseAffineStructureReference(AffineMap *map, IntegerSet *set) {
  assert((map || set) && "both map and set are non-null");
  if (getToken().isNot(Token::hash_identifier)) {
    // Try to parse inline affine map or integer set.
    return parseAffineStructureInline(map, set);
  }

  // Parse affine map / integer set identifier and verify that it exists.
  // Note that an id can't be in both affineMapDefinitions and
  // integerSetDefinitions since they use the same sigil '#'.
  StringRef affineStructId = getTokenSpelling().drop_front();
  if (getState().affineMapDefinitions.count(affineStructId) > 0) {
    consumeToken(Token::hash_identifier);
    if (map)
      *map = getState().affineMapDefinitions[affineStructId];
    if (set)
      *set = IntegerSet::Null();
    return;
  }

  if (getState().integerSetDefinitions.count(affineStructId) > 0) {
    consumeToken(Token::hash_identifier);
    if (set)
      *set = getState().integerSetDefinitions[affineStructId];
    if (map)
      *map = AffineMap::Null();
    return;
  }

  // The id isn't among any of the recorded definitions.
  // Emit the right message depending on what the caller expected.
  if (map && !set)
    emitError("undefined affine map id '" + affineStructId + "'");
  else if (set && !map)
    emitError("undefined integer set id '" + affineStructId + "'");
  else if (set && map)
    emitError("undefined affine map or integer set id '" + affineStructId +
              "'");

  if (map)
    *map = AffineMap::Null();
  if (set)
    *set = IntegerSet::Null();
}

/// Parse a reference to an integer set.
///  affine-map ::= affine-map-id | affine-map-inline
///  affine-map-id ::= `#` suffix-id
///
AffineMap Parser::parseAffineMapReference() {
  AffineMap map;
  parseAffineStructureReference(&map, nullptr);
  return map;
}

/// Parse a reference to an integer set.
///  integer-set ::= integer-set-id | integer-set-inline
///  integer-set-id ::= `#` suffix-id
///
IntegerSet Parser::parseIntegerSetReference() {
  IntegerSet set;
  parseAffineStructureReference(nullptr, &set);
  return set;
}

//===----------------------------------------------------------------------===//
// FunctionParser
//===----------------------------------------------------------------------===//

namespace {
/// This class contains parser state that is common across CFG and ML
/// functions, notably for dealing with operations and SSA values.
class FunctionParser : public Parser {
public:
  enum class Kind { CFGFunc, MLFunc };

  Kind getKind() const { return kind; }

  /// After the function is finished parsing, this function checks to see if
  /// there are any remaining issues.
  ParseResult finalizeFunction(Function *func, SMLoc loc);

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
  SSAValue *resolveSSAUse(SSAUseInfo useInfo, Type type);

  /// Register a definition of a value with the symbol table.
  ParseResult addDefinition(SSAUseInfo useInfo, SSAValue *value);

  // SSA parsing productions.
  ParseResult parseSSAUse(SSAUseInfo &result);
  ParseResult parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results);

  template <typename ResultType>
  ResultType parseSSADefOrUseAndType(
      const std::function<ResultType(SSAUseInfo, Type)> &action);

  SSAValue *parseSSAUseAndType() {
    return parseSSADefOrUseAndType<SSAValue *>(
        [&](SSAUseInfo useInfo, Type type) -> SSAValue * {
          return resolveSSAUse(useInfo, type);
        });
  }

  template <typename ValueTy>
  ParseResult
  parseOptionalSSAUseAndTypeList(SmallVectorImpl<ValueTy *> &results);

  // Operations
  ParseResult parseOperation(const CreateOperationFunction &createOpFunc);
  Operation *parseVerboseOperation(const CreateOperationFunction &createOpFunc);
  Operation *parseCustomOperation(const CreateOperationFunction &createOpFunc);

  /// Parse a single operation successor and it's operand list.
  virtual bool
  parseSuccessorAndUseList(BasicBlock *&dest,
                           SmallVectorImpl<SSAValue *> &operands) = 0;

protected:
  FunctionParser(ParserState &state, Kind kind) : Parser(state), kind(kind) {}

  virtual ~FunctionParser();

private:
  /// Kind indicates if this is CFG or ML function parser.
  Kind kind;
  /// This keeps track of all of the SSA values we are tracking, indexed by
  /// their name.  This has one entry per result number.
  llvm::StringMap<SmallVector<std::pair<SSAValue *, SMLoc>, 1>> values;

  /// These are all of the placeholders we've made along with the location of
  /// their first reference, to allow checking for use of undefined values.
  DenseMap<SSAValue *, SMLoc> forwardReferencePlaceholders;

  SSAValue *createForwardReferencePlaceholder(SMLoc loc, Type type);

  /// Return true if this is a forward reference.
  bool isForwardReferencePlaceholder(SSAValue *value) {
    return forwardReferencePlaceholders.count(value);
  }
};
} // end anonymous namespace

/// Create and remember a new placeholder for a forward reference.
SSAValue *FunctionParser::createForwardReferencePlaceholder(SMLoc loc,
                                                            Type type) {
  // Forward references are always created as instructions, even in ML
  // functions, because we just need something with a def/use chain.
  //
  // We create these placeholders as having an empty name, which we know
  // cannot be created through normal user input, allowing us to distinguish
  // them.
  auto name = OperationName("placeholder", getContext());
  auto *inst = Instruction::create(getEncodedSourceLocation(loc), name,
                                   /*operands=*/{}, type,
                                   /*attributes=*/{},
                                   /*successors=*/{}, getContext());
  forwardReferencePlaceholders[inst->getResult(0)] = loc;
  return inst->getResult(0);
}

/// Given an unbound reference to an SSA value and its type, return the value
/// it specifies.  This returns null on failure.
SSAValue *FunctionParser::resolveSSAUse(SSAUseInfo useInfo, Type type) {
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

  // Otherwise, this is a forward reference.  If we are in ML function return
  // an error. In CFG function, create a placeholder and remember
  // that we did so.
  if (getKind() == Kind::MLFunc)
    return (
        emitError(useInfo.loc, "use of undefined SSA value " + useInfo.name),
        nullptr);

  auto *result = createForwardReferencePlaceholder(useInfo.loc, type);
  entries[useInfo.number].first = result;
  entries[useInfo.number].second = useInfo.loc;
  return result;
}

/// Register a definition of a value with the symbol table.
ParseResult FunctionParser::addDefinition(SSAUseInfo useInfo, SSAValue *value) {
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
    existing->getDefiningInst()->destroy();
    forwardReferencePlaceholders.erase(existing);
  }

  entries[useInfo.number].first = value;
  entries[useInfo.number].second = useInfo.loc;
  return ParseSuccess;
}

/// After the function is finished parsing, this function checks to see if
/// there are any remaining issues.
ParseResult FunctionParser::finalizeFunction(Function *func, SMLoc loc) {
  // Check for any forward references that are left.  If we find any, error
  // out.
  if (!forwardReferencePlaceholders.empty()) {
    SmallVector<std::pair<const char *, SSAValue *>, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardReferencePlaceholders)
      errors.push_back({entry.second.getPointer(), entry.first});
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry.first);
      emitError(loc, "use of undeclared SSA value name");
    }
    return ParseFailure;
  }

  return ParseSuccess;
}

FunctionParser::~FunctionParser() {
  for (auto &fwd : forwardReferencePlaceholders) {
    // Drop all uses of undefined forward declared reference and destroy
    // defining instruction.
    for (auto &use : fwd.first->getUses())
      use.drop();
    fwd.first->getDefiningInst()->destroy();
  }
}

/// Parse a SSA operand for an instruction or statement.
///
///   ssa-use ::= ssa-id
///
ParseResult FunctionParser::parseSSAUse(SSAUseInfo &result) {
  result.name = getTokenSpelling();
  result.number = 0;
  result.loc = getToken().getLoc();
  if (parseToken(Token::percent_identifier, "expected SSA operand"))
    return ParseFailure;

  // If we have an affine map ID, it is a result number.
  if (getToken().is(Token::hash_identifier)) {
    if (auto value = getToken().getHashIdentifierNumber())
      result.number = value.getValue();
    else
      return emitError("invalid SSA value result number");
    consumeToken(Token::hash_identifier);
  }

  return ParseSuccess;
}

/// Parse a (possibly empty) list of SSA operands.
///
///   ssa-use-list ::= ssa-use (`,` ssa-use)*
///   ssa-use-list-opt ::= ssa-use-list?
///
ParseResult
FunctionParser::parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results) {
  if (getToken().isNot(Token::percent_identifier))
    return ParseSuccess;
  return parseCommaSeparatedList([&]() -> ParseResult {
    SSAUseInfo result;
    if (parseSSAUse(result))
      return ParseFailure;
    results.push_back(result);
    return ParseSuccess;
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
    return ParseFailure;

  // If there were no operands, then there is no colon or type lists.
  if (valueIDs.empty())
    return ParseSuccess;

  SmallVector<Type, 4> types;
  if (parseToken(Token::colon, "expected ':' in operand list") ||
      parseTypeListNoParens(types))
    return ParseFailure;

  if (valueIDs.size() != types.size())
    return emitError("expected " + Twine(valueIDs.size()) +
                     " types to match operand list");

  results.reserve(valueIDs.size());
  for (unsigned i = 0, e = valueIDs.size(); i != e; ++i) {
    if (auto *value = resolveSSAUse(valueIDs[i], types[i]))
      results.push_back(cast<ValueTy>(value));
    else
      return ParseFailure;
  }

  return ParseSuccess;
}

/// Parse the CFG or MLFunc operation.
///
///  operation ::=
///    (ssa-id `=`)? string '(' ssa-use-list? ')' attribute-dict?
///    `:` function-type
///
ParseResult
FunctionParser::parseOperation(const CreateOperationFunction &createOpFunc) {
  auto loc = getToken().getLoc();

  StringRef resultID;
  if (getToken().is(Token::percent_identifier)) {
    resultID = getTokenSpelling();
    consumeToken(Token::percent_identifier);
    if (parseToken(Token::equal, "expected '=' after SSA name"))
      return ParseFailure;
  }

  Operation *op;
  if (getToken().is(Token::bare_identifier) || getToken().isKeyword())
    op = parseCustomOperation(createOpFunc);
  else if (getToken().is(Token::string))
    op = parseVerboseOperation(createOpFunc);
  else
    return emitError("expected operation name in quotes");

  // If parsing of the basic operation failed, then this whole thing fails.
  if (!op)
    return ParseFailure;

  // We just parsed an operation.  If it is a recognized one, verify that it
  // is structurally as we expect.  If not, produce an error with a reasonable
  // source location.
  if (auto *opInfo = op->getAbstractOperation()) {
    // We don't wan't to verify branching terminators at this time because
    // the successors may not have been fully parsed yet.
    if (!(op->isTerminator() && op->getNumSuccessors() != 0) &&
        opInfo->verifyInvariants(op))
      return ParseFailure;
  }

  // If the instruction had a name, register it.
  if (!resultID.empty()) {
    if (op->getNumResults() == 0)
      return emitError(loc, "cannot name an operation with no results");

    for (unsigned i = 0, e = op->getNumResults(); i != e; ++i)
      if (addDefinition({resultID, i, loc}, op->getResult(i)))
        return ParseFailure;
  }

  return ParseSuccess;
}

Operation *FunctionParser::parseVerboseOperation(
    const CreateOperationFunction &createOpFunc) {

  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(getToken().getLoc());

  auto name = getToken().getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);
  if (name.find('\0') != StringRef::npos)
    return (emitError("null character not allowed in operation name"), nullptr);

  consumeToken(Token::string);

  OperationState result(builder.getContext(), srcLocation, name);

  // Parse the operand list.
  SmallVector<SSAUseInfo, 8> operandInfos;

  if (parseToken(Token::l_paren, "expected '(' to start operand list") ||
      parseOptionalSSAUseList(operandInfos) ||
      parseToken(Token::r_paren, "expected ')' to end operand list")) {
    return nullptr;
  }

  if (getToken().is(Token::l_brace)) {
    if (parseAttributeDict(result.attributes))
      return nullptr;
  }

  if (parseToken(Token::colon, "expected ':' followed by instruction type"))
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
    return (emitError(typeLoc, "expected " + llvm::utostr(operandInfos.size()) +
                                   " operand type" + plural + " but had " +
                                   llvm::utostr(operandTypes.size())),
            nullptr);
  }

  // Resolve all of the operands.
  for (unsigned i = 0, e = operandInfos.size(); i != e; ++i) {
    result.operands.push_back(resolveSSAUse(operandInfos[i], operandTypes[i]));
    if (!result.operands.back())
      return nullptr;
  }

  return createOpFunc(result);
}

namespace {
class CustomOpAsmParser : public OpAsmParser {
public:
  CustomOpAsmParser(SMLoc nameLoc, StringRef opName, FunctionParser &parser)
      : nameLoc(nameLoc), opName(opName), parser(parser) {}

  //===--------------------------------------------------------------------===//
  // High level parsing methods.
  //===--------------------------------------------------------------------===//

  bool getCurrentLocation(llvm::SMLoc *loc) override {
    *loc = parser.getToken().getLoc();
    return false;
  }
  bool parseComma() override {
    return parser.parseToken(Token::comma, "expected ','");
  }

  bool parseType(Type &result) override {
    return !(result = parser.parseType());
  }

  bool parseColonType(Type &result) override {
    return parser.parseToken(Token::colon, "expected ':'") ||
           !(result = parser.parseType());
  }

  bool parseColonTypeList(SmallVectorImpl<Type> &result) override {
    if (parser.parseToken(Token::colon, "expected ':'"))
      return true;

    do {
      if (auto type = parser.parseType())
        result.push_back(type);
      else
        return true;

    } while (parser.consumeIf(Token::comma));
    return false;
  }

  bool parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                                int requiredOperandCount,
                                Delimiter delimiter) override {
    if (parser.getToken().is(Token::comma)) {
      parseComma();
      return parseOperandList(result, requiredOperandCount, delimiter);
    }
    if (requiredOperandCount != -1)
      return emitError(parser.getToken().getLoc(),
                       "expected " + Twine(requiredOperandCount) + " operands");
    return false;
  }

  /// Parse a keyword followed by a type.
  bool parseKeywordType(const char *keyword, Type &result) override {
    if (parser.getTokenSpelling() != keyword)
      return parser.emitError("expected '" + Twine(keyword) + "'");
    parser.consumeToken();
    return !(result = parser.parseType());
  }

  /// Parse an arbitrary attribute of a given type and return it in result. This
  /// also adds the attribute to the specified attribute list with the specified
  /// name.
  bool parseAttribute(Attribute &result, Type type, const char *attrName,
                      SmallVectorImpl<NamedAttribute> &attrs) override {
    result = parser.parseAttribute(type);
    if (!result)
      return true;

    attrs.push_back(
        NamedAttribute(parser.builder.getIdentifier(attrName), result));
    return false;
  }

  /// Parse an arbitrary attribute and return it in result.  This also adds
  /// the attribute to the specified attribute list with the specified name.
  bool parseAttribute(Attribute &result, const char *attrName,
                      SmallVectorImpl<NamedAttribute> &attrs) override {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// If a named attribute list is present, parse is into result.
  bool
  parseOptionalAttributeDict(SmallVectorImpl<NamedAttribute> &result) override {
    if (parser.getToken().isNot(Token::l_brace))
      return false;
    return parser.parseAttributeDict(result) == ParseFailure;
  }

  /// Parse a function name like '@foo' and return the name in a form that can
  /// be passed to resolveFunctionName when a function type is available.
  virtual bool parseFunctionName(StringRef &result, llvm::SMLoc &loc) {
    loc = parser.getToken().getLoc();

    if (parser.getToken().isNot(Token::at_identifier))
      return emitError(loc, "expected function name");

    result = parser.getTokenSpelling();
    parser.consumeToken(Token::at_identifier);
    return false;
  }

  bool parseOperand(OperandType &result) override {
    FunctionParser::SSAUseInfo useInfo;
    if (parser.parseSSAUse(useInfo))
      return true;

    result = {useInfo.loc, useInfo.name, useInfo.number};
    return false;
  }

  bool
  parseSuccessorAndUseList(BasicBlock *&dest,
                           SmallVectorImpl<SSAValue *> &operands) override {
    // Defer successor parsing to the function parsers.
    return parser.parseSuccessorAndUseList(dest, operands);
  }

  bool parseOperandList(SmallVectorImpl<OperandType> &result,
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
        return false;
      LLVM_FALLTHROUGH;
    case Delimiter::Paren:
      if (parser.parseToken(Token::l_paren, "expected '(' in operand list"))
        return true;
      break;
    case Delimiter::OptionalSquare:
      if (parser.getToken().isNot(Token::l_square))
        return false;
      LLVM_FALLTHROUGH;
    case Delimiter::Square:
      if (parser.parseToken(Token::l_square, "expected '[' in operand list"))
        return true;
      break;
    }

    // Check for zero operands.
    if (parser.getToken().is(Token::percent_identifier)) {
      do {
        OperandType operand;
        if (parseOperand(operand))
          return true;
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
        return true;
      break;
    case Delimiter::OptionalSquare:
    case Delimiter::Square:
      if (parser.parseToken(Token::r_square, "expected ']' in operand list"))
        return true;
      break;
    }

    if (requiredOperandCount != -1 && result.size() != requiredOperandCount)
      return emitError(startLoc,
                       "expected " + Twine(requiredOperandCount) + " operands");
    return false;
  }

  /// Resolve a parse function name and a type into a function reference.
  virtual bool resolveFunctionName(StringRef name, FunctionType type,
                                   llvm::SMLoc loc, Function *&result) {
    result = parser.resolveFunctionReference(name, loc, type);
    return result == nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Methods for interacting with the parser
  //===--------------------------------------------------------------------===//

  Builder &getBuilder() const override { return parser.builder; }

  llvm::SMLoc getNameLoc() const override { return nameLoc; }

  bool resolveOperand(const OperandType &operand, Type type,
                      SmallVectorImpl<SSAValue *> &result) override {
    FunctionParser::SSAUseInfo operandInfo = {operand.name, operand.number,
                                              operand.location};
    if (auto *value = parser.resolveSSAUse(operandInfo, type)) {
      result.push_back(value);
      return false;
    }
    return true;
  }

  /// Emit a diagnostic at the specified location and return true.
  bool emitError(llvm::SMLoc loc, const Twine &message) override {
    parser.emitError(loc, "custom op '" + Twine(opName) + "' " + message);
    emittedError = true;
    return true;
  }

  bool didEmitError() const { return emittedError; }

private:
  SMLoc nameLoc;
  StringRef opName;
  FunctionParser &parser;
  bool emittedError = false;
};
} // end anonymous namespace.

Operation *FunctionParser::parseCustomOperation(
    const CreateOperationFunction &createOpFunc) {
  auto opLoc = getToken().getLoc();
  auto opName = getTokenSpelling();
  CustomOpAsmParser opAsmParser(opLoc, opName, *this);

  auto *opDefinition = AbstractOperation::lookup(opName, getContext());
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
  OperationState opState(builder.getContext(), srcLocation, opName);
  if (opDefinition->parseAssembly(&opAsmParser, &opState))
    return nullptr;

  // If it emitted an error, we failed.
  if (opAsmParser.didEmitError())
    return nullptr;

  // Otherwise, we succeeded.  Use the state it parsed as our op information.
  return createOpFunc(opState);
}

//===----------------------------------------------------------------------===//
// CFG Functions
//===----------------------------------------------------------------------===//

namespace {

/// This is a specialized parser for CFGFunction's, maintaining the state
/// transient to their bodies.
class CFGFunctionParser : public FunctionParser {
public:
  CFGFunctionParser(ParserState &state, CFGFunction *function)
      : FunctionParser(state, Kind::CFGFunc), function(function),
        builder(function) {}

  ParseResult parseFunctionBody();

  bool parseSuccessorAndUseList(BasicBlock *&dest,
                                SmallVectorImpl<SSAValue *> &operands);

private:
  CFGFunction *function;
  llvm::StringMap<std::pair<BasicBlock *, SMLoc>> blocksByName;
  DenseMap<BasicBlock *, SMLoc> forwardRef;

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
      forwardRef[blockAndLoc.first] = loc;
      function->push_back(blockAndLoc.first);
      blockAndLoc.second = loc;
    }

    return blockAndLoc.first;
  }

  // Define the basic block with the specified name. Returns the BasicBlock* or
  // nullptr in the case of redefinition.
  BasicBlock *defineBlockNamed(StringRef name, SMLoc loc) {
    auto &blockAndLoc = blocksByName[name];
    if (!blockAndLoc.first) {
      blockAndLoc.first = builder.createBlock();
      blockAndLoc.second = loc;
      return blockAndLoc.first;
    }

    // Forward declarations are removed once defined, so if we are defining a
    // existing block and it is not a forward declaration, then it is a
    // redeclaration.
    if (!forwardRef.erase(blockAndLoc.first))
      return nullptr;

    // Move the block to the end of the function.  Forward ref'd blocks are
    // inserted wherever they happen to be referenced.
    function->getBlocks().splice(function->end(), function->getBlocks(),
                                 blockAndLoc.first);
    return blockAndLoc.first;
  }

  ParseResult
  parseOptionalBasicBlockArgList(SmallVectorImpl<BBArgument *> &results,
                                 BasicBlock *owner);

  ParseResult parseBasicBlock();
};
} // end anonymous namespace

/// Parse a single operation successor and it's operand list.
///
///   successor ::= bb-id branch-use-list?
///   branch-use-list ::= `(` ssa-use-list ':' type-list-no-parens `)`
///
bool CFGFunctionParser::parseSuccessorAndUseList(
    BasicBlock *&dest, SmallVectorImpl<SSAValue *> &operands) {
  // Verify branch is identifier and get the matching block.
  if (!getToken().is(Token::bare_identifier))
    return emitError("expected basic block name");
  dest = getBlockNamed(getTokenSpelling(), getToken().getLoc());
  consumeToken();

  // Handle optional arguments.
  if (consumeIf(Token::l_paren) &&
      (parseOptionalSSAUseAndTypeList(operands) ||
       parseToken(Token::r_paren, "expected ')' to close argument list"))) {
    return true;
  }

  return false;
}

/// Parse a (possibly empty) list of SSA operands with types as basic block
/// arguments.
///
///   ssa-id-and-type-list ::= ssa-id-and-type (`,` ssa-id-and-type)*
///
ParseResult CFGFunctionParser::parseOptionalBasicBlockArgList(
    SmallVectorImpl<BBArgument *> &results, BasicBlock *owner) {
  if (getToken().is(Token::r_brace))
    return ParseSuccess;

  return parseCommaSeparatedList([&]() -> ParseResult {
    auto type = parseSSADefOrUseAndType<Type>(
        [&](SSAUseInfo useInfo, Type type) -> Type {
          BBArgument *arg = owner->addArgument(type);
          if (addDefinition(useInfo, arg))
            return {};
          return type;
        });
    return type ? ParseSuccess : ParseFailure;
  });
}

ParseResult CFGFunctionParser::parseFunctionBody() {
  auto braceLoc = getToken().getLoc();
  if (parseToken(Token::l_brace, "expected '{' in CFG function"))
    return ParseFailure;

  // Make sure we have at least one block.
  if (getToken().is(Token::r_brace))
    return emitError("CFG functions must have at least one basic block");

  // Parse the list of blocks.
  while (!consumeIf(Token::r_brace))
    if (parseBasicBlock())
      return ParseFailure;

  // Verify that all referenced blocks were defined.
  if (!forwardRef.empty()) {
    SmallVector<std::pair<const char *, BasicBlock *>, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardRef)
      errors.push_back({entry.second.getPointer(), entry.first});
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry.first);
      emitError(loc, "reference to an undefined basic block");
    }
    return ParseFailure;
  }

  // Now that the function body has been fully parsed we check the invariants
  // of any branching terminators.
  for (auto &block : *function) {
    auto *term = block.getTerminator();
    auto *abstractOp = term->getAbstractOperation();
    if (term->getNumSuccessors() != 0 && abstractOp)
      abstractOp->verifyInvariants(term);
  }

  return finalizeFunction(function, braceLoc);
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
  if (parseToken(Token::bare_identifier, "expected basic block name"))
    return ParseFailure;

  auto *block = defineBlockNamed(name, nameLoc);

  // Fail if redefinition.
  if (!block)
    return emitError(nameLoc, "redefinition of block '" + name.str() + "'");

  // If an argument list is present, parse it.
  if (consumeIf(Token::l_paren)) {
    SmallVector<BBArgument *, 8> bbArgs;
    if (parseOptionalBasicBlockArgList(bbArgs, block) ||
        parseToken(Token::r_paren, "expected ')' to end argument list"))
      return ParseFailure;
  }

  if (parseToken(Token::colon, "expected ':' after basic block name"))
    return ParseFailure;

  // Set the insertion point to the block we want to insert new operations
  // into.
  builder.setInsertionPoint(block);

  auto createOpFunc = [&](const OperationState &result) -> Operation * {
    return builder.createOperation(result);
  };

  // Parse the list of operations that make up the body of the block.
  while (getToken().isNot(Token::kw_return, Token::kw_br, Token::kw_cond_br)) {
    if (parseOperation(createOpFunc))
      return ParseFailure;
  }

  // Parse the terminator operation.
  if (parseOperation(createOpFunc))
    return ParseFailure;

  return ParseSuccess;
}

//===----------------------------------------------------------------------===//
// ML Functions
//===----------------------------------------------------------------------===//

namespace {
/// Refined parser for MLFunction bodies.
class MLFunctionParser : public FunctionParser {
public:
  MLFunctionParser(ParserState &state, MLFunction *function)
      : FunctionParser(state, Kind::MLFunc), function(function),
        builder(function, function->end()) {}

  ParseResult parseFunctionBody();

private:
  MLFunction *function;

  /// This builder intentionally shadows the builder in the base class, with a
  /// more specific builder type.
  MLFuncBuilder builder;

  ParseResult parseForStmt();
  ParseResult parseIntConstant(int64_t &val);
  ParseResult parseDimAndSymbolList(SmallVectorImpl<MLValue *> &operands,
                                    unsigned numDims, unsigned numOperands,
                                    const char *affineStructName);
  ParseResult parseBound(SmallVectorImpl<MLValue *> &operands, AffineMap &map,
                         bool isLower);
  ParseResult parseIfStmt();
  ParseResult parseElseClause(IfClause *elseClause);
  ParseResult parseStatements(StmtBlock *block);
  ParseResult parseStmtBlock(StmtBlock *block);

  bool parseSuccessorAndUseList(BasicBlock *&dest,
                                SmallVectorImpl<SSAValue *> &operands) {
    assert(false && "MLFunctions do not have terminators with successors.");
    return true;
  }
};
} // end anonymous namespace

ParseResult MLFunctionParser::parseFunctionBody() {
  auto braceLoc = getToken().getLoc();

  // Parse statements in this function.
  if (parseStmtBlock(function))
    return ParseFailure;

  return finalizeFunction(function, braceLoc);
}

/// For statement.
///
///    ml-for-stmt ::= `for` ssa-id `=` lower-bound `to` upper-bound
///                   (`step` integer-literal)? `{` ml-stmt* `}`
///
ParseResult MLFunctionParser::parseForStmt() {
  consumeToken(Token::kw_for);

  // Parse induction variable.
  if (getToken().isNot(Token::percent_identifier))
    return emitError("expected SSA identifier for the loop variable");

  auto loc = getToken().getLoc();
  StringRef inductionVariableName = getTokenSpelling();
  consumeToken(Token::percent_identifier);

  if (parseToken(Token::equal, "expected '='"))
    return ParseFailure;

  // Parse lower bound.
  SmallVector<MLValue *, 4> lbOperands;
  AffineMap lbMap;
  if (parseBound(lbOperands, lbMap, /*isLower*/ true))
    return ParseFailure;

  if (parseToken(Token::kw_to, "expected 'to' between bounds"))
    return ParseFailure;

  // Parse upper bound.
  SmallVector<MLValue *, 4> ubOperands;
  AffineMap ubMap;
  if (parseBound(ubOperands, ubMap, /*isLower*/ false))
    return ParseFailure;

  // Parse step.
  int64_t step = 1;
  if (consumeIf(Token::kw_step) && parseIntConstant(step))
    return ParseFailure;

  // The loop step is a positive integer constant. Since index is stored as an
  // int64_t type, we restrict step to be in the set of positive integers that
  // int64_t can represent.
  if (step < 1) {
    return emitError("step has to be a positive integer");
  }

  // Create for statement.
  ForStmt *forStmt =
      builder.createFor(getEncodedSourceLocation(loc), lbOperands, lbMap,
                        ubOperands, ubMap, step);

  // Create SSA value definition for the induction variable.
  if (addDefinition({inductionVariableName, 0, loc}, forStmt))
    return ParseFailure;

  // If parsing of the for statement body fails,
  // MLIR contains for statement with those nested statements that have been
  // successfully parsed.
  if (parseStmtBlock(forStmt))
    return ParseFailure;

  // Reset insertion point to the current block.
  builder.setInsertionPointToEnd(forStmt->getBlock());

  return ParseSuccess;
}

/// Parse integer constant as affine constant expression.
ParseResult MLFunctionParser::parseIntConstant(int64_t &val) {
  bool negate = consumeIf(Token::minus);

  if (getToken().isNot(Token::integer))
    return emitError("expected integer");

  auto uval = getToken().getUInt64IntegerValue();

  if (!uval.hasValue() || (int64_t)uval.getValue() < 0) {
    return emitError("bound or step is too large for index");
  }

  val = (int64_t)uval.getValue();
  if (negate)
    val = -val;
  consumeToken();

  return ParseSuccess;
}

/// Dimensions and symbol use list.
///
/// dim-use-list ::= `(` ssa-use-list? `)`
/// symbol-use-list ::= `[` ssa-use-list? `]`
/// dim-and-symbol-use-list ::= dim-use-list symbol-use-list?
///
ParseResult
MLFunctionParser::parseDimAndSymbolList(SmallVectorImpl<MLValue *> &operands,
                                        unsigned numDims, unsigned numOperands,
                                        const char *affineStructName) {
  if (parseToken(Token::l_paren, "expected '('"))
    return ParseFailure;

  SmallVector<SSAUseInfo, 4> opInfo;
  parseOptionalSSAUseList(opInfo);

  if (parseToken(Token::r_paren, "expected ')'"))
    return ParseFailure;

  if (numDims != opInfo.size())
    return emitError("dim operand count and " + Twine(affineStructName) +
                     " dim count must match");

  if (consumeIf(Token::l_square)) {
    parseOptionalSSAUseList(opInfo);
    if (parseToken(Token::r_square, "expected ']'"))
      return ParseFailure;
  }

  if (numOperands != opInfo.size())
    return emitError("symbol operand count and " + Twine(affineStructName) +
                     " symbol count must match");

  // Resolve SSA uses.
  Type indexType = builder.getIndexType();
  for (unsigned i = 0, e = opInfo.size(); i != e; ++i) {
    SSAValue *sval = resolveSSAUse(opInfo[i], indexType);
    if (!sval)
      return ParseFailure;

    auto *v = cast<MLValue>(sval);
    if (i < numDims && !v->isValidDim())
      return emitError(opInfo[i].loc, "value '" + opInfo[i].name.str() +
                                          "' cannot be used as a dimension id");
    if (i >= numDims && !v->isValidSymbol())
      return emitError(opInfo[i].loc, "value '" + opInfo[i].name.str() +
                                          "' cannot be used as a symbol");
    operands.push_back(v);
  }

  return ParseSuccess;
}

// Loop bound.
///
///  lower-bound ::= `max`? affine-map dim-and-symbol-use-list |
///  shorthand-bound upper-bound ::= `min`? affine-map dim-and-symbol-use-list
///  | shorthand-bound shorthand-bound ::= ssa-id | `-`? integer-literal
///
ParseResult MLFunctionParser::parseBound(SmallVectorImpl<MLValue *> &operands,
                                         AffineMap &map, bool isLower) {
  // 'min' / 'max' prefixes are syntactic sugar. Ignore them.
  if (isLower)
    consumeIf(Token::kw_max);
  else
    consumeIf(Token::kw_min);

  // Parse full form - affine map followed by dim and symbol list.
  if (getToken().isAny(Token::hash_identifier, Token::l_paren)) {
    map = parseAffineMapReference();
    if (!map)
      return ParseFailure;

    if (parseDimAndSymbolList(operands, map.getNumDims(), map.getNumInputs(),
                              "affine map"))
      return ParseFailure;
    return ParseSuccess;
  }

  // Parse shorthand form.
  if (getToken().isAny(Token::minus, Token::integer)) {
    int64_t val;
    if (!parseIntConstant(val)) {
      map = builder.getConstantAffineMap(val);
      return ParseSuccess;
    }
    return ParseFailure;
  }

  // Parse ssa-id as identity map.
  SSAUseInfo opInfo;
  if (parseSSAUse(opInfo))
    return ParseFailure;

  // TODO: improve error message when SSA value is not an affine integer.
  // Currently it is 'use of value ... expects different type than prior uses'
  if (auto *value = resolveSSAUse(opInfo, builder.getIndexType()))
    operands.push_back(cast<MLValue>(value));
  else
    return ParseFailure;

  // Create an identity map using dim id for an induction variable and
  // symbol otherwise. This representation is optimized for storage.
  // Analysis passes may expand it into a multi-dimensional map if desired.
  if (isa<ForStmt>(operands[0]))
    map = builder.getDimIdentityMap();
  else
    map = builder.getSymbolIdentityMap();

  return ParseSuccess;
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
///                affine-constraint-conjunction
///  affine-constraint-conjunction ::= /*empty*/
///                                 | affine-constraint (`,`
///                                 affine-constraint)*
///
IntegerSet AffineParser::parseIntegerSetConstraints(unsigned numDims,
                                                    unsigned numSymbols) {

  parseToken(Token::l_paren,
             "expected '(' at start of integer set constraint list");
  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;
  auto parseElt = [&]() -> ParseResult {
    bool isEq;
    auto elt = parseAffineConstraint(&isEq);
    ParseResult res = elt ? ParseSuccess : ParseFailure;
    if (elt) {
      constraints.push_back(elt);
      isEqs.push_back(isEq);
    }
    return res;
  };

  // Parse a list of affine constraints (comma-separated) .
  // Grammar: affine-constraint-conjunct ::= `(` affine-constraint (`,`
  // affine-constraint)* `)
  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt, true))
    return IntegerSet();

  // Parsed a valid integer set.
  return builder.getIntegerSet(numDims, numSymbols, constraints, isEqs);
}

IntegerSet Parser::parseIntegerSetInline() {
  IntegerSet set;
  AffineParser(state).parseAffineStructureInline(nullptr, &set);
  return set;
}

/// If statement.
///
///   ml-if-head ::= `if` ml-if-cond `{` ml-stmt* `}`
///               | ml-if-head `else` `if` ml-if-cond `{` ml-stmt* `}`
///   ml-if-stmt ::= ml-if-head
///               | ml-if-head `else` `{` ml-stmt* `}`
///
ParseResult MLFunctionParser::parseIfStmt() {
  auto loc = getToken().getLoc();
  consumeToken(Token::kw_if);

  IntegerSet set = parseIntegerSetReference();
  if (!set)
    return ParseFailure;

  SmallVector<MLValue *, 4> operands;
  if (parseDimAndSymbolList(operands, set.getNumDims(), set.getNumOperands(),
                            "integer set"))
    return ParseFailure;

  IfStmt *ifStmt =
      builder.createIf(getEncodedSourceLocation(loc), operands, set);

  IfClause *thenClause = ifStmt->getThen();

  // When parsing of an if statement body fails, the IR contains
  // the if statement with the portion of the body that has been
  // successfully parsed.
  if (parseStmtBlock(thenClause))
    return ParseFailure;

  if (consumeIf(Token::kw_else)) {
    auto *elseClause = ifStmt->createElse();
    if (parseElseClause(elseClause))
      return ParseFailure;
  }

  // Reset insertion point to the current block.
  builder.setInsertionPointToEnd(ifStmt->getBlock());

  return ParseSuccess;
}

ParseResult MLFunctionParser::parseElseClause(IfClause *elseClause) {
  if (getToken().is(Token::kw_if)) {
    builder.setInsertionPointToEnd(elseClause);
    return parseIfStmt();
  }

  return parseStmtBlock(elseClause);
}

///
/// Parse a list of statements ending with `return` or `}`
///
ParseResult MLFunctionParser::parseStatements(StmtBlock *block) {
  auto createOpFunc = [&](const OperationState &state) -> Operation * {
    return builder.createOperation(state);
  };

  builder.setInsertionPointToEnd(block);

  // Parse statements till we see '}' or 'return'.
  // Return statement is parsed separately to emit a more intuitive error
  // when '}' is missing after the return statement.
  while (getToken().isNot(Token::r_brace, Token::kw_return)) {
    switch (getToken().getKind()) {
    default:
      if (parseOperation(createOpFunc))
        return ParseFailure;
      break;
    case Token::kw_for:
      if (parseForStmt())
        return ParseFailure;
      break;
    case Token::kw_if:
      if (parseIfStmt())
        return ParseFailure;
      break;
    } // end switch
  }

  // Parse the return statement.
  if (getToken().is(Token::kw_return))
    if (parseOperation(createOpFunc))
      return ParseFailure;

  return ParseSuccess;
}

///
/// Parse `{` ml-stmt* `}`
///
ParseResult MLFunctionParser::parseStmtBlock(StmtBlock *block) {
  if (parseToken(Token::l_brace, "expected '{' before statement list") ||
      parseStatements(block) ||
      parseToken(Token::r_brace, "expected '}' after statement list"))
    return ParseFailure;

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
  ParseResult finalizeModule();

  ParseResult parseAffineStructureDef();

  // Functions.
  ParseResult parseMLArgumentList(SmallVectorImpl<Type> &argTypes,
                                  SmallVectorImpl<StringRef> &argNames);
  ParseResult parseFunctionSignature(StringRef &name, FunctionType &type,
                                     SmallVectorImpl<StringRef> *argNames);
  ParseResult parseFunctionAttribute(SmallVectorImpl<NamedAttribute> &attrs);
  ParseResult parseExtFunc();
  ParseResult parseCFGFunc();
  ParseResult parseMLFunc();
};
} // end anonymous namespace

/// Parses either an affine map declaration or an integer set declaration.
///
/// Affine map declaration.
///
///   affine-map-def ::= affine-map-id `=` affine-map-inline
///
/// Integer set declaration.
///
///  integer-set-decl ::= integer-set-id `=` integer-set-inline
///
ParseResult ModuleParser::parseAffineStructureDef() {
  assert(getToken().is(Token::hash_identifier));

  StringRef affineStructureId = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (getState().affineMapDefinitions.count(affineStructureId) > 0)
    return emitError("redefinition of affine map id '" + affineStructureId +
                     "'");
  if (getState().integerSetDefinitions.count(affineStructureId) > 0)
    return emitError("redefinition of integer set id '" + affineStructureId +
                     "'");

  consumeToken(Token::hash_identifier);

  // Parse the '='
  if (parseToken(Token::equal,
                 "expected '=' in affine map outlined definition"))
    return ParseFailure;

  AffineMap map;
  IntegerSet set;
  parseAffineStructureInline(&map, &set);
  if (!map && !set)
    return ParseFailure;

  if (map)
    getState().affineMapDefinitions[affineStructureId] = map;
  else
    getState().integerSetDefinitions[affineStructureId] = set;

  return ParseSuccess;
}

/// Parse a (possibly empty) list of MLFunction arguments with types.
///
/// ml-argument      ::= ssa-id `:` type
/// ml-argument-list ::= ml-argument (`,` ml-argument)* | /*empty*/
///
ParseResult
ModuleParser::parseMLArgumentList(SmallVectorImpl<Type> &argTypes,
                                  SmallVectorImpl<StringRef> &argNames) {
  consumeToken(Token::l_paren);

  auto parseElt = [&]() -> ParseResult {
    // Parse argument name
    if (getToken().isNot(Token::percent_identifier))
      return emitError("expected SSA identifier");

    StringRef name = getTokenSpelling();
    consumeToken(Token::percent_identifier);
    argNames.push_back(name);

    if (parseToken(Token::colon, "expected ':'"))
      return ParseFailure;

    // Parse argument type
    auto elt = parseType();
    if (!elt)
      return ParseFailure;
    argTypes.push_back(elt);

    return ParseSuccess;
  };

  return parseCommaSeparatedListUntil(Token::r_paren, parseElt);
}

/// Parse a function signature, starting with a name and including the
/// parameter list.
///
///   argument-list ::= type (`,` type)* | /*empty*/ | ml-argument-list
///   function-signature ::= function-id `(` argument-list `)` (`->`
///   type-list)?
///
ParseResult
ModuleParser::parseFunctionSignature(StringRef &name, FunctionType &type,
                                     SmallVectorImpl<StringRef> *argNames) {
  if (getToken().isNot(Token::at_identifier))
    return emitError("expected a function identifier like '@foo'");

  name = getTokenSpelling().drop_front();
  consumeToken(Token::at_identifier);

  if (getToken().isNot(Token::l_paren))
    return emitError("expected '(' in function signature");

  SmallVector<Type, 4> argTypes;
  ParseResult parseResult;

  if (argNames)
    parseResult = parseMLArgumentList(argTypes, *argNames);
  else
    parseResult = parseTypeList(argTypes);

  if (parseResult)
    return ParseFailure;

  // Parse the return type if present.
  SmallVector<Type, 4> results;
  if (consumeIf(Token::arrow)) {
    if (parseTypeList(results))
      return ParseFailure;
  }
  type = builder.getFunctionType(argTypes, results);
  return ParseSuccess;
}

/// Parse function attributes, starting with keyword "attributes".
///
///   function-attribute ::= (`attributes` attribute-dict)?
///
ParseResult
ModuleParser::parseFunctionAttribute(SmallVectorImpl<NamedAttribute> &attrs) {
  if (consumeIf(Token::kw_attributes)) {
    if (parseAttributeDict(attrs)) {
      return ParseFailure;
    }
  }
  return ParseSuccess;
}

/// External function declarations.
///
///   ext-func ::= `extfunc` function-signature
///                (`attributes` attribute-dict)?
///
ParseResult ModuleParser::parseExtFunc() {
  consumeToken(Token::kw_extfunc);
  auto loc = getToken().getLoc();

  StringRef name;
  FunctionType type;
  if (parseFunctionSignature(name, type, /*arguments*/ nullptr))
    return ParseFailure;

  SmallVector<NamedAttribute, 8> attrs;
  if (parseFunctionAttribute(attrs)) {
    return ParseFailure;
  }

  // Okay, the external function definition was parsed correctly.
  auto *function =
      new ExtFunction(getEncodedSourceLocation(loc), name, type, attrs);
  getModule()->getFunctions().push_back(function);

  // Verify no name collision / redefinition.
  if (function->getName() != name)
    return emitError(loc,
                     "redefinition of function named '" + name.str() + "'");

  return ParseSuccess;
}

/// CFG function declarations.
///
///   cfg-func ::= `cfgfunc` function-signature
///               (`attributes` attribute-dict)? `{` basic-block+ `}`
///
ParseResult ModuleParser::parseCFGFunc() {
  consumeToken(Token::kw_cfgfunc);
  auto loc = getToken().getLoc();

  StringRef name;
  FunctionType type;
  if (parseFunctionSignature(name, type, /*arguments*/ nullptr))
    return ParseFailure;

  SmallVector<NamedAttribute, 8> attrs;
  if (parseFunctionAttribute(attrs)) {
    return ParseFailure;
  }

  // Okay, the CFG function signature was parsed correctly, create the
  // function.
  auto *function =
      new CFGFunction(getEncodedSourceLocation(loc), name, type, attrs);
  getModule()->getFunctions().push_back(function);

  // Verify no name collision / redefinition.
  if (function->getName() != name)
    return emitError(loc,
                     "redefinition of function named '" + name.str() + "'");

  return CFGFunctionParser(getState(), function).parseFunctionBody();
}

/// ML function declarations.
///
///   ml-func ::= `mlfunc` ml-func-signature
///              (`attributes` attribute-dict)? `{` ml-stmt* ml-return-stmt
///              `}`
///
ParseResult ModuleParser::parseMLFunc() {
  consumeToken(Token::kw_mlfunc);

  StringRef name;
  FunctionType type;
  SmallVector<StringRef, 4> argNames;

  auto loc = getToken().getLoc();
  if (parseFunctionSignature(name, type, &argNames))
    return ParseFailure;

  SmallVector<NamedAttribute, 8> attrs;
  if (parseFunctionAttribute(attrs)) {
    return ParseFailure;
  }

  // Okay, the ML function signature was parsed correctly, create the
  // function.
  auto *function =
      MLFunction::create(getEncodedSourceLocation(loc), name, type, attrs);
  getModule()->getFunctions().push_back(function);

  // Verify no name collision / redefinition.
  if (function->getName() != name)
    return emitError(loc,
                     "redefinition of function named '" + name.str() + "'");

  // Create the parser.
  auto parser = MLFunctionParser(getState(), function);

  // Add definitions of the function arguments.
  for (unsigned i = 0, e = function->getNumArguments(); i != e; ++i) {
    if (parser.addDefinition({argNames[i], 0, loc}, function->getArgument(i)))
      return ParseFailure;
  }

  return parser.parseFunctionBody();
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
      forwardRef.second->emitError("reference to undefined function '" +
                                   name.str() + "'");
      return ParseFailure;
    }

    remappingTable[builder.getFunctionAttr(forwardRef.second)] =
        builder.getFunctionAttr(resolvedFunction);
  }

  // If there was nothing to remap, then we're done.
  if (remappingTable.empty())
    return ParseSuccess;

  // Otherwise, walk the entire module replacing uses of one attribute set
  // with the correct ones.
  remapFunctionAttrs(*getModule(), remappingTable);

  // Now that all references to the forward definition placeholders are
  // resolved, we can deallocate the placeholders.
  for (auto forwardRef : getState().functionForwardRefs)
    forwardRef.second->destroy();
  getState().functionForwardRefs.clear();
  return ParseSuccess;
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
      return finalizeModule();

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  Someday we could introduce error recovery if there was demand
    // for it.
    case Token::error:
      return ParseFailure;

    case Token::hash_identifier:
      if (parseAffineStructureDef())
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
  if (module->verify())
    return nullptr;

  return module.release();
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
