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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SourceMgr;
using llvm::SMLoc;

namespace {
class CFGFunctionParserState;

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
         const SMDiagnosticHandlerTy &errorReporter)
      : context(context),
        lex(sourceMgr, errorReporter),
        curToken(lex.lexToken()),
        errorReporter(errorReporter) {
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
  const SMDiagnosticHandlerTy &errorReporter;

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

  // Polyhedral structures
  ParseResult parseAffineMapDef();

  // Functions.
  ParseResult parseFunctionSignature(StringRef &name, FunctionType *&type);
  ParseResult parseExtFunc();
  ParseResult parseCFGFunc();
  ParseResult parseBasicBlock(CFGFunctionParserState &functionState);
  TerminatorInst *parseTerminator(BasicBlock *currentBB,
                                  CFGFunctionParserState &functionState);

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
// Polyhedral structures.
//===----------------------------------------------------------------------===//

/// Affine map declaration.
///
///  affine-map-def ::= affine-map-id `=` affine-map-inline
///  affine-map-inline ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///                        ( `size` `(` dim-size (`,` dim-size)* `)` )?
///  dim-size ::= affine-expr | `min` `(` affine-expr ( `,` affine-expr)+ `)`
///
ParseResult Parser::parseAffineMapDef() {
  assert(curToken.is(Token::affine_map_id));

  StringRef affineMapId = curToken.getSpelling().drop_front();
  // Check that 'affineMapId' is unique.
  // TODO(andydavis) Add a unit test for this case.
  if (affineMaps.count(affineMapId) > 0)
    return emitError("encountered non-unique affine map id");

  consumeToken(Token::affine_map_id);

  // TODO(andydavis,bondhugula) Parse affine map definition.
  affineMaps[affineMapId].reset(new AffineMap(1, 0));
  return ParseSuccess;
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
      blockAndLoc.first = new BasicBlock(function);
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
    if (!bb->getTerminator())
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
  if (block->getTerminator())
    return emitError(nameLoc, "redefinition of block '" + name.str() + "'");

  // References to blocks can occur in any order, but we need to reassemble the
  // function in the order that occurs in the source file.  Do this by moving
  // each block to the end of the list as it is defined.
  // FIXME: This is inefficient for large functions given that blockList is a
  // vector.  blockList will eventually be an ilist, which will make this fast.
  auto &blockList = functionState.function->blockList;
  if (blockList.back() != block) {
    auto it = std::find(blockList.begin(), blockList.end(), block);
    assert(it != blockList.end() && "Block has to be in the blockList");
    std::swap(*it, blockList.back());
  }

  // TODO: parse bb argument list.

  if (!consumeIf(Token::colon))
    return emitError("expected ':' after basic block name");


  // TODO(clattner): Verify block hasn't already been parsed (this would be a
  // redefinition of the same name) once we have a body implementation.

  // TODO(clattner): Move block to the end of the list, once we have a proper
  // block list representation in CFGFunction.

  // TODO: parse instruction list.

  // TODO: Generalize this once instruction list parsing is built out.

  auto *termInst = parseTerminator(block, functionState);
  if (!termInst)
    return ParseFailure;
  block->setTerminator(termInst);

  return ParseSuccess;
}


/// Parse the terminator instruction for a basic block.
///
///   terminator-stmt ::= `br` bb-id branch-use-list?
///   branch-use-list ::= `(` ssa-use-and-type-list? `)`
///   terminator-stmt ::=
///     `cond_br` ssa-use `,` bb-id branch-use-list? `,` bb-id branch-use-list?
///   terminator-stmt ::= `return` ssa-use-and-type-list?
///
TerminatorInst *Parser::parseTerminator(BasicBlock *currentBB,
                                        CFGFunctionParserState &functionState) {
  switch (curToken.getKind()) {
  default:
    return (emitError("expected terminator at end of basic block"), nullptr);

  case Token::kw_return:
    consumeToken(Token::kw_return);
    return new ReturnInst(currentBB);

  case Token::kw_br: {
    consumeToken(Token::kw_br);
    auto destBB = functionState.getBlockNamed(curToken.getSpelling(),
                                              curToken.getLoc());
    if (!consumeIf(Token::bare_identifier))
      return (emitError("expected basic block name"), nullptr);
    return new BranchInst(destBB, currentBB);
  }
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
    case Token::affine_map_id:
      if (parseAffineMapDef()) return nullptr;
      break;

    // TODO: mlfunc, affine entity declarations, etc.
    }
  }
}

//===----------------------------------------------------------------------===//

/// This parses the file specified by the indicated SourceMgr and returns an
/// MLIR module if it was valid.  If not, it emits diagnostics and returns null.
Module *mlir::parseSourceFile(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                              const SMDiagnosticHandlerTy &errorReporter) {
  return Parser(sourceMgr, context, errorReporter).parseModule();
}
