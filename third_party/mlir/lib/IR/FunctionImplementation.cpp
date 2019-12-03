//===- FunctionImplementation.cpp - Utilities for function-like ops -------===//
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

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

static ParseResult
parseArgumentList(OpAsmParser &parser, bool allowVariadic,
                  SmallVectorImpl<Type> &argTypes,
                  SmallVectorImpl<OpAsmParser::OperandType> &argNames,
                  SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs,
                  bool &isVariadic) {
  if (parser.parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's and
  // sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::OperandType argument;
    Type argumentType;
    if (succeeded(parser.parseOptionalRegionArgument(argument)) &&
        !argument.name.empty()) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return parser.emitError(loc, "expected type instead of SSA identifier");
      argNames.push_back(argument);

      if (parser.parseColonType(argumentType))
        return failure();
    } else if (allowVariadic && succeeded(parser.parseOptionalEllipsis())) {
      isVariadic = true;
      return success();
    } else if (!argNames.empty()) {
      // Reject this if the preceding argument had a name.
      return parser.emitError(loc, "expected SSA identifier");
    } else if (parser.parseType(argumentType)) {
      return failure();
    }

    // Add the argument type.
    argTypes.push_back(argumentType);

    // Parse any argument attributes.
    SmallVector<NamedAttribute, 2> attrs;
    if (parser.parseOptionalAttrDict(attrs))
      return failure();
    argAttrs.push_back(attrs);
    return success();
  };

  // Parse the function arguments.
  if (parser.parseOptionalRParen()) {
    do {
      unsigned numTypedArguments = argTypes.size();
      if (parseArgument())
        return failure();

      llvm::SMLoc loc = parser.getCurrentLocation();
      if (argTypes.size() == numTypedArguments &&
          succeeded(parser.parseOptionalComma()))
        return parser.emitError(
            loc, "variadic arguments must be in the end of the argument list");
    } while (succeeded(parser.parseOptionalComma()));
    parser.parseRParen();
  }

  return success();
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///                          | non-function-type
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= type attribute-dict?
///
static ParseResult parseFunctionResultList(
    OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &resultAttrs) {
  if (failed(parser.parseOptionalLParen())) {
    // We already know that there is no `(`, so parse a type.
    // Because there is no `(`, it cannot be a function type.
    Type ty;
    if (parser.parseType(ty))
      return failure();
    resultTypes.push_back(ty);
    resultAttrs.emplace_back();
    return success();
  }

  // Special case for an empty set of parens.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  // Parse individual function results.
  do {
    resultTypes.emplace_back();
    resultAttrs.emplace_back();
    if (parser.parseType(resultTypes.back()) ||
        parser.parseOptionalAttrDict(resultAttrs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// Parses a function signature using `parser`. The `allowVariadic` argument
/// indicates whether functions with variadic arguments are supported. The
/// trailing arguments are populated by this function with names, types and
/// attributes of the arguments and those of the results.
ParseResult mlir::impl::parseFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs, bool &isVariadic,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &resultAttrs) {
  if (parseArgumentList(parser, allowVariadic, argTypes, argNames, argAttrs,
                        isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return success();
}

void mlir::impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result,
    ArrayRef<SmallVector<NamedAttribute, 2>> argAttrs,
    ArrayRef<SmallVector<NamedAttribute, 2>> resultAttrs) {
  // Add the attributes to the function arguments.
  SmallString<8> attrNameBuf;
  for (unsigned i = 0, e = argAttrs.size(); i != e; ++i)
    if (!argAttrs[i].empty())
      result.addAttribute(getArgAttrName(i, attrNameBuf),
                          builder.getDictionaryAttr(argAttrs[i]));

  // Add the attributes to the function results.
  for (unsigned i = 0, e = resultAttrs.size(); i != e; ++i)
    if (!resultAttrs[i].empty())
      result.addAttribute(getResultAttrName(i, attrNameBuf),
                          builder.getDictionaryAttr(resultAttrs[i]));
}

/// Parser implementation for function-like operations.  Uses `funcTypeBuilder`
/// to construct the custom function type given lists of input and output types.
ParseResult
mlir::impl::parseFunctionLikeOp(OpAsmParser &parser, OperationState &result,
                                bool allowVariadic,
                                mlir::impl::FuncTypeBuilder funcTypeBuilder) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  auto signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (parseFunctionSignature(parser, allowVariadic, entryArgs, argTypes,
                             argAttrs, isVariadic, resultTypes, resultAttrs))
    return failure();

  std::string errorMessage;
  if (auto type = funcTypeBuilder(builder, argTypes, resultTypes,
                                  impl::VariadicFlag(isVariadic), errorMessage))
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  else
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  return parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? llvm::ArrayRef<Type>() : argTypes);
}

// Print a function result list.
static void printFunctionResultList(OpAsmPrinter &p, ArrayRef<Type> types,
                                    ArrayRef<ArrayRef<NamedAttribute>> attrs) {
  assert(!types.empty() && "Should not be called for empty result list.");
  auto &os = p.getStream();
  bool needsParens =
      types.size() > 1 || types[0].isa<FunctionType>() || !attrs[0].empty();
  if (needsParens)
    os << '(';
  interleaveComma(llvm::zip(types, attrs), os,
                  [&](const std::tuple<Type, ArrayRef<NamedAttribute>> &t) {
                    p.printType(std::get<0>(t));
                    p.printOptionalAttrDict(std::get<1>(t));
                  });
  if (needsParens)
    os << ')';
}

/// Print the signature of the function-like operation `op`.  Assumes `op` has
/// the FunctionLike trait and passed the verification.
void mlir::impl::printFunctionSignature(OpAsmPrinter &p, Operation *op,
                                        ArrayRef<Type> argTypes,
                                        bool isVariadic,
                                        ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    if (!isExternal) {
      p.printOperand(body.front().getArgument(i));
      p << ": ";
    }

    p.printType(argTypes[i]);
    p.printOptionalAttrDict(::mlir::impl::getArgAttrs(op, i));
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  if (!resultTypes.empty()) {
    p.getStream() << " -> ";
    SmallVector<ArrayRef<NamedAttribute>, 4> resultAttrs;
    for (int i = 0, e = resultTypes.size(); i < e; ++i)
      resultAttrs.push_back(::mlir::impl::getResultAttrs(op, i));
    printFunctionResultList(p, resultTypes, resultAttrs);
  }
}

/// Prints the list of function prefixed with the "attributes" keyword. The
/// attributes with names listed in "elided" as well as those used by the
/// function-like operation internally are not printed. Nothing is printed
/// if all attributes are elided. Assumes `op` has the `FunctionLike` trait and
/// passed the verification.
void mlir::impl::printFunctionAttributes(OpAsmPrinter &p, Operation *op,
                                         unsigned numInputs,
                                         unsigned numResults,
                                         ArrayRef<StringRef> elided) {
  // Print out function attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      ::mlir::SymbolTable::getSymbolAttrName(), getTypeAttrName()};
  ignoredAttrs.append(elided.begin(), elided.end());

  SmallString<8> attrNameBuf;

  // Ignore any argument attributes.
  std::vector<SmallString<8>> argAttrStorage;
  for (unsigned i = 0; i != numInputs; ++i)
    if (op->getAttr(getArgAttrName(i, attrNameBuf)))
      argAttrStorage.emplace_back(attrNameBuf);
  ignoredAttrs.append(argAttrStorage.begin(), argAttrStorage.end());

  // Ignore any result attributes.
  std::vector<SmallString<8>> resultAttrStorage;
  for (unsigned i = 0; i != numResults; ++i)
    if (op->getAttr(getResultAttrName(i, attrNameBuf)))
      resultAttrStorage.emplace_back(attrNameBuf);
  ignoredAttrs.append(resultAttrStorage.begin(), resultAttrStorage.end());

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);
}

/// Printer implementation for function-like operations.  Accepts lists of
/// argument and result types to use while printing.
void mlir::impl::printFunctionLikeOp(OpAsmPrinter &p, Operation *op,
                                     ArrayRef<Type> argTypes, bool isVariadic,
                                     ArrayRef<Type> resultTypes) {
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature(p, op, argTypes, isVariadic, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());

  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}
