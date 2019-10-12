//===- FunctionSupport.cpp - Utility types for function-like ops ----------===//
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

#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

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
    if (parser.parseOptionalAttributeDict(attrs))
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

/// Parse a function signature, starting with a name and including the
/// parameter list.
static ParseResult parseFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs, bool &isVariadic,
    SmallVectorImpl<Type> &results) {
  if (parseArgumentList(parser, allowVariadic, argTypes, argNames, argAttrs,
                        isVariadic))
    return failure();
  // Parse the return types if present.
  return parser.parseOptionalArrowTypeList(results);
}

/// Parser implementation for function-like operations.  Uses `funcTypeBuilder`
/// to construct the custom function type given lists of input and output types.
ParseResult
mlir::impl::parseFunctionLikeOp(OpAsmParser &parser, OperationState &result,
                                bool allowVariadic,
                                mlir::impl::FuncTypeBuilder funcTypeBuilder) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> results;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                            result.attributes))
    return failure();
  // Convert the parsed function attr into a string attr.
  result.attributes.back().second = builder.getStringAttr(nameAttr.getValue());

  // Parse the function signature.
  auto signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (parseFunctionSignature(parser, allowVariadic, entryArgs, argTypes,
                             argAttrs, isVariadic, results))
    return failure();

  std::string errorMessage;
  if (auto type = funcTypeBuilder(builder, argTypes, results,
                                  impl::VariadicFlag(isVariadic), errorMessage))
    result.addAttribute(getTypeAttrName(), builder.getTypeAttr(type));
  else
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;

  // If function attributes are present, parse them.
  if (succeeded(parser.parseOptionalKeyword("attributes")))
    if (parser.parseOptionalAttributeDict(result.attributes))
      return failure();

  // Add the attributes to the function arguments.
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
    if (!argAttrs[i].empty())
      result.addAttribute(getArgAttrName(i, argAttrName),
                          builder.getDictionaryAttr(argAttrs[i]));

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseOptionalRegion(*body, entryArgs,
                                 entryArgs.empty() ? llvm::ArrayRef<Type>()
                                                   : argTypes))
    return failure();

  return success();
}

/// Print the signature of the function-like operation `op`.  Assumes `op` has
/// the FunctionLike trait and passed the verification.
static void printSignature(OpAsmPrinter &p, Operation *op,
                           ArrayRef<Type> argTypes, bool isVariadic,
                           ArrayRef<Type> results) {
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
  p.printOptionalArrowTypeList(results);
}

/// Printer implementation for function-like operations.  Accepts lists of
/// argument and result types to use while printing.
void mlir::impl::printFunctionLikeOp(OpAsmPrinter &p, Operation *op,
                                     ArrayRef<Type> argTypes, bool isVariadic,
                                     ArrayRef<Type> results) {
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  // Print the signature.
  printSignature(p, op, argTypes, isVariadic, results);

  // Print out function attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      ::mlir::SymbolTable::getSymbolAttrName(), getTypeAttrName()};

  // Ignore any argument attributes.
  std::vector<SmallString<8>> argAttrStorage;
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
    if (op->getAttr(getArgAttrName(i, argAttrName)))
      argAttrStorage.emplace_back(argAttrName);
  ignoredAttrs.append(argAttrStorage.begin(), argAttrStorage.end());

  auto attrs = op->getAttrs();
  if (attrs.size() > ignoredAttrs.size()) {
    p << "\n  attributes ";
    p.printOptionalAttrDict(attrs, ignoredAttrs);
  }

  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}
