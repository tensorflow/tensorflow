//===- Function.cpp - MLIR Function Classes -------------------------------===//
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

#include "mlir/IR/Function.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

Function::Function(Location location, StringRef name, FunctionType type,
                   ArrayRef<NamedAttribute> attrs)
    : name(Identifier::get(name, type.getContext())), location(location),
      type(type), attrs(attrs), argAttrs(type.getNumInputs()), body(this) {}

Function::Function(Location location, StringRef name, FunctionType type,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<NamedAttributeList> argAttrs)
    : name(Identifier::get(name, type.getContext())), location(location),
      type(type), attrs(attrs), argAttrs(argAttrs), body(this) {}

MLIRContext *Function::getContext() { return getType().getContext(); }

/// Swap the name of the given function with this one.
void Function::takeName(Function &rhs) {
  auto *module = getModule();
  assert(module && module == rhs.getModule() && "expected same parent module");
  std::swap(module->symbolTable[name], module->symbolTable[rhs.getName()]);
  std::swap(name, rhs.name);
}

Module *llvm::ilist_traits<Function>::getContainingModule() {
  size_t Offset(
      size_t(&((Module *)nullptr->*Module::getSublistAccess(nullptr))));
  iplist<Function> *Anchor(static_cast<iplist<Function> *>(this));
  return reinterpret_cast<Module *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a Function is added to a Module.  We
/// keep the module pointer and module symbol table up to date.
void llvm::ilist_traits<Function>::addNodeToList(Function *function) {
  assert(!function->getModule() && "already in a module!");
  auto *module = getContainingModule();
  function->module = module;

  // Add this function to the symbol table of the module, uniquing the name if
  // a conflict is detected.
  if (!module->symbolTable.insert({function->getName(), function}).second) {
    // If a conflict was detected, then the function will not have been added to
    // the symbol table.  Try suffixes until we get to a unique name that works.
    SmallString<128> nameBuffer(function->getName().begin(),
                                function->getName().end());
    unsigned originalLength = nameBuffer.size();

    // Iteratively try suffixes until we find one that isn't used.  We use a
    // module level uniquing counter to avoid N^2 behavior.
    do {
      nameBuffer.resize(originalLength);
      nameBuffer += '_';
      nameBuffer += std::to_string(module->uniquingCounter++);
      function->name = Identifier::get(nameBuffer, module->getContext());
    } while (
        !module->symbolTable.insert({function->getName(), function}).second);
  }
}

/// This is a trait method invoked when a Function is removed from a Module.
/// We keep the module pointer up to date.
void llvm::ilist_traits<Function>::removeNodeFromList(Function *function) {
  assert(function->module && "not already in a module!");

  // Remove the symbol table entry.
  function->module->symbolTable.erase(function->getName());
  function->module = nullptr;
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<Function>::transferNodesFromList(
    ilist_traits<Function> &otherList, function_iterator first,
    function_iterator last) {
  // If we are transferring functions within the same module, the Module
  // pointer doesn't need to be updated.
  Module *curParent = getContainingModule();
  if (curParent == otherList.getContainingModule())
    return;

  // Update the 'module' member and symbol table records for each function.
  for (; first != last; ++first) {
    removeNodeFromList(&*first);
    addNodeToList(&*first);
  }
}

/// Unlink this function from its Module and delete it.
void Function::erase() {
  assert(getModule() && "Function has no parent");
  getModule()->getFunctions().erase(this);
}

/// Emit an error about fatal conditions with this function, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns failure.  NOTE: This may terminate the containing application, only
/// use when the IR is in an inconsistent state.
InFlightDiagnostic Function::emitError() { return emitError({}); }
InFlightDiagnostic Function::emitError(const Twine &message) {
  return getContext()->emitError(getLoc(), message);
}

/// Emit a warning about this function, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Function::emitWarning() { return emitWarning({}); }
InFlightDiagnostic Function::emitWarning(const Twine &message) {
  return getContext()->emitWarning(getLoc(), message);
}

/// Emit a remark about this function, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Function::emitRemark() { return emitRemark({}); }
InFlightDiagnostic Function::emitRemark(const Twine &message) {
  return getContext()->emitRemark(getLoc(), message);
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void Function::cloneInto(Function *dest, BlockAndValueMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<Identifier, Attribute> newAttrs;
  for (auto &attr : dest->getAttrs())
    newAttrs.insert(attr);
  for (auto &attr : getAttrs()) {
    auto insertPair = newAttrs.insert(attr);

    // TODO(riverriddle) Verify that the two functions have compatible
    // attributes.
    (void)insertPair;
    assert((insertPair.second || insertPair.first->second == attr.second) &&
           "the two functions have incompatible attributes");
  }
  dest->setAttrs(newAttrs.takeVector());

  // Clone the body.
  body.cloneInto(&dest->body, mapper, getContext());
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
Function *Function::clone(BlockAndValueMapping &mapper) {
  FunctionType newType = type;

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  bool isExternalFn = isExternal();
  if (!isExternalFn) {
    SmallVector<Type, 4> inputTypes;
    for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(type.getInput(i));
    newType = FunctionType::get(inputTypes, type.getResults(), getContext());
  }

  // Create the new function.
  Function *newFunc = new Function(getLoc(), getName(), newType);

  /// Set the argument attributes for arguments that aren't being replaced.
  for (unsigned i = 0, e = getNumArguments(), destI = 0; i != e; ++i)
    if (isExternalFn || !mapper.contains(getArgument(i)))
      newFunc->setArgAttrs(destI++, getArgAttrs(i));

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
Function *Function::clone() {
  BlockAndValueMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
// Function implementation.
//===----------------------------------------------------------------------===//

/// Add an entry block to an empty function, and set up the block arguments
/// to match the signature of the function.
void Function::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  entry->addArguments(type.getInputs());
}

void Function::walk(const std::function<void(Operation *)> &callback) {
  // Walk each of the blocks within the function.
  for (auto &block : getBlocks())
    block.walk(callback);
}

//===----------------------------------------------------------------------===//
// Function Operation.
//===----------------------------------------------------------------------===//

void FuncOp::build(Builder *builder, OperationState *result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  result->addAttribute("name", builder->getStringAttr(name));
  result->addAttribute("type", builder->getTypeAttr(type));
  result->attributes.append(attrs.begin(), attrs.end());
  result->addRegion();
}

/// Parsing/Printing methods.
static ParseResult
parseArgumentList(OpAsmParser *parser, SmallVectorImpl<Type> &argTypes,
                  SmallVectorImpl<OpAsmParser::OperandType> &argNames,
                  SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs) {
  if (parser->parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's and
  // sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc;
    parser->getCurrentLocation(&loc);

    // Parse argument name if present.
    OpAsmParser::OperandType argument;
    Type argumentType;
    if (succeeded(parser->parseOptionalRegionArgument(argument)) &&
        !argument.name.empty()) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return parser->emitError(loc,
                                 "expected type instead of SSA identifier");
      argNames.push_back(argument);

      if (parser->parseColonType(argumentType))
        return failure();
    } else if (!argNames.empty()) {
      // Reject this if the preceding argument had a name.
      return parser->emitError(loc, "expected SSA identifier");
    } else if (parser->parseType(argumentType)) {
      return failure();
    }

    // Add the argument type.
    argTypes.push_back(argumentType);

    // TODO(riverriddle) Parse argument attributes.
    // Parse the attribute dict.
    // SmallVector<NamedAttribute, 2> attrs;
    // if (parser->parseOptionalAttributeDict(attrs))
    //  return failure();
    // argAttrs.push_back(attrs);
    return success();
  };

  // Parse the function arguments.
  if (parser->parseOptionalRParen()) {
    do {
      if (parseArgument())
        return failure();
    } while (succeeded(parser->parseOptionalComma()));
    parser->parseRParen();
  }

  return success();
}

/// Parse a function signature, starting with a name and including the
/// parameter list.
static ParseResult parseFunctionSignature(
    OpAsmParser *parser, FunctionType &type,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<SmallVector<NamedAttribute, 2>> &argAttrs) {
  SmallVector<Type, 4> argTypes;
  if (parseArgumentList(parser, argTypes, argNames, argAttrs))
    return failure();

  // Parse the return types if present.
  SmallVector<Type, 4> results;
  if (parser->parseOptionalArrowTypeList(results))
    return failure();
  type = parser->getBuilder().getFunctionType(argTypes, results);
  return success();
}

ParseResult FuncOp::parse(OpAsmParser *parser, OperationState *result) {
  FunctionType type;
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;
  auto &builder = parser->getBuilder();

  // Parse the name as a function attribute.
  FunctionAttr nameAttr;
  if (parser->parseAttribute(nameAttr, "name", result->attributes))
    return failure();
  // Convert the parsed function attr into a string attr.
  result->attributes.back().second = builder.getStringAttr(nameAttr.getValue());

  // Parse the function signature.
  if (parseFunctionSignature(parser, type, entryArgs, argAttrs))
    return failure();
  result->addAttribute("type", builder.getTypeAttr(type));

  // If function attributes are present, parse them.
  if (succeeded(parser->parseOptionalKeyword("attributes")))
    if (parser->parseOptionalAttributeDict(result->attributes))
      return failure();

  // TODO(riverriddle) Parse argument attributes.
  // Add the attributes to the function arguments.
  // for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
  //   if (!argAttrs[i].empty())
  //     result->addAttribute(("arg" + Twine(i)).str(),
  //                          builder.getDictionaryAttr(argAttrs[i]));

  // Parse the optional function body.
  auto *body = result->addRegion();
  if (parser->parseOptionalRegion(
          *body, entryArgs, entryArgs.empty() ? llvm::None : type.getInputs()))
    return failure();

  return success();
}

static void printFunctionSignature(OpAsmPrinter *p, FuncOp op) {
  *p << '(';

  auto fnType = op.getType();
  bool isExternal = op.isExternal();
  for (unsigned i = 0, e = op.getNumArguments(); i != e; ++i) {
    if (i > 0)
      *p << ", ";

    // If this is an external function, don't print argument labels.
    if (!isExternal) {
      p->printOperand(op.getArgument(i));
      *p << ": ";
    }

    p->printType(fnType.getInput(i));

    // TODO(riverriddle) Print argument attributes.
    // Print the attributes for this argument.
    // p->printOptionalAttrDict(op.getArgAttrs(i));
  }
  *p << ')';

  switch (fnType.getResults().size()) {
  case 0:
    break;
  case 1: {
    *p << " -> ";
    auto resultType = fnType.getResults()[0];
    bool resultIsFunc = resultType.isa<FunctionType>();
    if (resultIsFunc)
      *p << '(';
    p->printType(resultType);
    if (resultIsFunc)
      *p << ')';
    break;
  }
  default:
    *p << " -> (";
    interleaveComma(fnType.getResults(), *p);
    *p << ')';
    break;
  }
}

void FuncOp::print(OpAsmPrinter *p) {
  *p << "func @" << getName();

  // Print the signature.
  printFunctionSignature(p, *this);

  // Print out function attributes, if present.
  auto attrs = getAttrs();

  // We must have more attributes than <name, type>.
  constexpr unsigned kNumHiddenAttrs = 2;
  if (attrs.size() > kNumHiddenAttrs) {
    *p << "\n  attributes ";
    p->printOptionalAttrDict(attrs, {"name", "type"});
  }

  // Print the body if this is not an external function.
  if (!isExternal()) {
    p->printRegion(getBody(), /*printEntryBlockArgs=*/false,
                   /*printBlockTerminators=*/true);
    *p << '\n';
  }
  *p << '\n';
}
