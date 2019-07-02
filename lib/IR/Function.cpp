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
using namespace mlir::detail;

FunctionStorage::FunctionStorage(Location location, StringRef name,
                                 FunctionType type,
                                 ArrayRef<NamedAttribute> attrs)
    : name(Identifier::get(name, type.getContext())), location(location),
      type(type), attrs(attrs), argAttrs(type.getNumInputs()), body(this) {}

FunctionStorage::FunctionStorage(Location location, StringRef name,
                                 FunctionType type,
                                 ArrayRef<NamedAttribute> attrs,
                                 ArrayRef<NamedAttributeList> argAttrs)
    : name(Identifier::get(name, type.getContext())), location(location),
      type(type), attrs(attrs), argAttrs(argAttrs), body(this) {}

MLIRContext *Function::getContext() { return getType().getContext(); }

Module *llvm::ilist_traits<FunctionStorage>::getContainingModule() {
  size_t Offset(
      size_t(&((Module *)nullptr->*Module::getSublistAccess(nullptr))));
  iplist<FunctionStorage> *Anchor(static_cast<iplist<FunctionStorage> *>(this));
  return reinterpret_cast<Module *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a Function is added to a Module.  We
/// keep the module pointer and module symbol table up to date.
void llvm::ilist_traits<FunctionStorage>::addNodeToList(
    FunctionStorage *function) {
  assert(!function->module && "already in a module!");
  function->module = getContainingModule();
}

/// This is a trait method invoked when a Function is removed from a Module.
/// We keep the module pointer up to date.
void llvm::ilist_traits<FunctionStorage>::removeNodeFromList(
    FunctionStorage *function) {
  assert(function->module && "not already in a module!");
  function->module = nullptr;
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<FunctionStorage>::transferNodesFromList(
    ilist_traits<FunctionStorage> &otherList, function_iterator first,
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
  if (auto *module = getModule())
    getModule()->functions.erase(impl);
  else
    delete impl;
}

/// Emit an error about fatal conditions with this function, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns failure.  NOTE: This may terminate the containing application, only
/// use when the IR is in an inconsistent state.
InFlightDiagnostic Function::emitError() { return emitError({}); }
InFlightDiagnostic Function::emitError(const Twine &message) {
  return mlir::emitError(getLoc(), message);
}

/// Emit a warning about this function, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Function::emitWarning() { return emitWarning({}); }
InFlightDiagnostic Function::emitWarning(const Twine &message) {
  return mlir::emitWarning(getLoc(), message);
}

/// Emit a remark about this function, reporting up to any diagnostic
/// handlers that may be listening.
InFlightDiagnostic Function::emitRemark() { return emitRemark({}); }
InFlightDiagnostic Function::emitRemark(const Twine &message) {
  return mlir::emitRemark(getLoc(), message);
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void Function::cloneInto(Function dest, BlockAndValueMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<Identifier, Attribute> newAttrs;
  for (auto &attr : dest.getAttrs())
    newAttrs.insert(attr);
  for (auto &attr : getAttrs()) {
    auto insertPair = newAttrs.insert(attr);

    // TODO(riverriddle) Verify that the two functions have compatible
    // attributes.
    (void)insertPair;
    assert((insertPair.second || insertPair.first->second == attr.second) &&
           "the two functions have incompatible attributes");
  }
  dest.setAttrs(newAttrs.takeVector());

  // Clone the body.
  impl->body.cloneInto(&dest.impl->body, mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
Function Function::clone(BlockAndValueMapping &mapper) {
  FunctionType newType = impl->type;

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  bool isExternalFn = isExternal();
  if (!isExternalFn) {
    SmallVector<Type, 4> inputTypes;
    for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(inputTypes, newType.getResults(), getContext());
  }

  // Create the new function.
  Function newFunc = Function::create(getLoc(), getName(), newType);

  /// Set the argument attributes for arguments that aren't being replaced.
  for (unsigned i = 0, e = getNumArguments(), destI = 0; i != e; ++i)
    if (isExternalFn || !mapper.contains(getArgument(i)))
      newFunc.setArgAttrs(destI++, getArgAttrs(i));

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
Function Function::clone() {
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
  entry->addArguments(impl->type.getInputs());
}

void Function::walk(llvm::function_ref<void(Operation *)> callback) {
  getBody().walk(callback);
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
    llvm::SMLoc loc = parser->getCurrentLocation();

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

    // Parse any argument attributes.
    SmallVector<NamedAttribute, 2> attrs;
    if (parser->parseOptionalAttributeDict(attrs))
      return failure();
    argAttrs.push_back(attrs);
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

  // Add the attributes to the function arguments.
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (!argAttrs[i].empty())
      result->addAttribute(getArgAttrName(i, argAttrName),
                           builder.getDictionaryAttr(argAttrs[i]));

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

    // Print the type followed by any argument attributes.
    p->printType(fnType.getInput(i));
    p->printOptionalAttrDict(op.getArgAttrs(i));
  }
  *p << ')';
  p->printOptionalArrowTypeList(fnType.getResults());
}

void FuncOp::print(OpAsmPrinter *p) {
  *p << "func @" << getName();

  // Print the signature.
  printFunctionSignature(p, *this);

  // Print out function attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {"name", "type"};

  // Ignore any argument attributes.
  std::vector<SmallString<8>> argAttrStorage;
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
    if (getAttr(getArgAttrName(i, argAttrName)))
      argAttrStorage.emplace_back(argAttrName);
  ignoredAttrs.append(argAttrStorage.begin(), argAttrStorage.end());

  auto attrs = getAttrs();
  if (attrs.size() > ignoredAttrs.size()) {
    *p << "\n  attributes ";
    p->printOptionalAttrDict(attrs, ignoredAttrs);
  }

  // Print the body if this is not an external function.
  if (!isExternal()) {
    p->printRegion(getBody(), /*printEntryBlockArgs=*/false,
                   /*printBlockTerminators=*/true);
    *p << '\n';
  }
  *p << '\n';
}

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.
  Block &entryBlock = front();
  auto fnInputTypes = getType().getInputs();
  if (fnInputTypes.size() != entryBlock.getNumArguments())
    return emitOpError("entry block must have ")
           << fnInputTypes.size() << " arguments to match function signature";

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i)->getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i)->getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

//===----------------------------------------------------------------------===//
// Function Argument Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the argument at 'index'.
void FuncOp::setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
  assert(index < getNumArguments() && "invalid argument number");
  SmallString<8> nameOut;
  getArgAttrName(index, nameOut);

  if (attributes.empty())
    return (void)removeAttr(nameOut);
  setAttr(nameOut, DictionaryAttr::get(attributes, getContext()));
}

void FuncOp::setArgAttrs(unsigned index, NamedAttributeList attributes) {
  assert(index < getNumArguments() && "invalid argument number");
  SmallString<8> nameOut;
  if (auto newAttr = attributes.getDictionary())
    return setAttr(getArgAttrName(index, nameOut), newAttr);
  removeAttr(getArgAttrName(index, nameOut));
}

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
void FuncOp::setArgAttr(unsigned index, Identifier name, Attribute value) {
  auto curAttr = getArgAttrDict(index);
  NamedAttributeList attrList(curAttr);
  attrList.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (curAttr != attrList.getDictionary())
    setArgAttrs(index, attrList);
}

/// Remove the attribute 'name' from the argument at 'index'.
NamedAttributeList::RemoveResult FuncOp::removeArgAttr(unsigned index,
                                                       Identifier name) {
  // Build an attribute list and remove the attribute at 'name'.
  NamedAttributeList attrList(getArgAttrDict(index));
  auto result = attrList.remove(name);

  // If the attribute was removed, then update the argument dictionary.
  if (result == NamedAttributeList::RemoveResult::Removed)
    setArgAttrs(index, attrList);
  return result;
}

/// Returns the attribute entry name for the set of argument attributes at index
/// 'arg'.
StringRef FuncOp::getArgAttrName(unsigned arg, SmallVectorImpl<char> &out) {
  out.clear();
  return ("arg" + Twine(arg)).toStringRef(out);
}
