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

#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
using namespace mlir;

Function::Function(StringRef name, FunctionType *type, Kind kind)
  : kind(kind), name(name.str()), type(type) {
}

MLIRContext *Function::getContext() const { return getType()->getContext(); }

/// Delete this object.
void Function::destroy() {
  switch (getKind()) {
  case Kind::ExtFunc:
    delete cast<ExtFunction>(this);
    break;
  case Kind::MLFunc:
    cast<MLFunction>(this)->destroy();
    break;
  case Kind::CFGFunc:
    delete cast<CFGFunction>(this);
    break;
  }
}

Module *llvm::ilist_traits<Function>::getContainingModule() {
  size_t Offset(
      size_t(&((Module *)nullptr->*Module::getSublistAccess(nullptr))));
  iplist<Function> *Anchor(static_cast<iplist<Function> *>(this));
  return reinterpret_cast<Module *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a Function is added to a Module.  We
/// keep the module pointer up to date.
void llvm::ilist_traits<Function>::addNodeToList(Function *function) {
  assert(!function->getModule() && "already in a module!");
  function->module = getContainingModule();
}

/// This is a trait method invoked when a Function is removed from a Module.
/// We keep the module pointer up to date.
void llvm::ilist_traits<Function>::removeNodeFromList(Function *function) {
  assert(function->module && "not already in a module!");
  function->module = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<Function>::transferNodesFromList(
    ilist_traits<Function> &otherList, function_iterator first,
    function_iterator last) {
  // If we are transferring functions within the same module, the Module
  // pointer doesn't need to be updated.
  Module *curParent = getContainingModule();
  if (curParent == otherList.getContainingModule())
    return;

  // Update the 'module' member of each function.
  for (; first != last; ++first)
    first->module = curParent;
}

/// Unlink this function from its Module and delete it.
void Function::eraseFromModule() {
  assert(getModule() && "Function has no parent");
  getModule()->getFunctions().erase(this);
}

//===----------------------------------------------------------------------===//
// ExtFunction implementation.
//===----------------------------------------------------------------------===//

ExtFunction::ExtFunction(StringRef name, FunctionType *type)
  : Function(name, type, Kind::ExtFunc) {
}

//===----------------------------------------------------------------------===//
// CFGFunction implementation.
//===----------------------------------------------------------------------===//

CFGFunction::CFGFunction(StringRef name, FunctionType *type)
  : Function(name, type, Kind::CFGFunc) {
}

CFGFunction::~CFGFunction() {
  // Instructions may have cyclic references, which need to be dropped before we
  // can start deleting them.
  for (auto &bb : *this) {
    for (auto &inst : bb)
      inst.dropAllReferences();
    if (bb.getTerminator())
      bb.getTerminator()->dropAllReferences();
  }
}

//===----------------------------------------------------------------------===//
// MLFunction implementation.
//===----------------------------------------------------------------------===//

/// Create a new MLFunction with the specific fields.
MLFunction *MLFunction::create(StringRef name, FunctionType *type) {
  const auto &argTypes = type->getInputs();
  auto byteSize = totalSizeToAlloc<MLFuncArgument>(argTypes.size());
  void *rawMem = malloc(byteSize);

  // Initialize the MLFunction part of the function object.
  auto function = ::new (rawMem) MLFunction(name, type);

  // Initialize the arguments.
  auto arguments = function->getArgumentsInternal();
  for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
    new (&arguments[i]) MLFuncArgument(argTypes[i], function);
  return function;
}

MLFunction::MLFunction(StringRef name, FunctionType *type)
    : Function(name, type, Kind::MLFunc), StmtBlock(StmtBlockKind::MLFunc) {}

MLFunction::~MLFunction() {
  // Explicitly erase statements instead of relying of 'StmtBlock' destructor
  // since child statements need to be destroyed before function arguments
  // are destroyed.
  clear();

  // Explicitly run the destructors for the function arguments.
  for (auto &arg : getArgumentsInternal())
    arg.~MLFuncArgument();
}

void MLFunction::destroy() {
  this->~MLFunction();
  free(this);
}

const OperationStmt *MLFunction::getReturnStmt() const {
  return cast<OperationStmt>(&back());
}

OperationStmt *MLFunction::getReturnStmt() {
  return cast<OperationStmt>(&back());
}
