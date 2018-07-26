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
    delete cast<MLFunction>(this);
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
    bb.getTerminator()->dropAllReferences();
  }
}

//===----------------------------------------------------------------------===//
// MLFunction implementation.
//===----------------------------------------------------------------------===//

MLFunction::MLFunction(StringRef name, FunctionType *type)
    : Function(name, type, Kind::MLFunc), StmtBlock(StmtBlockKind::MLFunc) {}

MLFunction::~MLFunction() {
  // TODO: When move SSA stuff is supported.
  // dropAllReferences();
}
