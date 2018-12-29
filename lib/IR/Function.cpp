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
#include "AttributeListStorage.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
using namespace mlir;

Function::Function(Kind kind, Location location, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs)
    : nameAndKind(Identifier::get(name, type.getContext()), kind),
      location(location), type(type), blocks(this) {
  this->attrs = AttributeListStorage::get(attrs, getContext());

  // Creating of a Function automatically populates the entry block and
  // arguments.
  // TODO(clattner): Unify this behavior.
  if (kind == Kind::MLFunc) {
    // The body of an ML Function always has one block.
    auto *entry = new Block();
    blocks.push_back(entry);

    // Initialize the arguments.
    entry->addArguments(type.getInputs());
  }
}

Function::~Function() {
  // Instructions may have cyclic references, which need to be dropped before we
  // can start deleting them.
  for (auto &bb : *this)
    for (auto &inst : bb)
      inst.dropAllReferences();

  // Clean up function attributes referring to this function.
  FunctionAttr::dropFunctionReference(this);
}

ArrayRef<NamedAttribute> Function::getAttrs() const {
  if (attrs)
    return attrs->getElements();
  else
    return {};
}

MLIRContext *Function::getContext() const { return getType().getContext(); }

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
      function->nameAndKind.setPointer(
          Identifier::get(nameBuffer, module->getContext()));
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

/// Emit a note about this instruction, reporting up to any diagnostic
/// handlers that may be listening.
void Function::emitNote(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Note);
}

/// Emit a warning about this operation, reporting up to any diagnostic
/// handlers that may be listening.
void Function::emitWarning(const Twine &message) const {
  getContext()->emitDiagnostic(getLoc(), message,
                               MLIRContext::DiagnosticKind::Warning);
}

/// Emit an error about fatal conditions with this operation, reporting up to
/// any diagnostic handlers that may be listening.  This function always
/// returns true.  NOTE: This may terminate the containing application, only use
/// when the IR is in an inconsistent state.
bool Function::emitError(const Twine &message) const {
  return getContext()->emitError(getLoc(), message);
}

//===----------------------------------------------------------------------===//
// Function implementation.
//===----------------------------------------------------------------------===//

void Function::walkInsts(std::function<void(Instruction *)> callback) {
  struct Walker : public InstWalker<Walker> {
    std::function<void(Instruction *)> const &callback;
    Walker(std::function<void(Instruction *)> const &callback)
        : callback(callback) {}

    void visitInstruction(Instruction *inst) { callback(inst); }
  };

  Walker v(callback);
  v.walk(this);
}

void Function::walkOps(std::function<void(OperationInst *)> callback) {
  struct Walker : public InstWalker<Walker> {
    std::function<void(OperationInst *)> const &callback;
    Walker(std::function<void(OperationInst *)> const &callback)
        : callback(callback) {}

    void visitOperationInst(OperationInst *opInst) { callback(opInst); }
  };

  Walker v(callback);
  v.walk(this);
}

void Function::walkInstsPostOrder(std::function<void(Instruction *)> callback) {
  struct Walker : public InstWalker<Walker> {
    std::function<void(Instruction *)> const &callback;
    Walker(std::function<void(Instruction *)> const &callback)
        : callback(callback) {}

    void visitOperationInst(Instruction *inst) { callback(inst); }
  };

  Walker v(callback);
  v.walkPostOrder(this);
}

void Function::walkOpsPostOrder(std::function<void(OperationInst *)> callback) {
  struct Walker : public InstWalker<Walker> {
    std::function<void(OperationInst *)> const &callback;
    Walker(std::function<void(OperationInst *)> const &callback)
        : callback(callback) {}

    void visitOperationInst(OperationInst *opInst) { callback(opInst); }
  };

  Walker v(callback);
  v.walkPostOrder(this);
}
