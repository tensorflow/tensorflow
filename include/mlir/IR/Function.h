//===- Function.h - MLIR Function Class -------------------------*- C++ -*-===//
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
// Functions are the basic unit of composition in MLIR.  There are three
// different kinds of functions: external functions, CFG functions, and ML
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/IR/Identifier.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"

namespace mlir {
class FunctionType;
class MLIRContext;
class Module;

/// This is the base class for all of the MLIR function types.
class Function : public llvm::ilist_node_with_parent<Function, Module> {
public:
  enum class Kind { ExtFunc, CFGFunc, MLFunc };

  Kind getKind() const { return kind; }

  /// Return the name of this function, without the @.
  Identifier getName() const { return name; }

  /// Return the type of this function.
  FunctionType *getType() const { return type; }

  MLIRContext *getContext() const;
  Module *getModule() { return module; }
  const Module *getModule() const { return module; }

  /// Unlink this instruction from its module and delete it.
  void eraseFromModule();

  /// Delete this object.
  void destroy();

  /// Perform (potentially expensive) checks of invariants, used to detect
  /// compiler bugs.  On error, this fills in the string and return true,
  /// or aborts if the string was not provided.
  bool verify(std::string *errorResult = nullptr) const;

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  Function(StringRef name, FunctionType *type, Kind kind);
  ~Function() {}

private:
  Kind kind;
  Module *module = nullptr;
  Identifier name;
  FunctionType *const type;

  void operator=(const Function &) = delete;
  friend struct llvm::ilist_traits<Function>;
};

/// An extfunc declaration is a declaration of a function signature that is
/// defined in some other module.
class ExtFunction : public Function {
public:
  ExtFunction(StringRef name, FunctionType *type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Function *func) {
    return func->getKind() == Kind::ExtFunc;
  }
};

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Function
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::Function>
    : public ilist_alloc_traits<::mlir::Function> {
  using Function = ::mlir::Function;
  using function_iterator = simple_ilist<Function>::iterator;

  static void deleteNode(Function *inst) { inst->destroy(); }

  void addNodeToList(Function *function);
  void removeNodeFromList(Function *function);
  void transferNodesFromList(ilist_traits<Function> &otherList,
                             function_iterator first, function_iterator last);

private:
  mlir::Module *getContainingModule();
};
} // end namespace llvm

#endif  // MLIR_IR_FUNCTION_H
