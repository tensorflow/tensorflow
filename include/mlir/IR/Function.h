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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"

namespace mlir {
class AttributeListStorage;
class FunctionType;
class MLIRContext;
class Module;

/// NamedAttribute is used for function attribute lists, it holds an
/// identifier for the name and a value for the attribute.  The attribute
/// pointer should always be non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

/// This is the base class for all of the MLIR function types.
class Function : public llvm::ilist_node_with_parent<Function, Module> {
public:
  enum class Kind { ExtFunc, CFGFunc, MLFunc };

  Kind getKind() const { return (Kind)nameAndKind.getInt(); }

  /// The source location the operation was defined or derived from.
  Location getLoc() const { return location; }

  /// Return the name of this function, without the @.
  Identifier getName() const { return nameAndKind.getPointer(); }

  /// Return the type of this function.
  FunctionType getType() const { return type; }

  /// Returns all of the attributes on this function.
  ArrayRef<NamedAttribute> getAttrs() const;

  MLIRContext *getContext() const;
  Module *getModule() { return module; }
  const Module *getModule() const { return module; }

  /// Unlink this instruction from its module and delete it.
  void eraseFromModule();

  /// Delete this object.
  void destroy();

  /// Perform (potentially expensive) checks of invariants, used to detect
  /// compiler bugs.  On error, this reports the error through the MLIRContext
  /// and returns true.
  bool verify() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

protected:
  Function(Kind kind, Location location, StringRef name, FunctionType type,
           ArrayRef<NamedAttribute> attrs = {});
  ~Function();

private:
  /// The name of the function and the kind of function this is.
  llvm::PointerIntPair<Identifier, 2, Kind> nameAndKind;

  /// The module this function is embedded into.
  Module *module = nullptr;

  /// The source location the function was defined or derived from.
  Location location;

  /// The type of the function.
  FunctionType type;

  /// This holds general named attributes for the function.
  AttributeListStorage *attrs;

  void operator=(const Function &) = delete;
  friend struct llvm::ilist_traits<Function>;
};

/// An extfunc declaration is a declaration of a function signature that is
/// defined in some other module.
class ExtFunction : public Function {
public:
  ExtFunction(Location location, StringRef name, FunctionType type,
              ArrayRef<NamedAttribute> attrs = {});

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
