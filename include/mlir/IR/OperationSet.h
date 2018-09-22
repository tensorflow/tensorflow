//===- OperationSet.h -------------------------------------------*- C++ -*-===//
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
// This file defines the AbstractOperation and OperationSet classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATIONSET_H
#define MLIR_IR_OPERATIONSET_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Attribute;
class Operation;
class OperationState;
class OpAsmParser;
class OpAsmParserResult;
class OpAsmPrinter;
class MLIRContextImpl;
class MLIRContext;

/// This is a "type erased" representation of a registered operation.  This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks.  Most user code should use
/// the concrete operation types.
class AbstractOperation {
public:
  template <typename T>
  static AbstractOperation get() {
    return AbstractOperation(T::getOperationName(), T::isClassFor,
                             T::parseAssembly, T::printAssembly,
                             T::verifyInvariants, T::constantFoldHook);
  }

  /// This is the name of the operation.
  const StringRef name;

  /// Return true if this "op class" can match against the specified operation.
  bool (&isClassFor)(const Operation *op);

  /// Use the specified object to parse this ops custom assembly format.
  bool (&parseAssembly)(OpAsmParser *parser, OperationState *result);

  /// This hook implements the AsmPrinter for this operation.
  void (&printAssembly)(const Operation *op, OpAsmPrinter *p);

  /// This hook implements the verifier for this operation.  It should emits an
  /// error message and returns true if a problem is detected, or returns false
  /// if everything is ok.
  bool (&verifyInvariants)(const Operation *op);

  /// This hook implements a constant folder for this operation.  It returns
  /// true if folding failed, or returns false and fills in `results` on
  /// success.
  bool (&constantFoldHook)(const Operation *op, ArrayRef<Attribute *> operands,
                           SmallVectorImpl<Attribute *> &results);

private:
  AbstractOperation(
      StringRef name, bool (&isClassFor)(const Operation *op),
      bool (&parseAssembly)(OpAsmParser *parser, OperationState *result),
      void (&printAssembly)(const Operation *op, OpAsmPrinter *p),
      bool (&verifyInvariants)(const Operation *op),
      bool (&constantFoldHook)(const Operation *op,
                               ArrayRef<Attribute *> operands,
                               SmallVectorImpl<Attribute *> &results))
      : name(name), isClassFor(isClassFor), parseAssembly(parseAssembly),
        printAssembly(printAssembly), verifyInvariants(verifyInvariants),
        constantFoldHook(constantFoldHook) {}
};

/// An instance of OperationSet is owned and maintained by MLIRContext.  It
/// contains any specialized operations that the compiler executable may be
/// aware of.  This can include things like high level operations for
/// TensorFlow, target specific instructions for code generation, or other for
/// any other purpose.
///
/// Operations do not need to be registered with an OperationSet to work, but
/// doing so grants special parsing, printing, and validation capabilities.
///
class OperationSet {
public:
  ~OperationSet();

  /// Return the operation set for this context.  Clients can register their own
  /// operations with this, and internal systems use those registered hooked to
  /// print, parse, and verify the operations.
  static OperationSet &get(MLIRContext *context);

  /// Look up the specified operation in the operation set and return a pointer
  /// to it if present.  Otherwise, return a null pointer.
  const AbstractOperation *lookup(StringRef opName) const;

  /// This method is used by derived classes to add their operations to the set.
  ///
  /// The prefix should be common across all ops in this set, e.g. "" for the
  /// standard operation set, and "tf." for the TensorFlow ops like "tf.add".
  template <typename... Args>
  void addOperations(StringRef prefix) {
    VariadicOperationAdder<Args...>::addToSet(prefix, *this);
  }

private:
  friend class MLIRContextImpl;
  explicit OperationSet();

  // It would be nice to define this as variadic functions instead of a nested
  // variadic type, but we can't do that: function template partial
  // specialization is not allowed, and we can't define an overload set because
  // we don't have any arguments of the types we are pushing around.
  template <typename First, typename... Rest>
  class VariadicOperationAdder {
  public:
    static void addToSet(StringRef prefix, OperationSet &set) {
      set.addOperation(prefix, AbstractOperation::get<First>());
      VariadicOperationAdder<Rest...>::addToSet(prefix, set);
    }
  };

  template <typename First>
  class VariadicOperationAdder<First> {
  public:
    static void addToSet(StringRef prefix, OperationSet &set) {
      set.addOperation(prefix, AbstractOperation::get<First>());
    }
  };

  void addOperation(StringRef prefix, AbstractOperation opInfo);

  OperationSet(const OperationSet &) = delete;
  void operator=(OperationSet &) = delete;

  // This is a pointer to the implementation using the pImpl idiom.
  void *pImpl;
};

using InitializeOpsFunction = std::function<void(MLIRContext *context)>;

// Use OpInitializeRegistration as a global initialiser that registers Op
// initializers.
//
// Usage:
//
//   // At namespace scope.
//   static OpInitializeRegistration Unused([] { ... });
struct OpInitializeRegistration {
  OpInitializeRegistration(const InitializeOpsFunction &function);
};

/// Initializes all registers ops in the given MLIRContext.
void initializeAllRegisteredOps(MLIRContext *context);

} // namespace mlir

#endif
