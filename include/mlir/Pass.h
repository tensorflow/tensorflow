//===- Pass.h - Base classes for compiler passes ----------------*- C++ -*-===//
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

#ifndef MLIR_PASS_H
#define MLIR_PASS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <functional>

namespace mlir {
class Function;
class CFGFunction;
class MLFunction;
class Module;

// Values that can be used by to signal success/failure. This can be implicitly
// converted to/from boolean values, with false representing success and true
// failure.
struct LLVM_NODISCARD PassResult {
  enum ResultEnum { Success, Failure } value;
  PassResult(ResultEnum v) : value(v) {}
  operator bool() const { return value == Failure; }
};

class PassInfo;

class Pass {
public:
  explicit Pass(const void *passID) : passID(passID) {}
  virtual ~Pass() = default;
  virtual PassResult runOnModule(Module *m) = 0;

  /// Returns the unique identifier that corresponds to this pass.
  const void *getPassID() const { return passID; }

  static PassResult success() { return PassResult::Success; }
  static PassResult failure() { return PassResult::Failure; }

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookupPassInfo(const void *passID);

  /// Returns the pass info for this pass.
  const PassInfo *lookupPassInfo() const { return lookupPassInfo(passID); }

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();

  /// Unique identifier for pass.
  const void *const passID;
};

class ModulePass : public Pass {
public:
  explicit ModulePass(const void *passID) : Pass(passID) {}

  virtual PassResult runOnModule(Module *m) override = 0;

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();
};

/// FunctionPass's are run on every function in a module, and multiple functions
/// may be optimized concurrently by different instances of the function pass.
/// By subclassing this, your pass promises only to look at the function psased
/// in to it, it isn't allowed to inspect or modify other functions in the
/// module.
class FunctionPass : public Pass {
public:
  explicit FunctionPass(const void *passID) : Pass(passID) {}

  /// Implement this function to be run on every function in the module.  If you
  /// do not implement this, the default implementation will dispatch to
  /// runOnCFGFunction or runOnMLFunction.
  virtual PassResult runOnFunction(Function *fn);

  /// Implement this function if you want to see CFGFunction's specifically.
  virtual PassResult runOnCFGFunction(CFGFunction *fn) { return success(); }

  /// Implement this function if you want to see MLFunction's specifically.
  virtual PassResult runOnMLFunction(MLFunction *fn) { return success(); }

  // Iterates over all functions in a module, halting upon failure.
  virtual PassResult runOnModule(Module *m) override;
};

using PassAllocatorFunction = std::function<Pass *()>;

/// Structure to group information about a pass (argument to invoke via
/// mlir-opt, description, pass allocator and unique ID).
class PassInfo {
public:
  /// PassInfo constructor should not be invoked directly, instead use
  /// PassRegistration or registerPass.
  PassInfo(StringRef arg, StringRef description, const void *passID,
           PassAllocatorFunction allocator)
      : arg(arg), description(description), allocator(allocator),
        passID(passID){};

  /// Returns an allocated instance of this pass.
  Pass *createPass() const {
    assert(allocator &&
           "Cannot call createPass on PassInfo without default allocator");
    return allocator();
  }

  /// Returns the command line option that may be passed to 'mlir-opt' that will
  /// cause this pass to run or null if there is no such argument.
  StringRef getPassArgument() const { return arg; }

  /// Returns a description for the pass, this never returns null.
  StringRef getPassDescription() const { return description; }

private:
  // The argument with which to invoke the pass via mlir-opt.
  StringRef arg;

  // Description of the pass.
  StringRef description;

  // Allocator to construct an instance of this pass.
  PassAllocatorFunction allocator;

  // Unique identifier for pass.
  const void *passID;
};

/// Register a specific dialect creation function with the system, typically
/// used through the PassRegistration template.
void registerPass(StringRef arg, StringRef description, const void *passID,
                  const PassAllocatorFunction &function);

/// PassRegistration provides a global initializer that registers a Pass
/// allocation routine.
///
/// Usage:
///
///   // At namespace scope.
///   static PassRegistration<MyPass> Unused("unused", "Unused pass");
template <typename ConcretePass> struct PassRegistration {
  PassRegistration(StringRef arg, StringRef description) {
    registerPass(arg, description, &ConcretePass::passID,
                 [&]() { return new ConcretePass(); });
  }
};
} // end namespace mlir

#endif // MLIR_PASS_H
