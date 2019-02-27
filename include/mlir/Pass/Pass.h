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

#ifndef MLIR_PASS_PASS_H
#define MLIR_PASS_PASS_H

#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {
class Function;
class Module;

// Values that can be used by to signal success/failure. This can be implicitly
// converted to/from boolean values, with false representing success and true
// failure.
struct LLVM_NODISCARD PassResult {
  enum ResultEnum { Success, Failure } value;
  PassResult(ResultEnum v) : value(v) {}
  operator bool() const { return value == Failure; }
};

/// The abstract base pass class. This class contains information describing the
/// derived pass object, e.g its kind and abstract PassInfo.
class Pass {
public:
  enum class Kind { FunctionPass, ModulePass };

  virtual ~Pass() = default;

  /// TODO: This is deprecated and should be removed.
  virtual PassResult runOnModule(Module *m) { return failure(); }

  /// Returns the unique identifier that corresponds to this pass.
  const PassID *getPassID() const { return passIDAndKind.getPointer(); }

  static PassResult success() { return PassResult::Success; }
  static PassResult failure() { return PassResult::Failure; }

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookupPassInfo(const PassID *passID);
  template <typename PassT> static const PassInfo *lookupPassInfo() {
    return lookupPassInfo(PassID::getID<PassT>());
  }

  /// Returns the pass info for this pass.
  const PassInfo *lookupPassInfo() const { return lookupPassInfo(getPassID()); }

  /// Return the kind of this pass.
  Kind getKind() const { return passIDAndKind.getInt(); }

protected:
  Pass(const PassID *passID, Kind kind) : passIDAndKind(passID, kind) {}

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();

  /// Represents a unique identifier for the pass and its kind.
  llvm::PointerIntPair<const PassID *, 1, Kind> passIDAndKind;
};

/// Deprecated Function and Module Pass definitions.
/// TODO(riverriddle) Remove these.
class ModulePass : public Pass {
public:
  explicit ModulePass(const PassID *passID) : Pass(passID, Kind::ModulePass) {}

  virtual PassResult runOnModule(Module *m) override = 0;

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();
};
class FunctionPass : public Pass {
public:
  explicit FunctionPass(const PassID *passID)
      : Pass(passID, Kind::FunctionPass) {}

  /// Implement this function to be run on every function in the module.
  virtual PassResult runOnFunction(Function *fn) = 0;

  // Iterates over all functions in a module, halting upon failure.
  virtual PassResult runOnModule(Module *m) override;
};

namespace detail {
class FunctionPassExecutor;
class ModulePassExecutor;

/// The state for a single execution of a pass. This provides a unified
/// interface for accessing and initializing necessary state for pass execution.
template <typename IRUnitT> struct PassExecutionState {
  explicit PassExecutionState(IRUnitT *ir) : ir(ir) {}

  /// The current IR unit being transformed.
  IRUnitT *ir;
};
} // namespace detail

/// Pass to transform a specific function within a module. Derived passes should
/// not inherit from this class directly, and instead should use the CRTP
/// FunctionPass class.
class FunctionPassBase : public Pass {
public:
  static bool classof(const Pass *pass) {
    return pass->getKind() == Kind::FunctionPass;
  }

protected:
  explicit FunctionPassBase(const PassID *id) : Pass(id, Kind::FunctionPass) {}

  /// The polymorphic API that runs the pass over the currently held function.
  virtual PassResult runOnFunction() = 0;

  /// Return the current function being transformed.
  Function &getFunction() {
    assert(passState && "pass state was never initialized");
    return *passState->ir;
  }

private:
  /// Forwarding function to execute this pass.
  PassResult run(Function *fn);

  /// The current execution state for the pass.
  llvm::Optional<detail::PassExecutionState<Function>> passState;

  /// Allow access to 'run'.
  friend detail::FunctionPassExecutor;
};

/// Pass to transform a module. Derived passes should not inherit from this
/// class directly, and instead should use the CRTP ModulePass class.
class ModulePassBase : public Pass {
public:
  static bool classof(const Pass *pass) {
    return pass->getKind() == Kind::ModulePass;
  }

protected:
  explicit ModulePassBase(const PassID *id) : Pass(id, Kind::ModulePass) {}

  /// The polymorphic API that runs the pass over the currently held module.
  virtual PassResult runOnModule() = 0;

  /// Return the current module being transformed.
  Module &getModule() {
    assert(passState && "pass state was never initialized");
    return *passState->ir;
  }

private:
  /// Forwarding function to execute this pass.
  PassResult run(Module *module);

  /// The current execution state for the pass.
  llvm::Optional<detail::PassExecutionState<Module>> passState;

  /// TODO(riverriddle) Remove this using directive when the old pass
  /// functionality is removed.
  using Pass::runOnModule;

  /// Allow access to 'run'.
  friend detail::ModulePassExecutor;
};

//===----------------------------------------------------------------------===//
// Pass Model Definitions
//===----------------------------------------------------------------------===//
namespace detail {
/// The opaque CRTP model of a pass. This class provides utilities for derived
/// pass execution and handles all of the necessary polymorphic API.
template <typename IRUnitT, typename PassT, typename BasePassT>
class PassModel : public BasePassT {
protected:
  PassModel() : BasePassT(PassID::getID<PassT>()) {}

  /// TODO(riverriddle) Provide additional utilities for cloning, getting the
  /// derived class name, etc..
};

// TODO(riverriddle): Move these to the mlir namespace when the current passes
// have been ported.

/// A model for providing function pass specific utilities.
///
/// Function passes must not:
///   - read or modify any other functions within the parent module, as
///     other threads may be manipulating them concurrently.
///   - modify any state within the parent module, this includes adding
///     additional functions.
///
/// Derived function passes are expected to provide the following:
///   - A 'PassResult runOnFunction()' method.
template <typename T>
using FunctionPass = PassModel<Function, T, FunctionPassBase>;

/// A model for providing module pass specific utilities.
///
/// Derived module passes are expected to provide the following:
///   - A 'PassResult runOnModule()' method.
template <typename T> using ModulePass = PassModel<Module, T, ModulePassBase>;
} // end namespace detail
} // end namespace mlir

#endif // MLIR_PASS_PASS_H
