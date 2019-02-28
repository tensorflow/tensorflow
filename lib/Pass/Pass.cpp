//===- Pass.cpp - Pass infrastructure implementation ----------------------===//
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
// This file implements common pass infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Forwarding function to execute this pass.
bool FunctionPassBase::run(Function *fn) {
  /// Initialize the pass state.
  passState.emplace(fn);

  /// Invoke the virtual runOnFunction function.
  runOnFunction();

  // Return false if the pass signaled a failure.
  return !passState->irAndPassFailed.getInt();
}

/// Forwarding function to execute this pass. Returns false if the pass
/// execution failed, true otherwise.
bool ModulePassBase::run(Module *module) {
  /// Initialize the pass state.
  passState.emplace(module);

  /// Invoke the virtual runOnModule function.
  runOnModule();

  // Return false if the pass signaled a failure.
  return !passState->irAndPassFailed.getInt();
}

//===----------------------------------------------------------------------===//
// PassExecutor
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
/// The abstract base pass executor class.
class PassExecutor {
public:
  enum Kind { FunctionExecutor, ModuleExecutor };
  explicit PassExecutor(Kind kind) : kind(kind) {}

  /// Get the kind of this executor.
  Kind getKind() const { return kind; }

private:
  /// The kind of executor this object is.
  Kind kind;
};

/// A pass executor that contains a list of passes over a function.
class FunctionPassExecutor : public PassExecutor {
public:
  FunctionPassExecutor() : PassExecutor(Kind::FunctionExecutor) {}
  FunctionPassExecutor(FunctionPassExecutor &&) = default;

  // TODO(riverriddle) Allow copying.
  FunctionPassExecutor(const FunctionPassExecutor &) = delete;
  FunctionPassExecutor &operator=(const FunctionPassExecutor &) = delete;

  /// Run the executor on the given function. Returns false if the pass
  /// execution failed, true otherwise.
  LLVM_NODISCARD
  bool run(Function *function);

  /// Add a pass to the current executor. This takes ownership over the provided
  /// pass pointer.
  void addPass(FunctionPassBase *pass) { passes.emplace_back(pass); }

  static bool classof(const PassExecutor *pe) {
    return pe->getKind() == Kind::FunctionExecutor;
  }

private:
  std::vector<std::unique_ptr<FunctionPassBase>> passes;
};

/// A pass executor that contains a list of passes over a module unit.
class ModulePassExecutor : public PassExecutor {
public:
  ModulePassExecutor() : PassExecutor(Kind::ModuleExecutor) {}
  ModulePassExecutor(ModulePassExecutor &&) = default;

  // Don't allow copying.
  ModulePassExecutor(const ModulePassExecutor &) = delete;
  ModulePassExecutor &operator=(const ModulePassExecutor &) = delete;

  /// Run the executor on the given module. Returns false if the pass
  /// execution failed, true otherwise.
  LLVM_NODISCARD
  bool run(Module *module);

  /// Add a pass to the current executor. This takes ownership over the provided
  /// pass pointer.
  void addPass(ModulePassBase *pass) { passes.emplace_back(pass); }

  static bool classof(const PassExecutor *pe) {
    return pe->getKind() == Kind::ModuleExecutor;
  }

private:
  /// Set of passes to run on the given module.
  std::vector<std::unique_ptr<ModulePassBase>> passes;
};
} // end namespace detail
} // end namespace mlir

/// Run all of the passes in this manager over the current function.
bool detail::FunctionPassExecutor::run(Function *function) {
  for (auto &pass : passes) {
    /// Create an execution state for this pass.
    if (!pass->run(function))
      return false;
    // TODO: This should be opt-out and handled separately.
    if (function->verify())
      return false;
  }
  return true;
}

/// Run all of the passes in this manager over the current module.
bool detail::ModulePassExecutor::run(Module *module) {
  for (auto &pass : passes) {
    if (!pass->run(module))
      return false;
    // TODO: This should be opt-out and handled separately.
    if (module->verify())
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ModuleToFunctionPassAdaptor
//===----------------------------------------------------------------------===//

namespace {
/// An adaptor module pass used to run function passes over all of the
/// non-external functions of a module.
class ModuleToFunctionPassAdaptor
    : public ModulePass<ModuleToFunctionPassAdaptor> {
public:
  ModuleToFunctionPassAdaptor() = default;
  ModuleToFunctionPassAdaptor(ModuleToFunctionPassAdaptor &&) = default;

  // TODO(riverriddle) Allow copying.
  ModuleToFunctionPassAdaptor(const ModuleToFunctionPassAdaptor &) = delete;
  ModuleToFunctionPassAdaptor &
  operator=(const ModuleToFunctionPassAdaptor &) = delete;

  /// Run the held function pipeline over all non-external functions within the
  /// module.
  void runOnModule() override;

  /// Returns the function pass executor for this adaptor.
  FunctionPassExecutor &getFunctionExecutor() { return fpe; }

private:
  FunctionPassExecutor fpe;
};
} // end anonymous namespace

/// Execute the held function pass over all non-external functions within the
/// module.
void ModuleToFunctionPassAdaptor::runOnModule() {
  for (auto &func : getModule()) {
    // Skip external functions.
    if (func.isExternal())
      continue;

    // Run the held function pipeline over the current function.
    if (!fpe.run(&func))
      return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::PassManager() : mpe(new ModulePassExecutor()) {}

PassManager::~PassManager() {}

/// Add an opaque pass pointer to the current manager. This takes ownership
/// over the provided pass pointer.
void PassManager::addPass(Pass *pass) {
  switch (pass->getKind()) {
  case Pass::Kind::FunctionPass:
    addPass(cast<FunctionPassBase>(pass));
    break;
  case Pass::Kind::ModulePass:
    addPass(cast<ModulePassBase>(pass));
    break;
  }
}

/// Add a module pass to the current manager. This takes ownership over the
/// provided pass pointer.
void PassManager::addPass(ModulePassBase *pass) {
  nestedExecutorStack.clear();
  mpe->addPass(pass);
}

/// Add a function pass to the current manager. This takes ownership over the
/// provided pass pointer. This will automatically create a function pass
/// executor if necessary.
void PassManager::addPass(FunctionPassBase *pass) {
  detail::FunctionPassExecutor *fpe;
  if (nestedExecutorStack.empty()) {
    /// Create an executor adaptor for this pass.
    auto *adaptor = new ModuleToFunctionPassAdaptor();
    mpe->addPass(adaptor);

    /// Add the executor to the stack.
    fpe = &adaptor->getFunctionExecutor();
    nestedExecutorStack.push_back(fpe);
  } else {
    fpe = cast<detail::FunctionPassExecutor>(nestedExecutorStack.back());
  }
  fpe->addPass(pass);
}

/// Run the passes within this manager on the provided module.
bool PassManager::run(Module *module) { return mpe->run(module); }
