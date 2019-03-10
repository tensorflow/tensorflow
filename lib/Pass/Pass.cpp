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
LogicalResult FunctionPassBase::run(Function *fn,
                                    FunctionAnalysisManager &fam) {
  // Initialize the pass state.
  passState.emplace(fn, fam);

  // Instrument before the pass has run.
  auto pi = fam.getPassInstrumentor();
  if (pi)
    pi->runBeforePass(this, fn);

  // Invoke the virtual runOnFunction function.
  runOnFunction();

  // Invalidate any non preserved analyses.
  fam.invalidate(passState->preservedAnalyses);

  // Instrument after the pass has run.
  bool passFailed = passState->irAndPassFailed.getInt();
  if (pi) {
    if (passFailed)
      pi->runAfterPassFailed(this, fn);
    else
      pi->runAfterPass(this, fn);
  }

  // Return if the pass signaled a failure.
  return passFailed ? LogicalResult::failure() : LogicalResult::success();
}

/// Forwarding function to execute this pass.
LogicalResult ModulePassBase::run(Module *module, ModuleAnalysisManager &mam) {
  // Initialize the pass state.
  passState.emplace(module, mam);

  // Instrument before the pass has run.
  auto pi = mam.getPassInstrumentor();
  if (pi)
    pi->runBeforePass(this, module);

  // Invoke the virtual runOnModule function.
  runOnModule();

  // Invalidate any non preserved analyses.
  mam.invalidate(passState->preservedAnalyses);

  // Instrument after the pass has run.
  bool passFailed = passState->irAndPassFailed.getInt();
  if (pi) {
    if (passFailed)
      pi->runAfterPassFailed(this, module);
    else
      pi->runAfterPass(this, module);
  }

  // Return if the pass signaled a failure.
  return passFailed ? LogicalResult::failure() : LogicalResult::success();
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

  /// Run the executor on the given function.
  LogicalResult run(Function *function, FunctionAnalysisManager &fam);

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

  /// Run the executor on the given module.
  LogicalResult run(Module *module, ModuleAnalysisManager &mam);

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
LogicalResult detail::FunctionPassExecutor::run(Function *function,
                                                FunctionAnalysisManager &fam) {
  // Run each of the held passes.
  for (auto &pass : passes)
    if (failed(pass->run(function, fam)))
      return LogicalResult::failure();
  return LogicalResult::success();
}

/// Run all of the passes in this manager over the current module.
LogicalResult detail::ModulePassExecutor::run(Module *module,
                                              ModuleAnalysisManager &mam) {
  // Run each of the held passes.
  for (auto &pass : passes)
    if (failed(pass->run(module, mam)))
      return LogicalResult::failure();
  return LogicalResult::success();
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
  ModuleAnalysisManager &mam = getAnalysisManager();
  for (auto &func : getModule()) {
    // Skip external functions.
    if (func.isExternal())
      continue;

    // Run the held function pipeline over the current function.
    auto fam = mam.slice(&func);
    if (failed(fpe.run(&func, fam)))
      return signalPassFailure();

    // Clear out any computed function analyses. These analyses won't be used
    // any more in this pipeline, and this helps reduce the current working set
    // of memory. If preserving these analyses becomes important in the future
    // we can re-evalutate this.
    fam.clear();
  }
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

namespace {
/// Pass to verify a function and signal failure if necessary.
class FunctionVerifier : public FunctionPass<FunctionVerifier> {
  void runOnFunction() {
    if (getFunction()->verify())
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

/// Pass to verify a module and signal failure if necessary.
class ModuleVerifier : public ModulePass<ModuleVerifier> {
  void runOnModule() {
    if (getModule().verify())
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};
} // end anonymous namespace

PassManager::PassManager(bool verifyPasses)
    : mpe(new ModulePassExecutor()), verifyPasses(verifyPasses) {}

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

  // Add a verifier run if requested.
  if (verifyPasses)
    mpe->addPass(new ModuleVerifier());
}

/// Add a function pass to the current manager. This takes ownership over the
/// provided pass pointer. This will automatically create a function pass
/// executor if necessary.
void PassManager::addPass(FunctionPassBase *pass) {
  detail::FunctionPassExecutor *fpe;
  if (nestedExecutorStack.empty()) {
    /// Create an executor adaptor for this pass.
    auto *adaptor = new ModuleToFunctionPassAdaptor();
    addPass(adaptor);

    /// Add the executor to the stack.
    fpe = &adaptor->getFunctionExecutor();
    nestedExecutorStack.push_back(fpe);
  } else {
    fpe = cast<detail::FunctionPassExecutor>(nestedExecutorStack.back());
  }
  fpe->addPass(pass);

  // Add a verifier run if requested.
  if (verifyPasses)
    fpe->addPass(new FunctionVerifier());
}

/// Run the passes within this manager on the provided module.
LogicalResult PassManager::run(Module *module) {
  ModuleAnalysisManager mam(module, instrumentor.get());
  return mpe->run(module, mam);
}

/// Add the provided instrumentation to the pass manager. This takes ownership
/// over the given pointer.
void PassManager::addInstrumentation(PassInstrumentation *pi) {
  if (!instrumentor)
    instrumentor.reset(new PassInstrumentor());

  instrumentor->addInstrumentation(pi);
}

//===----------------------------------------------------------------------===//
// AnalysisManager
//===----------------------------------------------------------------------===//

/// Returns a pass instrumentation object for the current function.
PassInstrumentor *FunctionAnalysisManager::getPassInstrumentor() const {
  return parent->getPassInstrumentor();
}

/// Create an analysis slice for the given child function.
FunctionAnalysisManager ModuleAnalysisManager::slice(Function *function) {
  assert(function->getModule() == moduleAnalyses.getIRUnit() &&
         "function has a different parent module");
  auto it = functionAnalyses.try_emplace(function, function);
  return {this, &it.first->second};
}

/// Invalidate any non preserved analyses.
void ModuleAnalysisManager::invalidate(const detail::PreservedAnalyses &pa) {
  // If all analyses were preserved, then there is nothing to do here.
  if (pa.isAll())
    return;

  // Invalidate the module analyses directly.
  moduleAnalyses.invalidate(pa);

  // If no analyses were preserved, then just simply clear out the function
  // analysis results.
  if (pa.isNone()) {
    functionAnalyses.clear();
    return;
  }

  // Otherwise, invalidate each function analyses.
  for (auto &analysisPair : functionAnalyses)
    analysisPair.second.invalidate(pa);
}
