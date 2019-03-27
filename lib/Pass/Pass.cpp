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
#include "PassDetail.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Threading.h"

using namespace mlir;
using namespace mlir::detail;

static llvm::cl::opt<bool> enableThreads(
    "experimental-mt-pm",
    llvm::cl::desc("Enable experimental multithreading in the pass manager"),
    llvm::cl::init(false));

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
  return failure(passFailed);
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
  return failure(passFailed);
}

//===----------------------------------------------------------------------===//
// PassExecutor
//===----------------------------------------------------------------------===//

FunctionPassExecutor::FunctionPassExecutor(const FunctionPassExecutor &rhs)
    : PassExecutor(Kind::FunctionExecutor) {
  for (auto &pass : rhs.passes)
    addPass(pass->clone());
}

/// Run all of the passes in this manager over the current function.
LogicalResult detail::FunctionPassExecutor::run(Function *function,
                                                FunctionAnalysisManager &fam) {
  // Run each of the held passes.
  for (auto &pass : passes)
    if (failed(pass->run(function, fam)))
      return failure();
  return success();
}

/// Run all of the passes in this manager over the current module.
LogicalResult detail::ModulePassExecutor::run(Module *module,
                                              ModuleAnalysisManager &mam) {
  // Run each of the held passes.
  for (auto &pass : passes)
    if (failed(pass->run(module, mam)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleToFunctionPassAdaptor
//===----------------------------------------------------------------------===//

/// Utility to run the given function and analysis manager on a provided
/// function pass executor.
static LogicalResult runFunctionPipeline(FunctionPassExecutor &fpe,
                                         Function *func,
                                         FunctionAnalysisManager &fam) {
  // Run the function pipeline over the provided function.
  auto result = fpe.run(func, fam);

  // Clear out any computed function analyses. These analyses won't be used
  // any more in this pipeline, and this helps reduce the current working set
  // of memory. If preserving these analyses becomes important in the future
  // we can re-evalutate this.
  fam.clear();
  return result;
}

/// Run the held function pipeline over all non-external functions within the
/// module.
void ModuleToFunctionPassAdaptor::runOnModule() {
  ModuleAnalysisManager &mam = getAnalysisManager();
  for (auto &func : getModule()) {
    // Skip external functions.
    if (func.isExternal())
      continue;

    // Run the held function pipeline over the current function.
    auto fam = mam.slice(&func);
    if (failed(runFunctionPipeline(fpe, &func, fam)))
      return signalPassFailure();

    // Clear out any computed function analyses. These analyses won't be used
    // any more in this pipeline, and this helps reduce the current working set
    // of memory. If preserving these analyses becomes important in the future
    // we can re-evalutate this.
    fam.clear();
  }
}

// Run the held function pipeline synchronously across the functions within
// the module.
void ModuleToFunctionPassAdaptorParallel::runOnModule() {
  ModuleAnalysisManager &mam = getAnalysisManager();

  // Create the async executors if they haven't been created, or if the main
  // function pipeline has changed.
  if (asyncExecutors.empty() || asyncExecutors.front().size() != fpe.size())
    asyncExecutors = {llvm::hardware_concurrency(), fpe};

  // Run a prepass over the module to collect the functions to execute a over.
  // This ensures that an analysis manager exists for each function, as well as
  // providing a queue of functions to execute over.
  std::vector<std::pair<Function *, FunctionAnalysisManager>> funcAMPairs;
  for (auto &func : getModule())
    if (!func.isExternal())
      funcAMPairs.emplace_back(&func, mam.slice(&func));

  // An index for the current function/analysis manager pair.
  std::atomic<unsigned> funcIt = 0;

  // An atomic failure variable for the async executors.
  std::atomic<bool> passFailed = false;
  llvm::parallel::for_each(
      llvm::parallel::par, asyncExecutors.begin(),
      std::next(asyncExecutors.begin(),
                std::min(asyncExecutors.size(), funcAMPairs.size())),
      [&](FunctionPassExecutor &executor) {
        for (auto e = funcAMPairs.size(); !passFailed && funcIt < e;) {
          // Get the next available function index.
          unsigned nextID = funcIt++;
          if (nextID >= e)
            break;

          // Run the executor over the current function.
          auto &it = funcAMPairs[nextID];
          if (failed(runFunctionPipeline(executor, it.first, it.second))) {
            passFailed = true;
            break;
          }
        }
      });

  // Signal a failure if any of the executors failed.
  if (passFailed)
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

namespace {
/// Pass to verify a function and signal failure if necessary.
class FunctionVerifier : public FunctionPass<FunctionVerifier> {
  void runOnFunction() {
    if (getFunction().verify())
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
    : mpe(new ModulePassExecutor()), verifyPasses(verifyPasses),
      passTiming(false) {}

PassManager::~PassManager() {}

/// Run the passes within this manager on the provided module.
LogicalResult PassManager::run(Module *module) {
  ModuleAnalysisManager mam(module, instrumentor.get());
  return mpe->run(module, mam);
}

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
    if (enableThreads && llvm::llvm_is_multithreaded()) {
      // If multi-threading is enabled, then create an asynchronous adaptor.
      auto *adaptor = new ModuleToFunctionPassAdaptorParallel();
      addPass(adaptor);
      fpe = &adaptor->getFunctionExecutor();
    } else {
      auto *adaptor = new ModuleToFunctionPassAdaptor();
      addPass(adaptor);
      fpe = &adaptor->getFunctionExecutor();
    }

    /// Add the executor to the stack.
    nestedExecutorStack.push_back(fpe);
  } else {
    fpe = cast<detail::FunctionPassExecutor>(nestedExecutorStack.back());
  }
  fpe->addPass(pass);

  // Add a verifier run if requested.
  if (verifyPasses)
    fpe->addPass(new FunctionVerifier());
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

//===----------------------------------------------------------------------===//
// PassInstrumentation
//===----------------------------------------------------------------------===//

PassInstrumentation::~PassInstrumentation() {}

//===----------------------------------------------------------------------===//
// PassInstrumentor
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct PassInstrumentorImpl {
  /// Mutex to keep instrumentation access thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// Set of registered instrumentations.
  std::vector<std::unique_ptr<PassInstrumentation>> instrumentations;
};
} // end namespace detail
} // end namespace mlir

PassInstrumentor::PassInstrumentor() : impl(new PassInstrumentorImpl()) {}
PassInstrumentor::~PassInstrumentor() {}

/// See PassInstrumentation::runBeforePass for details.
void PassInstrumentor::runBeforePass(Pass *pass, const llvm::Any &ir) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePass(pass, ir);
}

/// See PassInstrumentation::runAfterPass for details.
void PassInstrumentor::runAfterPass(Pass *pass, const llvm::Any &ir) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPass(pass, ir);
}

/// See PassInstrumentation::runAfterPassFailed for details.
void PassInstrumentor::runAfterPassFailed(Pass *pass, const llvm::Any &ir) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPassFailed(pass, ir);
}

/// See PassInstrumentation::runBeforeAnalysis for details.
void PassInstrumentor::runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                                         const llvm::Any &ir) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforeAnalysis(name, id, ir);
}

/// See PassInstrumentation::runAfterAnalysis for details.
void PassInstrumentor::runAfterAnalysis(llvm::StringRef name, AnalysisID *id,
                                        const llvm::Any &ir) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterAnalysis(name, id, ir);
}

/// Add the given instrumentation to the collection. This takes ownership over
/// the given pointer.
void PassInstrumentor::addInstrumentation(PassInstrumentation *pi) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  impl->instrumentations.emplace_back(pi);
}

constexpr AnalysisID mlir::detail::PreservedAnalyses::allAnalysesID;
