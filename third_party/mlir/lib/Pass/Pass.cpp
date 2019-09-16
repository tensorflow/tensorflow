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
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Threading.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Forwarding function to execute this pass.
LogicalResult Pass::run(Operation *op, AnalysisManager am) {
  passState.emplace(op, am);

  // Instrument before the pass has run.
  auto pi = am.getPassInstrumentor();
  if (pi)
    pi->runBeforePass(this, op);

  // Invoke the virtual runOnOperation method.
  runOnOperation();

  // Invalidate any non preserved analyses.
  am.invalidate(passState->preservedAnalyses);

  // Instrument after the pass has run.
  bool passFailed = passState->irAndPassFailed.getInt();
  if (pi) {
    if (passFailed)
      pi->runAfterPassFailed(this, op);
    else
      pi->runAfterPass(this, op);
  }

  // Return if the pass signaled a failure.
  return failure(passFailed);
}

//===----------------------------------------------------------------------===//
// Verifier Passes
//===----------------------------------------------------------------------===//

void VerifierPass::runOnOperation() {
  if (failed(verify(getOperation())))
    signalPassFailure();
  markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// OpPassManager
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct OpPassManagerImpl {
  OpPassManagerImpl(OperationName name, bool disableThreads, bool verifyPasses)
      : name(name), disableThreads(disableThreads), verifyPasses(verifyPasses) {
  }

  /// Returns the pass manager instance corresponding to the last pass added
  /// if that pass was a PassAdaptor.
  OpPassManager *getLastNestedPM() {
    if (passes.empty())
      return nullptr;
    auto lastPassIt = passes.rbegin();

    // If this pass was a verifier, skip it as it is opaque to ordering for
    // pipeline construction.
    if (isa<VerifierPass>(*lastPassIt))
      ++lastPassIt;

    // Get the internal pass manager if this pass is an adaptor.
    if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(lastPassIt->get()))
      return &adaptor->getPassManager();
    if (auto *adaptor = dyn_cast<OpToOpPassAdaptorParallel>(lastPassIt->get()))
      return &adaptor->getPassManager();
    return nullptr;
  }

  /// The name of the operation that passes of this pass manager operate on.
  OperationName name;

  /// Flag to disable multi-threading of passes.
  bool disableThreads : 1;

  /// Flag that specifies if the IR should be verified after each pass has run.
  bool verifyPasses : 1;

  /// The set of passes to run as part of this pass manager.
  std::vector<std::unique_ptr<Pass>> passes;
};
} // end namespace detail
} // end namespace mlir

OpPassManager::OpPassManager(OperationName name, bool disableThreads,
                             bool verifyPasses)
    : impl(new OpPassManagerImpl(name, disableThreads, verifyPasses)) {
  assert(name.getAbstractOperation() &&
         "OpPassManager can only operate on registered operations");
  assert(name.getAbstractOperation()->hasProperty(
             OperationProperty::IsolatedFromAbove) &&
         "OpPassManager only supports operating on operations marked as "
         "'IsolatedFromAbove'");
}
OpPassManager::OpPassManager(const OpPassManager &rhs)
    : impl(new OpPassManagerImpl(rhs.impl->name, rhs.impl->disableThreads,
                                 rhs.impl->verifyPasses)) {
  for (auto &pass : rhs.impl->passes)
    impl->passes.emplace_back(pass->clone());
}

OpPassManager::~OpPassManager() {}

/// Run all of the passes in this manager over the current operation.
LogicalResult OpPassManager::run(Operation *op, AnalysisManager am) {
  // Run each of the held passes.
  for (auto &pass : impl->passes)
    if (failed(pass->run(op, am)))
      return failure();
  return success();
}

/// Nest a new operation pass manager for the given operation kind under this
/// pass manager.
OpPassManager &OpPassManager::nest(const OperationName &nestedName) {
  // Check to see if an existing nested pass manager already exists.
  if (auto *nestedPM = impl->getLastNestedPM()) {
    if (nestedPM->getOpName() == nestedName)
      return *nestedPM;
  }

  std::unique_ptr<OpPassManager> nested(
      new OpPassManager(nestedName, impl->disableThreads, impl->verifyPasses));
  auto &nestedRef = *nested;

  /// Create an executor adaptor for this pass. If multi-threading is disabled,
  /// then create a synchronous adaptor.
  if (impl->disableThreads || !llvm::llvm_is_multithreaded())
    addPass(std::make_unique<OpToOpPassAdaptor>(std::move(nested)));
  else
    addPass(std::make_unique<OpToOpPassAdaptorParallel>(std::move(nested)));
  return nestedRef;
}
OpPassManager &OpPassManager::nest(StringRef nestedName) {
  return nest(OperationName(nestedName, getContext()));
}

/// Add the given pass to this pass manager. The pass must either be an opaque
/// `OperationPass`, or an `OpPass` that operates on operations of the same
/// type as this pass manager.
void OpPassManager::addPass(std::unique_ptr<Pass> pass) {
  // If this pass runs on a different operation than this pass manager, then
  // implicitly nest a pass manager for this operation.
  auto passOpName = pass->getOpName();
  if (passOpName && passOpName != impl->name.getStringRef())
    return nest(*passOpName).addPass(std::move(pass));

  impl->passes.emplace_back(std::move(pass));
  if (impl->verifyPasses)
    impl->passes.emplace_back(std::make_unique<VerifierPass>());
}

/// Returns the number of passes held by this manager.
size_t OpPassManager::size() const { return impl->passes.size(); }

/// Returns the internal implementation instance.
OpPassManagerImpl &OpPassManager::getImpl() { return *impl; }

/// Return an instance of the context.
MLIRContext *OpPassManager::getContext() const {
  return impl->name.getAbstractOperation()->dialect.getContext();
}

/// Return the operation name that this pass manager operates on.
const OperationName &OpPassManager::getOpName() const { return impl->name; }

//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

/// Utility to run the given operation and analysis manager on a provided op
/// pass manager.
static LogicalResult runPipeline(OpPassManager &pm, Operation *op,
                                 AnalysisManager am) {
  // Run the pipeline over the provided operation.
  auto result = pm.run(op, am);

  // Clear out any computed operation analyses. These analyses won't be used
  // any more in this pipeline, and this helps reduce the current working set
  // of memory. If preserving these analyses becomes important in the future
  // we can re-evalutate this.
  am.clear();
  return result;
}

OpToOpPassAdaptor::OpToOpPassAdaptor(std::unique_ptr<OpPassManager> mgr)
    : mgr(std::move(mgr)) {}
OpToOpPassAdaptor::OpToOpPassAdaptor(const OpToOpPassAdaptor &rhs)
    : mgr(new OpPassManager(*rhs.mgr)) {}

/// Run the held pipeline over all nested operations.
void OpToOpPassAdaptor::runOnOperation() {
  auto am = getAnalysisManager();
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        // Run the held pipeline over the current operation.
        if (op.getName() == mgr->getOpName() &&
            failed(runPipeline(*mgr, &op, am.slice(&op))))
          return signalPassFailure();
      }
    }
  }
}

OpToOpPassAdaptorParallel::OpToOpPassAdaptorParallel(
    std::unique_ptr<OpPassManager> mgr)
    : mgr(std::move(mgr)) {}
OpToOpPassAdaptorParallel::OpToOpPassAdaptorParallel(
    const OpToOpPassAdaptorParallel &rhs)
    : mgr(std::make_unique<OpPassManager>(*rhs.mgr)) {}

// Run the held pipeline asynchronously across the functions within the module.
void OpToOpPassAdaptorParallel::runOnOperation() {
  AnalysisManager am = getAnalysisManager();

  // Create the async executors if they haven't been created, or if the main
  // pipeline has changed.
  if (asyncExecutors.empty() || asyncExecutors.front().size() != mgr->size())
    asyncExecutors = {llvm::hardware_concurrency(), *mgr};

  // Run a prepass over the module to collect the operations to execute over.
  // This ensures that an analysis manager exists for each operation, as well as
  // providing a queue of operations to execute over.
  std::vector<std::pair<Operation *, AnalysisManager>> opAMPairs;
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        // Add this operation iff the name matches the current pass manager.
        if (op.getName() == mgr->getOpName())
          opAMPairs.emplace_back(&op, am.slice(&op));
      }
    }
  }

  // A parallel diagnostic handler that provides deterministic diagnostic
  // ordering.
  ParallelDiagnosticHandler diagHandler(&getContext());

  // An index for the current operation/analysis manager pair.
  std::atomic<unsigned> opIt(0);

  // An atomic failure variable for the async executors.
  std::atomic<bool> passFailed(false);
  llvm::parallel::for_each(
      llvm::parallel::par, asyncExecutors.begin(),
      std::next(asyncExecutors.begin(),
                std::min(asyncExecutors.size(), opAMPairs.size())),
      [&](OpPassManager &pm) {
        for (auto e = opAMPairs.size(); !passFailed && opIt < e;) {
          // Get the next available operation index.
          unsigned nextID = opIt++;
          if (nextID >= e)
            break;

          // Set the order id for this thread in the diagnostic handler.
          diagHandler.setOrderIDForThread(nextID);

          // Run the executor over the current operation.
          auto &it = opAMPairs[nextID];
          if (failed(runPipeline(pm, it.first, it.second))) {
            passFailed = true;
            break;
          }
        }
      });

  // Signal a failure if any of the executors failed.
  if (passFailed)
    signalPassFailure();
}

/// Utility function to return the operation name that the given adaptor pass
/// operates on. Return None if the given pass is not an adaptor pass.
Optional<StringRef> mlir::detail::getAdaptorPassOpName(Pass *pass) {
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(pass))
    return adaptor->getPassManager().getOpName().getStringRef();
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptorParallel>(pass))
    return adaptor->getPassManager().getOpName().getStringRef();
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::PassManager(MLIRContext *ctx, bool verifyPasses)
    : opPassManager(OperationName(ModuleOp::getOperationName(), ctx),
                    /*disableThreads=*/false, verifyPasses),
      passTiming(false) {}

PassManager::~PassManager() {}

/// Run the passes within this manager on the provided module.
LogicalResult PassManager::run(ModuleOp module) {
  ModuleAnalysisManager am(module, instrumentor.get());
  return opPassManager.run(module, am);
}

/// Disable support for multi-threading within the pass manager.
void PassManager::disableMultithreading(bool disable) {
  opPassManager.getImpl().disableThreads = disable;
}

/// Add an opaque pass pointer to the current manager. This takes ownership
/// over the provided pass pointer.
void PassManager::addPass(std::unique_ptr<Pass> pass) {
  opPassManager.addPass(std::move(pass));
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

/// Returns a pass instrumentation object for the current operation.
PassInstrumentor *AnalysisManager::getPassInstrumentor() const {
  ParentPointerT curParent = parent;
  while (auto *parentAM = curParent.dyn_cast<const AnalysisManager *>())
    curParent = parentAM->parent;
  return curParent.get<const ModuleAnalysisManager *>()->getPassInstrumentor();
}

/// Get an analysis manager for the given child operation.
AnalysisManager AnalysisManager::slice(Operation *op) {
  assert(op->getParentOp() == impl->getOperation() &&
         "'op' has a different parent operation");
  auto it = impl->childAnalyses.find(op);
  if (it == impl->childAnalyses.end())
    it = impl->childAnalyses
             .try_emplace(op, std::make_unique<NestedAnalysisMap>(op))
             .first;
  return {this, it->second.get()};
}

/// Invalidate any non preserved analyses.
void detail::NestedAnalysisMap::invalidate(
    const detail::PreservedAnalyses &pa) {
  // If all analyses were preserved, then there is nothing to do here.
  if (pa.isAll())
    return;

  // Invalidate the analyses for the current operation directly.
  analyses.invalidate(pa);

  // If no analyses were preserved, then just simply clear out the child
  // analysis results.
  if (pa.isNone()) {
    childAnalyses.clear();
    return;
  }

  // Otherwise, invalidate each child analysis map.
  SmallVector<NestedAnalysisMap *, 8> mapsToInvalidate(1, this);
  while (!mapsToInvalidate.empty()) {
    auto *map = mapsToInvalidate.pop_back_val();
    for (auto &analysisPair : map->childAnalyses) {
      analysisPair.second->invalidate(pa);
      if (!analysisPair.second->childAnalyses.empty())
        mapsToInvalidate.push_back(analysisPair.second.get());
    }
  }
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
void PassInstrumentor::runBeforePass(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePass(pass, op);
}

/// See PassInstrumentation::runAfterPass for details.
void PassInstrumentor::runAfterPass(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPass(pass, op);
}

/// See PassInstrumentation::runAfterPassFailed for details.
void PassInstrumentor::runAfterPassFailed(Pass *pass, Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPassFailed(pass, op);
}

/// See PassInstrumentation::runBeforeAnalysis for details.
void PassInstrumentor::runBeforeAnalysis(llvm::StringRef name, AnalysisID *id,
                                         Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforeAnalysis(name, id, op);
}

/// See PassInstrumentation::runAfterAnalysis for details.
void PassInstrumentor::runAfterAnalysis(llvm::StringRef name, AnalysisID *id,
                                        Operation *op) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterAnalysis(name, id, op);
}

/// Add the given instrumentation to the collection. This takes ownership over
/// the given pointer.
void PassInstrumentor::addInstrumentation(PassInstrumentation *pi) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  impl->instrumentations.emplace_back(pi);
}

constexpr AnalysisID mlir::detail::PreservedAnalyses::allAnalysesID;
