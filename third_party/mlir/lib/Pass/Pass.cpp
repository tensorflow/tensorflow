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
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Prints out the pass in the textual representation of pipelines. If this is
/// an adaptor pass, print with the op_name(sub_pass,...) format.
void Pass::printAsTextualPipeline(raw_ostream &os) {
  // Special case for adaptors to use the 'op_name(sub_passes)' format.
  if (auto *adaptor = getAdaptorPassBase(this)) {
    interleaveComma(adaptor->getPassManagers(), os, [&](OpPassManager &pm) {
      os << pm.getOpName() << "(";
      pm.printAsTextualPipeline(os);
      os << ")";
    });
  } else if (const PassInfo *info = lookupPassInfo()) {
    os << info->getPassArgument();
  } else {
    os << getName();
  }
}

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
// OpPassManagerImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct OpPassManagerImpl {
  OpPassManagerImpl(OperationName name, bool disableThreads, bool verifyPasses)
      : name(name), disableThreads(disableThreads), verifyPasses(verifyPasses) {
  }

  /// Merge the passes of this pass manager into the one provided.
  void mergeInto(OpPassManagerImpl &rhs) {
    assert(name == rhs.name && "merging unrelated pass managers");
    for (auto &pass : passes)
      rhs.passes.push_back(std::move(pass));
    passes.clear();
  }

  /// Coalesce adjacent AdaptorPasses into one large adaptor. This runs
  /// recursively through the pipeline graph.
  void coalesceAdjacentAdaptorPasses();

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

/// Coalesce adjacent AdaptorPasses into one large adaptor. This runs
/// recursively through the pipeline graph.
void OpPassManagerImpl::coalesceAdjacentAdaptorPasses() {
  // Bail out early if there are no adaptor passes.
  if (llvm::none_of(passes, [](std::unique_ptr<Pass> &pass) {
        return isAdaptorPass(pass.get());
      }))
    return;

  // Walk the pass list and merge adjacent adaptors.
  OpToOpPassAdaptorBase *lastAdaptor = nullptr;
  for (auto it = passes.begin(), e = passes.end(); it != e; ++it) {
    // Check to see if this pass is an adaptor.
    if (auto *currentAdaptor = getAdaptorPassBase(it->get())) {
      // If it is the first adaptor in a possible chain, remember it and
      // continue.
      if (!lastAdaptor) {
        lastAdaptor = currentAdaptor;
        continue;
      }

      // Otherwise, merge into the existing adaptor and delete the current one.
      currentAdaptor->mergeInto(*lastAdaptor);
      it->reset();

      // If the verifier is enabled, then next pass is a verifier run so
      // drop it. Verifier passes are inserted after every pass, so this one
      // would be a duplicate.
      if (verifyPasses) {
        assert(std::next(it) != e && isa<VerifierPass>(*std::next(it)));
        (++it)->reset();
      }
    } else if (lastAdaptor && !isa<VerifierPass>(*it)) {
      // If this pass is not an adaptor and not a verifier pass, then coalesce
      // and forget any existing adaptor.
      for (auto &pm : lastAdaptor->getPassManagers())
        pm.getImpl().coalesceAdjacentAdaptorPasses();
      lastAdaptor = nullptr;
    }
  }

  // If there was an adaptor at the end of the manager, coalesce it as well.
  if (lastAdaptor) {
    for (auto &pm : lastAdaptor->getPassManagers())
      pm.getImpl().coalesceAdjacentAdaptorPasses();
  }

  // Now that the adaptors have been merged, erase the empty slot corresponding
  // to the merged adaptors that were nulled-out in the loop above.
  llvm::erase_if(passes, std::logical_not<std::unique_ptr<Pass>>());
}

//===----------------------------------------------------------------------===//
// OpPassManager
//===----------------------------------------------------------------------===//

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
OpPassManager::OpPassManager(OpPassManager &&rhs) : impl(std::move(rhs.impl)) {}
OpPassManager::OpPassManager(const OpPassManager &rhs) { *this = rhs; }
OpPassManager &OpPassManager::operator=(const OpPassManager &rhs) {
  impl.reset(new OpPassManagerImpl(rhs.impl->name, rhs.impl->disableThreads,
                                   rhs.impl->verifyPasses));
  for (auto &pass : rhs.impl->passes)
    impl->passes.emplace_back(pass->clone());
  return *this;
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
  OpPassManager nested(nestedName, impl->disableThreads, impl->verifyPasses);

  /// Create an adaptor for this pass. If multi-threading is disabled, then
  /// create a synchronous adaptor.
  if (impl->disableThreads || !llvm::llvm_is_multithreaded()) {
    auto *adaptor = new OpToOpPassAdaptor(std::move(nested));
    addPass(std::unique_ptr<Pass>(adaptor));
    return adaptor->getPassManagers().front();
  }

  auto *adaptor = new OpToOpPassAdaptorParallel(std::move(nested));
  addPass(std::unique_ptr<Pass>(adaptor));
  return adaptor->getPassManagers().front();
}
OpPassManager &OpPassManager::nest(StringRef nestedName) {
  return nest(OperationName(nestedName, getContext()));
}

/// Add the given pass to this pass manager. If this pass has a concrete
/// operation type, it must be the same type as this pass manager.
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

/// Prints out the passes of the pass mangager as the textual representation
/// of pipelines.
void OpPassManager::printAsTextualPipeline(raw_ostream &os) {
  // Filter out passes that are not part of the public pipeline.
  auto filteredPasses = llvm::make_filter_range(
      impl->passes, [](const std::unique_ptr<Pass> &pass) {
        return !isa<VerifierPass>(pass);
      });
  interleaveComma(filteredPasses, os, [&](const std::unique_ptr<Pass> &pass) {
    pass->printAsTextualPipeline(os);
  });
}

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

/// Find an operation pass manager that can operate on an operation of the given
/// type, or nullptr if one does not exist.
static OpPassManager *findPassManagerFor(MutableArrayRef<OpPassManager> mgrs,
                                         const OperationName &name) {
  auto it = llvm::find_if(
      mgrs, [&](OpPassManager &mgr) { return mgr.getOpName() == name; });
  return it == mgrs.end() ? nullptr : &*it;
}

OpToOpPassAdaptorBase::OpToOpPassAdaptorBase(OpPassManager &&mgr) {
  mgrs.emplace_back(std::move(mgr));
}

/// Merge the current pass adaptor into given 'rhs'.
void OpToOpPassAdaptorBase::mergeInto(OpToOpPassAdaptorBase &rhs) {
  for (auto &pm : mgrs) {
    // If an existing pass manager exists, then merge the given pass manager
    // into it.
    if (auto *existingPM = findPassManagerFor(rhs.mgrs, pm.getOpName())) {
      pm.getImpl().mergeInto(existingPM->getImpl());
    } else {
      // Otherwise, add the given pass manager to the list.
      rhs.mgrs.emplace_back(std::move(pm));
    }
  }
  mgrs.clear();

  // After coalescing, sort the pass managers within rhs by name.
  llvm::array_pod_sort(rhs.mgrs.begin(), rhs.mgrs.end(),
                       [](const OpPassManager *lhs, const OpPassManager *rhs) {
                         return lhs->getOpName().getStringRef().compare(
                             rhs->getOpName().getStringRef());
                       });
}

OpToOpPassAdaptor::OpToOpPassAdaptor(OpPassManager &&mgr)
    : OpToOpPassAdaptorBase(std::move(mgr)) {}

/// Run the held pipeline over all nested operations.
void OpToOpPassAdaptor::runOnOperation() {
  auto am = getAnalysisManager();
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        this};
  auto *instrumentor = am.getPassInstrumentor();
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        auto *mgr = findPassManagerFor(mgrs, op.getName());
        if (!mgr)
          continue;

        // Run the held pipeline over the current operation.
        if (instrumentor)
          instrumentor->runBeforePipeline(mgr->getOpName(), parentInfo);
        auto result = runPipeline(*mgr, &op, am.slice(&op));
        if (instrumentor)
          instrumentor->runAfterPipeline(mgr->getOpName(), parentInfo);

        if (failed(result))
          return signalPassFailure();
      }
    }
  }
}

OpToOpPassAdaptorParallel::OpToOpPassAdaptorParallel(OpPassManager &&mgr)
    : OpToOpPassAdaptorBase(std::move(mgr)) {}

/// Utility functor that checks if the two ranges of pass managers have a size
/// mismatch.
static bool hasSizeMismatch(ArrayRef<OpPassManager> lhs,
                            ArrayRef<OpPassManager> rhs) {
  return lhs.size() != rhs.size() ||
         llvm::any_of(llvm::seq<size_t>(0, lhs.size()),
                      [&](size_t i) { return lhs[i].size() != rhs[i].size(); });
}

// Run the held pipeline asynchronously across the functions within the module.
void OpToOpPassAdaptorParallel::runOnOperation() {
  AnalysisManager am = getAnalysisManager();

  // Create the async executors if they haven't been created, or if the main
  // pipeline has changed.
  if (asyncExecutors.empty() || hasSizeMismatch(asyncExecutors.front(), mgrs))
    asyncExecutors.assign(llvm::hardware_concurrency(), mgrs);

  // Run a prepass over the module to collect the operations to execute over.
  // This ensures that an analysis manager exists for each operation, as well as
  // providing a queue of operations to execute over.
  std::vector<std::pair<Operation *, AnalysisManager>> opAMPairs;
  for (auto &region : getOperation()->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        // Add this operation iff the name matches the any of the pass managers.
        if (findPassManagerFor(mgrs, op.getName()))
          opAMPairs.emplace_back(&op, am.slice(&op));
      }
    }
  }

  // A parallel diagnostic handler that provides deterministic diagnostic
  // ordering.
  ParallelDiagnosticHandler diagHandler(&getContext());

  // An index for the current operation/analysis manager pair.
  std::atomic<unsigned> opIt(0);

  // Get the current thread for this adaptor.
  PassInstrumentation::PipelineParentInfo parentInfo = {llvm::get_threadid(),
                                                        this};
  auto *instrumentor = am.getPassInstrumentor();

  // An atomic failure variable for the async executors.
  std::atomic<bool> passFailed(false);
  llvm::parallel::for_each(
      llvm::parallel::par, asyncExecutors.begin(),
      std::next(asyncExecutors.begin(),
                std::min(asyncExecutors.size(), opAMPairs.size())),
      [&](MutableArrayRef<OpPassManager> pms) {
        for (auto e = opAMPairs.size(); !passFailed && opIt < e;) {
          // Get the next available operation index.
          unsigned nextID = opIt++;
          if (nextID >= e)
            break;

          // Set the order id for this thread in the diagnostic handler.
          diagHandler.setOrderIDForThread(nextID);

          // Get the pass manager for this operation and execute it.
          auto &it = opAMPairs[nextID];
          auto *pm = findPassManagerFor(pms, it.first->getName());
          assert(pm && "expected valid pass manager for operation");

          if (instrumentor)
            instrumentor->runBeforePipeline(pm->getOpName(), parentInfo);
          auto pipelineResult = runPipeline(*pm, it.first, it.second);
          if (instrumentor)
            instrumentor->runAfterPipeline(pm->getOpName(), parentInfo);

          // Drop this thread from being tracked by the diagnostic handler.
          // After this task has finished, the thread may be used outside of
          // this pass manager context meaning that we don't want to track
          // diagnostics from it anymore.
          diagHandler.eraseOrderIDForThread();

          // Handle a failed pipeline result.
          if (failed(pipelineResult)) {
            passFailed = true;
            break;
          }
        }
      });

  // Signal a failure if any of the executors failed.
  if (passFailed)
    signalPassFailure();
}

/// Utility function to convert the given class to the base adaptor it is an
/// adaptor pass, returns nullptr otherwise.
OpToOpPassAdaptorBase *mlir::detail::getAdaptorPassBase(Pass *pass) {
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptor>(pass))
    return adaptor;
  if (auto *adaptor = dyn_cast<OpToOpPassAdaptorParallel>(pass))
    return adaptor;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PassCrashReproducer
//===----------------------------------------------------------------------===//

/// Safely run the pass manager over the given module, creating a reproducible
/// on failure or crash.
static LogicalResult runWithCrashRecovery(OpPassManager &pm,
                                          ModuleAnalysisManager &am,
                                          ModuleOp module,
                                          StringRef crashReproducerFileName) {
  /// Enable crash recovery.
  llvm::CrashRecoveryContext::Enable();

  // Grab the textual pipeline executing within the pass manager first, just in
  // case the pass manager becomes compromised.
  std::string pipeline;
  {
    llvm::raw_string_ostream pipelineOS(pipeline);
    pm.printAsTextualPipeline(pipelineOS);
  }

  // Clone the initial module before running it through the pass pipeline.
  OwningModuleRef reproducerModule = module.clone();

  // Safely invoke the pass manager within a recovery context.
  LogicalResult passManagerResult = failure();
  llvm::CrashRecoveryContext recoveryContext;
  recoveryContext.RunSafelyOnThread(
      [&] { passManagerResult = pm.run(module, am); });

  /// Disable crash recovery.
  llvm::CrashRecoveryContext::Disable();
  if (succeeded(passManagerResult))
    return success();

  // The conversion failed, so generate a reproducible.
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(crashReproducerFileName, &error);
  if (!outputFile)
    return emitError(UnknownLoc::get(pm.getContext()),
                     "<MLIR-PassManager-Crash-Reproducer>: ")
           << error;
  auto &outputOS = outputFile->os();

  // Output the current pass manager configuration.
  outputOS << "// configuration: -pass-pipeline='" << pipeline << "'";
  if (pm.getImpl().disableThreads)
    outputOS << " -disable-pass-threading";

  // TODO(riverriddle) Should this also be configured with a pass manager flag?
  outputOS << "\n// note: verifyPasses="
           << (pm.getImpl().verifyPasses ? "true" : "false") << "\n";

  // Output the .mlir module.
  reproducerModule->print(outputOS);
  outputFile->keep();

  return reproducerModule->emitError()
         << "A crash has been detected while processing the MLIR module, a "
            "reproducer has been generated in '"
         << crashReproducerFileName << "'";
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::PassManager(MLIRContext *ctx, bool verifyPasses)
    : OpPassManager(OperationName(ModuleOp::getOperationName(), ctx),
                    /*disableThreads=*/false, verifyPasses),
      passTiming(false) {}

PassManager::~PassManager() {}

/// Run the passes within this manager on the provided module.
LogicalResult PassManager::run(ModuleOp module) {
  // Before running, make sure to coalesce any adjacent pass adaptors in the
  // pipeline.
  getImpl().coalesceAdjacentAdaptorPasses();

  // Construct an analysis manager for the pipeline.
  ModuleAnalysisManager am(module, instrumentor.get());

  // If reproducer generation is enabled, run the pass manager with crash
  // handling enabled.
  if (crashReproducerFileName)
    return runWithCrashRecovery(*this, am, module, *crashReproducerFileName);
  return OpPassManager::run(module, am);
}

/// Disable support for multi-threading within the pass manager.
void PassManager::disableMultithreading(bool disable) {
  getImpl().disableThreads = disable;
}

/// Enable support for the pass manager to generate a reproducer on the event
/// of a crash or a pass failure. `outputFile` is a .mlir filename used to write
/// the generated reproducer.
void PassManager::enableCrashReproducerGeneration(StringRef outputFile) {
  crashReproducerFileName = outputFile;
}

/// Add the provided instrumentation to the pass manager.
void PassManager::addInstrumentation(std::unique_ptr<PassInstrumentation> pi) {
  if (!instrumentor)
    instrumentor = std::make_unique<PassInstrumentor>();

  instrumentor->addInstrumentation(std::move(pi));
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

/// See PassInstrumentation::runBeforePipeline for details.
void PassInstrumentor::runBeforePipeline(
    const OperationName &name,
    const PassInstrumentation::PipelineParentInfo &parentInfo) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : impl->instrumentations)
    instr->runBeforePipeline(name, parentInfo);
}

/// See PassInstrumentation::runAfterPipeline for details.
void PassInstrumentor::runAfterPipeline(
    const OperationName &name,
    const PassInstrumentation::PipelineParentInfo &parentInfo) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  for (auto &instr : llvm::reverse(impl->instrumentations))
    instr->runAfterPipeline(name, parentInfo);
}

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

/// Add the given instrumentation to the collection.
void PassInstrumentor::addInstrumentation(
    std::unique_ptr<PassInstrumentation> pi) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(impl->mutex);
  impl->instrumentations.emplace_back(std::move(pi));
}

constexpr AnalysisID mlir::detail::PreservedAnalyses::allAnalysesID;
