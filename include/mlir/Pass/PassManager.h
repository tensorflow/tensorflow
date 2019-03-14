//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
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

#ifndef MLIR_PASS_PASSMANAGER_H
#define MLIR_PASS_PASSMANAGER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class Any;
} // end namespace llvm

namespace mlir {
class FunctionPassBase;
class Module;
class ModulePassBase;
class Pass;
class PassInstrumentation;
class PassInstrumentor;

namespace detail {
class PassExecutor;
class ModulePassExecutor;
} // end namespace detail

/// An enum describing the different display modes for the pass timing
/// information within the pass manager.
enum class PassTimingDisplayMode {
  // In this mode the results are displayed in a list sorted by total time,
  // with each pass/analysis instance aggregated into one unique result.
  List,

  // In this mode the results are displayed in a nested pipeline view that
  // mirrors the internal pass pipeline that is being executed in the pass
  // manager.
  Pipeline,
};

/// The main pass manager and pipeline builder.
class PassManager {
public:
  // If verifyPasses is true, the verifier is run after each pass.
  PassManager(bool verifyPasses = true);
  ~PassManager();

  /// Run the passes within this manager on the provided module.
  LLVM_NODISCARD
  LogicalResult run(Module *module);

  //===--------------------------------------------------------------------===//
  // Pipeline Building
  //===--------------------------------------------------------------------===//

  /// Add an opaque pass pointer to the current manager. This takes ownership
  /// over the provided pass pointer.
  void addPass(Pass *pass);

  /// Add a module pass to the current manager. This takes ownership over the
  /// provided pass pointer.
  void addPass(ModulePassBase *pass);

  /// Add a function pass to the current manager. This takes ownership over the
  /// provided pass pointer. This will automatically create a function pass
  /// executor if necessary.
  void addPass(FunctionPassBase *pass);

  //===--------------------------------------------------------------------===//
  // Instrumentations
  //===--------------------------------------------------------------------===//

  /// Add the provided instrumentation to the pass manager. This takes ownership
  /// over the given pointer.
  void addInstrumentation(PassInstrumentation *pi);

  /// Add an instrumentation to print the IR before and after pass execution.
  /// * 'shouldPrintBeforePass' and 'shouldPrintAfterPass' correspond to filter
  ///   functions that take a 'Pass *'. These function should return true if the
  ///   IR should be printed or not.
  /// * 'printModuleScope' signals if the module IR should be printed, even for
  ///   non module passes.
  /// * 'out' corresponds to the stream to output the printed IR to.
  void enableIRPrinting(std::function<bool(Pass *)> shouldPrintBeforePass,
                        std::function<bool(Pass *)> shouldPrintAfterPass,
                        bool printModuleScope, raw_ostream &out);

  /// Add an instrumentation to time the execution of passes and the computation
  /// of analyses.
  /// Note: Timing should be enabled after all other instrumentations to avoid
  /// any potential "ghost" timing from other instrumentations being
  /// unintentionally included in the timing results.
  void enableTiming(
      PassTimingDisplayMode displayMode = PassTimingDisplayMode::Pipeline);

private:
  /// A stack of nested pass executors on sub-module IR units, e.g. function.
  llvm::SmallVector<detail::PassExecutor *, 1> nestedExecutorStack;

  /// The top level module pass executor.
  std::unique_ptr<detail::ModulePassExecutor> mpe;

  /// Flag that specifies if the IR should be verified after each pass has run.
  bool verifyPasses : 1;

  /// Flag that specifies if pass timing is enabled.
  bool passTiming : 1;

  /// A manager for pass instrumentations.
  std::unique_ptr<PassInstrumentor> instrumentor;
};

} // end namespace mlir

#endif // MLIR_PASS_PASSMANAGER_H
