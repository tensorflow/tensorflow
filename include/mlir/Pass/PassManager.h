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

#include "llvm/ADT/SmallVector.h"

namespace mlir {
class FunctionPassBase;
class Module;
class ModulePassBase;
class Pass;

namespace detail {
class PassExecutor;
class ModulePassExecutor;
} // end namespace detail

/// The main pass manager and pipeline builder.
class PassManager {
public:
  // If verifyPasses is true, the verifier is run after each pass.
  PassManager(bool verifyPasses = true);
  ~PassManager();

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

  /// Run the passes within this manager on the provided module. Returns false
  /// if the run failed, true otherwise.
  LLVM_NODISCARD
  bool run(Module *module);

private:
  /// A stack of nested pass executors on sub-module IR units, e.g. function.
  llvm::SmallVector<detail::PassExecutor *, 1> nestedExecutorStack;

  /// The top level module pass executor.
  std::unique_ptr<detail::ModulePassExecutor> mpe;

  /// Flag that specifies if the IR should be verified after each pass has run.
  bool verifyPasses;
};

} // end namespace mlir

#endif // MLIR_PASS_PASSMANAGER_H
