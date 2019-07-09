//===- PassDetail.h - MLIR Pass details -------------------------*- C++ -*-===//
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
#ifndef MLIR_PASS_PASSDETAIL_H_
#define MLIR_PASS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace detail {

//===----------------------------------------------------------------------===//
// PassExecutor
//===----------------------------------------------------------------------===//

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
  FunctionPassExecutor(const FunctionPassExecutor &rhs);

  /// Run the executor on the given function.
  LogicalResult run(FuncOp function, FunctionAnalysisManager &fam);

  /// Add a pass to the current executor. This takes ownership over the provided
  /// pass pointer.
  void addPass(FunctionPassBase *pass) { passes.emplace_back(pass); }

  /// Returns the number of passes held by this executor.
  size_t size() const { return passes.size(); }

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
  LogicalResult run(Module module, ModuleAnalysisManager &mam);

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

//===----------------------------------------------------------------------===//
// ModuleToFunctionPassAdaptor
//===----------------------------------------------------------------------===//

/// An adaptor module pass used to run function passes over all of the
/// non-external functions of a module synchronously on a single thread.
class ModuleToFunctionPassAdaptor
    : public ModulePass<ModuleToFunctionPassAdaptor> {
public:
  /// Run the held function pipeline over all non-external functions within the
  /// module.
  void runOnModule() override;

  /// Returns the function pass executor for this adaptor.
  FunctionPassExecutor &getFunctionExecutor() { return fpe; }

private:
  FunctionPassExecutor fpe;
};

/// An adaptor module pass used to run function passes over all of the
/// non-external functions of a module asynchronously across multiple threads.
class ModuleToFunctionPassAdaptorParallel
    : public ModulePass<ModuleToFunctionPassAdaptorParallel> {
public:
  /// Run the held function pipeline over all non-external functions within the
  /// module.
  void runOnModule() override;

  /// Returns the function pass executor for this adaptor.
  FunctionPassExecutor &getFunctionExecutor() { return fpe; }

private:
  // The main function pass executor for this adaptor.
  FunctionPassExecutor fpe;

  // A set of executors, cloned from the main executor, that run asynchronously
  // on different threads.
  std::vector<FunctionPassExecutor> asyncExecutors;
};

/// Utility function to return if a pass refers to an
/// ModuleToFunctionPassAdaptor instance.
inline bool isModuleToFunctionAdaptorPass(Pass *pass) {
  return isa<ModuleToFunctionPassAdaptorParallel>(pass) ||
         isa<ModuleToFunctionPassAdaptor>(pass);
}

/// Utility function to return if a pass refers to an adaptor pass. Adaptor
/// passes are those that internally execute a pipeline, such as the
/// ModuleToFunctionPassAdaptor.
inline bool isAdaptorPass(Pass *pass) {
  return isModuleToFunctionAdaptorPass(pass);
}

} // end namespace detail
} // end namespace mlir
#endif // MLIR_PASS_PASSDETAIL_H_
