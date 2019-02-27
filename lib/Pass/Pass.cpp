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

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Forwarding function to execute this pass.
PassResult FunctionPassBase::run(Function *fn) {
  /// Initialize the pass state.
  passState.emplace(fn);

  /// Invoke the virtual runOnFunction function.
  return runOnFunction();
}

/// Forwarding function to execute this pass.
PassResult ModulePassBase::run(Module *module) {
  /// Initialize the pass state.
  passState.emplace(module);

  /// Invoke the virtual runOnModule function.
  return runOnModule();
}

/// Run all of the passes in this manager over the current function.
bool detail::FunctionPassExecutor::run(Function *function) {
  for (auto &pass : passes) {
    /// Create an execution state for this pass.
    if (pass->run(function))
      return true;
    // TODO: This should be opt-out and handled separately.
    if (function->verify())
      return true;
  }
  return false;
}

/// Run all of the passes in this manager over the current module.
bool detail::ModulePassExecutor::run(Module *module) {
  for (auto &pass : passes) {
    if (pass->run(module))
      return true;
    // TODO: This should be opt-out and handled separately.
    if (module->verify())
      return true;
  }
  return false;
}

/// Execute the held function pass over all non-external functions within the
/// module.
PassResult detail::ModuleToFunctionPassAdaptor::runOnModule() {
  for (auto &func : getModule()) {
    // Skip external functions.
    if (func.isExternal())
      continue;

    // Run the held function pipeline over the current function.
    if (fpe.run(&func))
      return failure();
  }
  return success();
}
