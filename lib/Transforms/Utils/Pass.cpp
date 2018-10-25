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

#include "mlir/Transforms/Pass.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
using namespace mlir;

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void ModulePass::anchor() {}

/// Function passes walk a module and look at each function with their
/// corresponding hooks and terminates upon error encountered.
PassResult FunctionPass::runOnModule(Module *m) {
  for (auto &fn : *m) {
    if (runOnFunction(&fn))
      return failure();
  }
  return success();
}

PassResult FunctionPass::runOnFunction(Function *fn) {
  if (auto *mlFunc = dyn_cast<MLFunction>(fn))
    return runOnMLFunction(mlFunc);
  if (auto *cfgFunc = dyn_cast<CFGFunction>(fn))
    return runOnCFGFunction(cfgFunc);

  return success();
}
