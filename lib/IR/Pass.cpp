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
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Pass.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"

using namespace mlir;

/// Function passes walk a module and look at each function with their
/// corresponding hooks.
void FunctionPass::runOnModule(Module *m) {
  for (auto &fn : *m) {
    if (auto *mlFunc = dyn_cast<MLFunction>(&fn))
      runOnMLFunction(mlFunc);
    if (auto *cfgFunc = dyn_cast<CFGFunction>(&fn))
      runOnCFGFunction(cfgFunc);
  }
}
