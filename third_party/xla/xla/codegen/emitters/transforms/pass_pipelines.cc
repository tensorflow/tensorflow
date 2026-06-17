/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/codegen/emitters/transforms/pass_pipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla::emitters {

void RegisterOptimizationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<mlir::func::FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createCSEPass());
}

}  // namespace xla::emitters
