/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/transforms/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace xla {
namespace ifrt {

void createIfrtToOutlinedAtomProgramsPipeline(mlir::OpPassManager& pm) {
  // Passes that verify the correctness of the module.
  pm.addNestedPass<mlir::func::FuncOp>(createIfrtVerifyDonationPass());

  pm.addPass(createIfrtOutlineAtomProgramToModulePass());
  // DCE the sdy.mesh ops at the program-level.
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::func::FuncOp>(createIfrtVerifyShardingSpecifiedPass());
  // We can split ifrt.Reshard to ifrt.CopyArrays because all the shardings
  // are specified. Moreover, it is necessary to run
  // IfrtMergeCopiesAndReshardsPass after this pass because it introduces
  // non-merged CopyArrays ops.
  pm.addPass(createIfrtReshardToCopyArraysPass());
  // Insert CopyArrays for arrays that are returned multiple times so that the
  // donation/reuse semantics are correctly handled. This pass must run before
  // InsertMergeCopiesAndReshardsPass so that the newly inserted CopyArrays ops
  // can be merged.
  pm.addNestedPass<mlir::func::FuncOp>(
      createIfrtInsertCopyArraysForReturnedManyTimesPass());
  // IfrtMergeCopiesAndReshardsPass doesn't handle control dependencies, so we
  // need to run it before adding the control dependencies.
  pm.addNestedPass<mlir::func::FuncOp>(createIfrtMergeCopiesAndReshardsPass());
  pm.addNestedPass<mlir::func::FuncOp>(createIfrtAddCtrlDependenciesPass());
}

void createIfrtPopulateAtomProgramMetadataPipeline(mlir::OpPassManager& pm) {
  pm.addPass(createIfrtPopulateAtomProgramMetadataPass());
  pm.addPass(createIfrtDuplicatedCalleeEliminationPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void registerIfrtPassesAndPipelines() {
  registerIfrtIrPasses();
  mlir::PassPipelineRegistration<>(
      "ifrt-to-outlined-atom-programs-pipeline",
      "Runs passes that do not require compilation-time information",
      createIfrtToOutlinedAtomProgramsPipeline);
  mlir::PassPipelineRegistration<>(
      "ifrt-populate-atom-program-metadata-pipeline",
      "Run passes to populate atom program metadata with IFRT info",
      createIfrtPopulateAtomProgramMetadataPipeline);
}

}  // namespace ifrt
}  // namespace xla
