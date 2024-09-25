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
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace ifrt {

void CreateIfrtToOutlinedAtomProgramsPipeline(
    mlir::OpPassManager& pm,
    const IfrtToOutlinedAtomProgramsPipelineOptions& options) {
  // Passes that verify the correctness of the module.
  pm.addPass(xla::ifrt::CreateSpmdExpandableInterfaceVerificationPass(
      {{mlir::mhlo::MhloDialect::getDialectNamespace().str(),
        mlir::stablehlo::StablehloDialect::getDialectNamespace().str()}}));
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::ifrt::CreateIfrtVerifyDonationPass());

  // Passes that outline atom programs to modules and set their metadata.
  pm.addPass(xla::ifrt::CreateIfrtOutlineAtomProgramToModulePass());
  pm.addPass(xla::ifrt::CreateIfrtPopulateAtomProgramMetadataPass());
  pm.addPass(xla::ifrt::CreateIfrtDuplicatedCalleeEliminationPass());
  pm.addPass(mlir::createSymbolDCEPass());

  if (!options.propagate_shardings) {
    pm.addPass(xla::ifrt::CreateIfrtVerifyShardingSpecifiedPass());
    // We can split ifrt.Reshard to ifrt.CopyArrays because all the shardings
    // are specified.
    pm.addPass(xla::ifrt::CreateIfrtReshardToCopyArraysPass());
  }
}

void RegisterIfrtPassesAndPipelines() {
  registerIfrtIrPasses();
  mlir::PassPipelineRegistration<IfrtToOutlinedAtomProgramsPipelineOptions>(
      "ifrt-to-outlined-atom-programs-pipeline",
      "Runs passes that do not require compilation-time information",
      CreateIfrtToOutlinedAtomProgramsPipeline);
}

}  // namespace ifrt
}  // namespace xla
