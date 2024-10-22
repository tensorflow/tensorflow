/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DECL
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSpmdExpandableInterfaceVerificationPass(
    SpmdExpandableInterfaceVerificationPassOptions options = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSpmdExpansionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtDuplicatedCalleeEliminationPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtMergeReshardsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtOutlineAtomProgramToModulePass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyDonationPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyShardingSpecifiedPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPopulateAtomProgramMetadataPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtReshardToCopyArraysPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtLowerShardingToXlaPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtRemoveIfrtAttrsPass();

// Generated definitions. This should be placed after all Pass creations.
#define GEN_PASS_REGISTRATION
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

struct IfrtToOutlinedAtomProgramsPipelineOptions
    : mlir::PassPipelineOptions<IfrtToOutlinedAtomProgramsPipelineOptions> {
  Option<bool> propagate_shardings{
      *this, "propagate_shardings",
      llvm::cl::desc("Whether to propagate shardings from executables for "
                     "unspecified shardings.")};
};

// Creates pipeline of all the IFRT IR passes that do not require
// compilation-time information (e.g., device assignments).
void CreateIfrtToOutlinedAtomProgramsPipeline(
    mlir::OpPassManager& pm,
    const IfrtToOutlinedAtomProgramsPipelineOptions& options);

// Creates pipeline to lower an IFRT XLA program to be ready for compilation.
void CreateIfrtCompileXlaPreprocessingPipeline(mlir::OpPassManager& pm);

// Registers passes and pipelines to ifrt-opt.
void RegisterIfrtPassesAndPipelines();

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
