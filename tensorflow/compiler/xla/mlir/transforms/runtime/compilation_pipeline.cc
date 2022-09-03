/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/compilation_pipeline.h"

#include <utility>

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"

namespace xla {
namespace runtime {

void RegisterDefaultXlaRuntimeDialects(mlir::DialectRegistry& registry) {
  // Register MLIR dialects supported by the compiled executables.
  registry.insert<mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::func::FuncDialect, RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerLLVMDialectTranslation(registry);
}

void CreateDefaultXlaRuntimeCompilationPipeline(
    mlir::OpPassManager& pm, const CompilationPipelineOptions& opts) {
  pm.addPass(mlir::createConvertSCFToCFPass());

  // Convert entry function to the XLA entrypoint.
  pm.addPass(CreateConvertToEntrypoint());

  // Convert runtime operations and custom calls to LLVM dialect.
  ConvertRuntimeToLLvmOpts rt_to_llvm_opts = {
      opts.populate_type_id_names, opts.populate_type_conversions,
      opts.populate_arg_encodings, opts.populate_attr_encodings};
  pm.addPass(CreateConvertRuntimeToLLVMPass(std::move(rt_to_llvm_opts)));

  // Convert everythinG else to LLVM dialect.
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

static void CreateDefaultPipeline(mlir::OpPassManager& pm) {
  CompilationPipelineOptions copts;
  CreateDefaultXlaRuntimeCompilationPipeline(pm, copts);
}

static mlir::PassPipelineRegistration<> kXlaRuntimePipeline(
    "xla-runtime-default-pipeline", "Default XLA runtime compilation pipeline",
    CreateDefaultPipeline);

}  // namespace runtime
}  // namespace xla
