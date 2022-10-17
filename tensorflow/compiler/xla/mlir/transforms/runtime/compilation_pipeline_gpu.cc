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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/compilation_pipeline_gpu.h"

#include <utility>

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/ir/runtime/tests/testlib.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/compiler.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"

namespace xla {
namespace runtime {

void RegisterDefaultXlaGpuRuntimeDialects(DialectRegistry& dialects) {
  // Register MLIR dialects supported by the compiled executables.
  dialects->insert<mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                   mlir::func::FuncDialect, RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerLLVMDialectTranslation(*dialects);
}

void RegisterLmhloGpuDialect(DialectRegistry& dialects) {
  dialects->insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
}

void RegisterTestlibDialect(DialectRegistry& dialects) {
  dialects->insert<TestlibDialect>();
}

void CreateDefaultXlaGpuRuntimeCompilationPipeline(
    PassManager& passes, const CompilationPipelineOptions& opts) {
  passes->addPass(mlir::createConvertSCFToCFPass());

  // Export functions to the XLA runtime.
  passes->addPass(CreateExportRuntimeFunctionsPass());
  passes->addPass(CreateConvertCustomCallsPass());
  passes->addPass(CreateConvertAssertsPass());

  // Convert runtime operations and custom calls to LLVM dialect.
  ConvertRuntimeToLLvmOpts rt_to_llvm_opts = {
      opts.populate_type_id_names, opts.populate_type_conversions,
      opts.populate_arg_encodings, opts.populate_ret_encodings,
      opts.populate_attr_encodings};
  passes->addPass(CreateConvertRuntimeToLLVMPass(std::move(rt_to_llvm_opts)));

  // Convert everythinG else to LLVM dialect.
  passes->addPass(mlir::createMemRefToLLVMConversionPass());
  passes->addPass(mlir::createConvertFuncToLLVMPass());
  passes->addPass(mlir::createReconcileUnrealizedCastsPass());
}

void AppendXlaGpuDialectRegistry(mlir::MLIRContext& context) {
  DialectRegistry dialects;
  RegisterDefaultXlaGpuRuntimeDialects(dialects);
  context.appendDialectRegistry(*dialects);
}

static void CreateDefaultGpuPipeline(mlir::OpPassManager& pm) {
  CompilationPipelineOptions copts;
  PassManager passes(&pm);
  CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
}

static mlir::PassPipelineRegistration<> kXlaRuntimePipeline(
    "xla-runtime-default-gpu-pipeline",
    "Default XLA-GPU runtime compilation pipeline", CreateDefaultGpuPipeline);
}  // namespace runtime
}  // namespace xla
