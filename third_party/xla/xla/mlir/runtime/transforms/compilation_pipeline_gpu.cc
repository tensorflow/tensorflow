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

#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"

#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Async/IR/Async.h"  // from @llvm-project
#include "mlir/Dialect/Async/Passes.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_dialect.h"
#include "xla/mlir/runtime/ir/tests/testlib.h"
#include "xla/mlir/runtime/transforms/compiler.h"
#include "xla/mlir/runtime/transforms/passes.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"

namespace xla {
namespace runtime {

void RegisterDefaultXlaGpuRuntimeDialects(DialectRegistry& dialects) {
  // Register MLIR dialects supported by the compiled executables.
  dialects->insert<mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                   mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                   mlir::lmhlo_gpu::LmhloGpuDialect, mlir::lmhlo::LmhloDialect,
                   mlir::mhlo::MhloDialect, mlir::async::AsyncDialect,
                   mlir::arith::ArithDialect, RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerBuiltinDialectTranslation(*dialects);
  mlir::registerLLVMDialectTranslation(*dialects);
}

void RegisterLmhloGpuDialect(DialectRegistry& dialects) {
  dialects->insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
}

void RegisterTestlibDialect(DialectRegistry& dialects) {
  dialects->insert<TestlibDialect>();
}

static void CreateDefaultXlaGpuRuntimeCompilationPipeline(
    mlir::OpPassManager& pm, const CompilationPipelineOptions& opts,
    bool add_async_passes) {
  pm.addPass(mlir::createConvertSCFToCFPass());

  if (add_async_passes) pm.addPass(mlir::createAsyncFuncToAsyncRuntimePass());

  // Export functions to the XLA runtime.
  pm.addPass(CreateExportRuntimeFunctionsPass());
  pm.addPass(CreateAddInitializationsPass());
  pm.addPass(CreateConvertCustomCallsPass());
  pm.addPass(CreateConvertAssertsPass());

  if (add_async_passes) {
    // Lower from high level async operations to async runtime.
    pm.addPass(mlir::createAsyncToAsyncRuntimePass());

    // Add async.runtime reference counting operations.
    pm.addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());
  }

  // Prepare memrefs for lowering to LLVM.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::memref::createExpandStridedMetadataPass());

  // Convert runtime operations and custom calls to LLVM dialect.
  ConvertRuntimeToLLvmOpts rt_to_llvm_opts = {
      opts.populate_type_id_names, opts.populate_type_conversions,
      opts.populate_arg_encodings, opts.populate_ret_encodings,
      opts.populate_attr_encodings};
  pm.addPass(CreateConvertRuntimeToLLVMPass(std::move(rt_to_llvm_opts)));

  // Convert async dialect to LLVM once everything else is in the LLVM dialect.
  if (add_async_passes) pm.addPass(mlir::createConvertAsyncToLLVMPass());

  // Convert everything else to LLVM dialect.
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Clean up IR before passing it to LLVM.
  pm.addPass(mlir::createCSEPass());
}

void CreateDefaultXlaGpuRuntimeCompilationPipeline(
    PassManager& passes, const CompilationPipelineOptions& opts,
    bool add_async_passes) {
  CreateDefaultXlaGpuRuntimeCompilationPipeline(*passes, opts,
                                                add_async_passes);
}

void AppendXlaGpuDialectRegistry(mlir::MLIRContext& context) {
  DialectRegistry dialects;
  RegisterDefaultXlaGpuRuntimeDialects(dialects);
  context.appendDialectRegistry(*dialects);
}

static void CreateDefaultGpuPipeline(mlir::OpPassManager& pm) {
  CompilationPipelineOptions copts;
  CreateDefaultXlaGpuRuntimeCompilationPipeline(pm, copts, false);
}

static mlir::PassPipelineRegistration<> kXlaRuntimePipeline(
    "xla-runtime-default-gpu-pipeline",
    "Default XLA-GPU runtime compilation pipeline", CreateDefaultGpuPipeline);
}  // namespace runtime
}  // namespace xla
