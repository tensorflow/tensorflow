/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/runtime/transforms/compilation_pipeline_cpu.h"

#include <utility>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Async/IR/Async.h"  // from @llvm-project
#include "mlir/Dialect/Async/Passes.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#ifdef TF_LLVM_X86_AVAILABLE
#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"  // from @llvm-project
#endif
#if defined(TF_LLVM_AARCH64_AVAILABLE) || defined(TF_LLVM_AARCH32_AVAILABLE)
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"  // from @llvm-project
#ifdef TF_LLVM_AARCH64_AVAILABLE
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"  // from @llvm-project
#endif
#endif
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#ifdef TF_LLVM_X86_AVAILABLE
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"  // from @llvm-project
#endif
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/mlir/backends/cpu/transforms/passes.h"
#include "xla/mlir/memref/transforms/passes.h"
#include "xla/mlir/runtime/ir/rt_dialect.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_options.h"
#include "xla/mlir/runtime/transforms/compiler.h"
#include "xla/mlir/runtime/transforms/passes.h"
#include "xla/mlir_hlo/transforms/passes.h"
#include "tsl/platform/logging.h"

#ifdef EXPERIMENTAL_MLIR_GPU
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Transforms/Passes.h"  // from @llvm-project
#endif  // EXPERIMENTAL_MLIR_GPU

namespace xla {
namespace runtime {

void RegisterDefaultXlaCpuRuntimeDialects(DialectRegistry& dialects) {
  // Register MLIR dialects supported by the compiled executables.
  dialects->insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                   mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                   mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                   mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                   mlir::func::FuncDialect,
                   mlir::sparse_tensor::SparseTensorDialect,
                   mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                   RuntimeDialect>();

  mlir::func::registerAllExtensions(*dialects);

  // Register MLIR dialects that can be translated to LLVM IR.
#ifdef TF_LLVM_AARCH64_AVAILABLE
  mlir::registerArmSVEDialectTranslation(*dialects);
#endif
#if defined(TF_LLVM_AARCH64_AVAILABLE) || defined(TF_LLVM_AARCH32_AVAILABLE)
  mlir::registerArmNeonDialectTranslation(*dialects);
#endif
#ifdef TF_LLVM_X86_AVAILABLE
  mlir::registerAMXDialectTranslation(*dialects);
  mlir::registerX86VectorDialectTranslation(*dialects);
#endif
  mlir::registerBuiltinDialectTranslation(*dialects);
  mlir::registerLLVMDialectTranslation(*dialects);
}

static void CreateXlaCpuCompilationPipeline(mlir::OpPassManager& pm,
                                            const CpuPipelineOptions& opts) {
  pm.addPass(mlir::createAsyncFuncToAsyncRuntimePass());

  // Convert entry function to the XLA entrypoint.
  pm.addPass(CreateExportRuntimeFunctionsPass());
  pm.addPass(cpu::createConvertXlaCpuToCpuRuntimePass());
  pm.addPass(CreateConvertCustomCallsPass());
  pm.addPass(CreateConvertAssertsPass());

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Convert all linalg operations to parallel loops.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize generated scf.parallel operations to remove single iterations.
  pm.addPass(mlir::createCanonicalizerPass());

  // TODO(ecg,ezhulenev): add conversion of scf.parallel to async.

  // Lower from high level async operations to async runtime.
  pm.addPass(mlir::createAsyncToAsyncRuntimePass());

  // Move all memref.alloca to entry block for all functions.
  pm.addPass(CreateMoveAllocasToEntryBlockPass());

  // Add async.runtime reference counting operations.
  pm.addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());

  // Expand math operations into std/arith dialect operations.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Add alignment attribute to all memref allocations.
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::CreateAlignedAllocationsPass(opts.alignment));

  // Lower everything down to LLVM dialect.
  // Convert runtime operations and custom calls to LLVM dialect.
  const CompilationPipelineOptions& copts = opts.common_options;
  ConvertRuntimeToLLvmOpts rt_to_llvm_opts = {
      copts.populate_type_id_names, copts.populate_type_conversions,
      copts.populate_arg_encodings, copts.populate_ret_encodings,
      copts.populate_attr_encodings};
  pm.addPass(CreateConvertRuntimeToLLVMPass(std::move(rt_to_llvm_opts)));

  // Convert async to LLVM once everything else is in the LLVM dialect.
  pm.addPass(mlir::createConvertAsyncToLLVMPass());

  // Convert everything else to LLVM dialect.
  mlir::GenericHostToLLVMPassOptions llvm_options;
  llvm_options.enableAvx2 = opts.math_avx2;
  pm.addPass(mlir::hlo::createGenericHostToLLVMPass(llvm_options));
  const bool gpuCodegen = opts.xla_cpu_sparse_cuda_threads > 0;
#ifdef EXPERIMENTAL_MLIR_GPU
  if (gpuCodegen) {
#ifdef MLIR_GPU_TO_CUBIN_PASS_ENABLE
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(
        mlir::createGpuSerializeToCubinPass(opts.cuda_triplet, opts.cuda_arch,
                                            opts.cuda_features));
#endif
    pm.addPass(mlir::createGpuToLLVMConversionPass());
  }
#else   // EXPERIMENTAL_MLIR_GPU
  CHECK(!gpuCodegen)
      << "Experimental MLIR GPU code generation was not enabled at build time";
#endif  // EXPERIMENTAL_MLIR_GPU
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Prepare module for translation to LLVM.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void CreateDefaultXlaCpuRuntimeCompilationPipeline(
    PassManager& passes, const CpuPipelineOptions& opts) {
  CreateXlaCpuCompilationPipeline(*passes, opts);
}

static void CreateDefaultCpuPipeline(mlir::OpPassManager& pm) {
  CpuPipelineOptions opts;
  CreateXlaCpuCompilationPipeline(pm, opts);
}

static mlir::PassPipelineRegistration<> kXlaRuntimePipeline(
    "xla-runtime-default-cpu-pipeline",
    "Default XLA-CPU runtime compilation pipeline", CreateDefaultCpuPipeline);

}  // namespace runtime
}  // namespace xla
