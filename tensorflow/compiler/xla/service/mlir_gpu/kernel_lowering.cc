/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"

#include "absl/memory/memory.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"  // from @llvm-project
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"  // from @llvm-project
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/passes.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace mlir_gpu {

Status LowerLHLOToGPU(mlir::ModuleOp module, LowerLHLOToGPUOptions options) {
  mlir::PassManager pm(module.getContext());
  tensorflow::applyTensorflowAndCLOptions(pm);

  // We have to anticipate later unrolling in tiling to make sure that we get
  // the requested tiling after unrolling. Compute the new tiling here if
  // needed.
  llvm::SmallVector<unsigned, 4> tiling_for_unrolling;
  llvm::SmallVector<int64_t, 4> as_int64;
  if (!options.unroll_factors.empty()) {
    tiling_for_unrolling.reserve(options.tile_sizes.size());
    for (auto pair : llvm::zip(options.tile_sizes, options.unroll_factors)) {
      tiling_for_unrolling.push_back(std::get<0>(pair) * std::get<1>(pair));
      as_int64.push_back(std::get<1>(pair));
    }
  } else {
    tiling_for_unrolling.append(options.tile_sizes.begin(),
                                options.tile_sizes.end());
  }

  // Legalize from HLO to LHLO.
  pm.addPass(::mlir::mhlo::createLegalizeToLhloPass());
  // Moving `AllocOp`s and inserting missing `DeallocOp`s
  pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferDeallocationPass());
  // Next, we can strip the outer fusion operation.
  pm.addNestedPass<mlir::FuncOp>(createFusionOpRemoverPass());
  // Remove unnecessary LHLO copies.
  pm.addNestedPass<mlir::FuncOp>(::mlir::createCopyRemovalPass());
  // Legalize reduce operations directly to GPU dialect.
  pm.addNestedPass<mlir::FuncOp>(::mlir::lmhlo::createLegalizeToGpuPass());
  // Transform LHLO operations to LinAlg.
  pm.addNestedPass<mlir::FuncOp>(
      ::mlir::lmhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations.
  pm.addNestedPass<mlir::FuncOp>(::mlir::lmhlo::createLhloFuseLinalgPass(
      /*use_parallel_loops=*/true, tiling_for_unrolling));
  // Transform the Linalg operations inside of the loop nest into parallel
  // loops.
  pm.addNestedPass<mlir::FuncOp>(
      ::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Fuse the inner-most loops.
  pm.addNestedPass<mlir::FuncOp>(createFuseInnerParallelLoopsPass());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addNestedPass<mlir::FuncOp>(createStoreForwardingPass());
  // Remove now unused temporary buffers.
  pm.addNestedPass<mlir::FuncOp>(createDeadTempBufferRemovalPass());
  if (!options.unroll_factors.empty()) {
    pm.addNestedPass<mlir::FuncOp>(
        ::mlir::createParallelLoopTilingPass(as_int64));
  }
  // Project all loop dimensions to X if necessary.
  if (options.collapse_parallel_loops) {
    pm.addNestedPass<mlir::FuncOp>(
        createParallelLoopCollapsingToFirstDimPass());
  }
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Greedily map the remaining loop to GPU hardware dimensions.
  pm.addNestedPass<::mlir::FuncOp>(createMapParallelLoopsPass());
  // Apply the mapping.
  pm.addNestedPass<::mlir::FuncOp>(mlir::createParallelLoopToGpuPass());
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  // Only do this if we unrolled in the first place.
  if (!options.unroll_factors.empty()) {
    pm.addNestedPass<::mlir::FuncOp>(mlir::createForLoopSpecializationPass());
  }
  // Approximate of requested.
  if (options.use_approximations) {
    pm.addNestedPass<::mlir::FuncOp>(
        ::mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
  }
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());
  // Make sure the kernel signature resembled the original function's
  // signature
  if (options.rewrite_signature) {
    pm.addNestedPass<::mlir::FuncOp>(createRewriteKernelSignaturePass());
  }
  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class LowerToNVVMPass
    : public ::mlir::PassWrapper<
          LowerToNVVMPass, ::mlir::OperationPass<::mlir::gpu::GPUModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ::mlir::gpu::GPUModuleOp m = getOperation();

    ::mlir::OwningRewritePatternList patterns;
    ::mlir::LLVMTypeConverter converter(m.getContext());
    ::mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    // TODO(b/145824979) Remove linalg once sliceop is in std.
    ::mlir::populateLinalgToLLVMConversionPatterns(converter, patterns);
    ::mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
    ::mlir::populateAffineToStdConversionPatterns(patterns, m.getContext());
    ::mlir::ConversionTarget target(getContext());
    ::mlir::configureGpuToNVVMConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  ::mlir::PassManager pm(module.getContext());
  // We cannot verify as the signature of the kernel is rewritten.
  pm.enableVerifier(false);
  tensorflow::applyTensorflowAndCLOptions(pm);

  // Rewrite kernel functions to LLVM IR.
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createLowerToCFGPass());
  kernelPm.addPass(absl::make_unique<LowerToNVVMPass>());
  // Some basic cleanup.
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Remove all location information to prevent a debug build.
  pm.addPass(::mlir::createStripDebugInfoPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to NVVM IR failed.");
  }
  return Status::OK();
}

namespace {

/// A pass that does the final lowering to ROCDL. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class LowerToROCDLPass
    : public ::mlir::PassWrapper<
          LowerToROCDLPass, ::mlir::OperationPass<::mlir::gpu::GPUModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::ROCDL::ROCDLDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ::mlir::gpu::GPUModuleOp m = getOperation();

    {
      ::mlir::OwningRewritePatternList patterns;
      ::mlir::populateGpuRewritePatterns(m.getContext(), patterns);
      if (failed(mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    ::mlir::OwningRewritePatternList patterns;
    ::mlir::LLVMTypeConverter converter(m.getContext());
    ::mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    // TODO(b/145824979) Remove linalg once sliceop is in std.
    ::mlir::populateLinalgToLLVMConversionPatterns(converter, patterns);
    ::mlir::populateGpuToROCDLConversionPatterns(converter, patterns);
    ::mlir::populateAffineToStdConversionPatterns(patterns, m.getContext());

    ::mlir::ConversionTarget target(getContext());
    ::mlir::configureGpuToROCDLConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

Status LowerKernelBodiesToROCDL(mlir::ModuleOp module) {
  ::mlir::PassManager pm(module.getContext());
  // We cannot verify as the signature of the kernel is rewritten.
  pm.enableVerifier(false);
  tensorflow::applyTensorflowAndCLOptions(pm);

  auto enable_if_vlog_is_on = [](mlir::Pass*, mlir::Operation*) {
    return VLOG_IS_ON(1);
  };
  pm.enableIRPrinting(/*shouldPrintBeforePass=*/{},
                      /*shouldPrintAfterPass=*/enable_if_vlog_is_on,
                      /*printModuleScope=*/false,
                      /*printAfterOnlyOnChange=*/false,
                      /*out=*/llvm::dbgs());

  // Rewrite kernel functions to LLVM IR.
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createLowerToCFGPass());
  kernelPm.addPass(absl::make_unique<LowerToROCDLPass>());

  // Some basic cleanup.
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Remove all location information to prevent a debug build.
  kernelPm.addPass(::mlir::createStripDebugInfoPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to ROCDL IR failed.");
  }
  return Status::OK();
}

StatusOr<mlir::ModuleOp> ExtractKernelModule(mlir::ModuleOp module) {
  auto kernelModule = ::mlir::ModuleOp::create(module.getLoc());
  // TODO(b/137624192): This also needs to resolve naming conflicts.
  module.walk([&kernelModule](mlir::gpu::GPUModuleOp nestedModule) {
    for (auto& fn : nestedModule.body().front()) {
      kernelModule.push_back(fn.clone());
    }
  });
  return kernelModule;
}

}  // namespace mlir_gpu
}  // namespace xla
