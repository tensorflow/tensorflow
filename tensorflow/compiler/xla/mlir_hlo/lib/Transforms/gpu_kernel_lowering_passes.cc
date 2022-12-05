/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_GPUKERNELTONVVMPASS
#define GEN_PASS_DEF_GPUKERNELTOROCDLPASS
#include "mlir-hlo/Transforms/gpu_passes.h.inc"

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToNVVMPass
    : public impl::GpuKernelToNVVMPassBase<GpuKernelToNVVMPass> {
  void runOnOperation() override;
};

/// A pass that does the final lowering to ROCDL. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToROCDLPass
    : public impl::GpuKernelToROCDLPassBase<GpuKernelToROCDLPass> {
  void runOnOperation() override;
};

}  // namespace

static void populateAllCommonVectorProgressiveLoweringPatterns(
    RewritePatternSet& patterns) {
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(patterns);
  vector::populateVectorMaskOpLoweringPatterns(patterns);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns);
  // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
}

static void populateCommonPatterns(LLVMTypeConverter& converter,
                                   RewritePatternSet& patterns) {
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateComplexToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
}

void GpuKernelToNVVMPass::runOnOperation() {
  {
    RewritePatternSet patterns(&getContext());
    populateAllCommonVectorProgressiveLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions llvmOpts(&getContext(), DataLayout(getOperation()));
  LLVMTypeConverter converter(&getContext(), llvmOpts);
  populateCommonPatterns(converter, patterns);
  populateGpuToNVVMConversionPatterns(converter, patterns);
  ConversionTarget target(getContext());
  configureGpuToNVVMConversionLegality(target);
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void GpuKernelToROCDLPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  populateCommonPatterns(converter, patterns);
  populateGpuToROCDLConversionPatterns(converter, patterns,
                                       gpu::amd::Runtime::Unknown);
  ConversionTarget target(getContext());
  configureGpuToROCDLConversionLegality(target);
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToNvvmPass() {
  return std::make_unique<GpuKernelToNVVMPass>();
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToRocdlPass() {
  return std::make_unique<GpuKernelToROCDLPass>();
}

}  // namespace mlir
