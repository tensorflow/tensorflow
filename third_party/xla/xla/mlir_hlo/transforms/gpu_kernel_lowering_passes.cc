/* Copyright 2020 The OpenXLA Authors.

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

#include <cassert>
#include <memory>
#include <utility>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
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
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transforms/gpu_passes.h"

namespace mlir {

#define GEN_PASS_DEF_GPUKERNELTONVVMPASS
#define GEN_PASS_DEF_GPUKERNELTOROCDLPASS
#include "transforms/gpu_passes.h.inc"

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToNVVMPass
    : public impl::GpuKernelToNVVMPassBase<GpuKernelToNVVMPass> {
 public:
  explicit GpuKernelToNVVMPass(bool useBarePtrCallConv) {
    this->useBarePtrCallConv = useBarePtrCallConv;
  }
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
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorContractLowering());
  vector::populateVectorMaskOpLoweringPatterns(patterns);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransposeLowering());
  // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
}

static void populateCommonPatterns(LLVMTypeConverter& converter,
                                   RewritePatternSet& patterns) {
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateComplexToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
}

void GpuKernelToNVVMPass::runOnOperation() {
  {
    RewritePatternSet patterns(&getContext());
    populateAllCommonVectorProgressiveLoweringPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions llvmOpts(&getContext(), DataLayout(getOperation()));
  llvmOpts.useBarePtrCallConv = useBarePtrCallConv;
  LLVMTypeConverter converter(&getContext(), llvmOpts);

  populateCommonPatterns(converter, patterns);
  populateGpuToNVVMConversionPatterns(converter, patterns);

  populateGpuMemorySpaceAttributeConversions(
      converter, [](gpu::AddressSpace space) {
        switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
        }
        assert(false && "unknown address space enum value");
        return 0;
      });

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

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToNvvmPass(
    bool useBarePtrCallConv) {
  return std::make_unique<GpuKernelToNVVMPass>(useBarePtrCallConv);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuKernelToRocdlPass() {
  return std::make_unique<GpuKernelToROCDLPass>();
}

}  // namespace mlir
