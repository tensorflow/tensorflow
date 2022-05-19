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

#include "mlir-hlo/Transforms/GPUPassDetail.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

using gpu::GPUModuleOp;

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToNVVMPass
    : public GpuKernelToNVVMPassBase<GpuKernelToNVVMPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    GPUModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvm_opts(m.getContext(), DataLayout(m));

    LLVMTypeConverter converter(m.getContext(), llvm_opts);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateGpuToNVVMConversionPatterns(converter, patterns);
    populateComplexToLLVMConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

/// A pass that does the final lowering to ROCDL. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToROCDLPass
    : public GpuKernelToROCDLPassBase<GpuKernelToROCDLPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::ROCDL::ROCDLDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(m.getContext());
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateGpuToROCDLConversionPatterns(converter, patterns,
                                         gpu::amd::Runtime::Unknown);
    populateComplexToLLVMConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<GPUModuleOp> > CreateGpuKernelToNvvmPass() {
  return std::make_unique<GpuKernelToNVVMPass>();
}

std::unique_ptr<OperationPass<GPUModuleOp> > CreateGpuKernelToRocdlPass() {
  return std::make_unique<GpuKernelToROCDLPass>();
}

}  // namespace mlir
