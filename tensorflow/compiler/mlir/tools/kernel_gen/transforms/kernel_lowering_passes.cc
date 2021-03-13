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

#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {

using gpu::GPUModuleOp;

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

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

    OwningRewritePatternList patterns;
    mlir::LowerToLLVMOptions llvm_opts;
    llvm_opts.indexBitwidth = 32;
    LLVMTypeConverter converter(m.getContext(), llvm_opts);
    populateStdToLLVMConversionPatterns(converter, patterns);
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
    : public GpuKernelToNVVMPassBase<GpuKernelToROCDLPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::ROCDL::ROCDLDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToROCDLConversionPatterns(converter, patterns);
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

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
