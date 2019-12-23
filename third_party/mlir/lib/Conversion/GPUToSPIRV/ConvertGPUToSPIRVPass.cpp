//===- ConvertGPUToSPIRVPass.cpp - GPU to SPIR-V dialect lowering passes --===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spv.module operation
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts those functions
/// that have the "gpu.kernel" attribute, i.e. those functions that are
/// referenced in gpu::LaunchKernelOp operations. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUToSPIRVPass : public ModulePass<GPUToSPIRVPass> {
public:
  GPUToSPIRVPass() = default;
  GPUToSPIRVPass(const GPUToSPIRVPass &) {}
  GPUToSPIRVPass(ArrayRef<int64_t> workGroupSize) {
    this->workGroupSize = workGroupSize;
  }

  void runOnModule() override;

private:
  /// Command line option to specify the workgroup size.
  ListOption<int64_t> workGroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Workgroup Sizes in the SPIR-V module for x, followed by y, followed "
          "by z dimension of the dispatch (others will be ignored)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
} // namespace

void GPUToSPIRVPass::runOnModule() {
  auto context = &getContext();
  auto module = getModule();

  SmallVector<Operation *, 1> kernelModules;
  OpBuilder builder(context);
  module.walk([&builder, &kernelModules](ModuleOp moduleOp) {
    if (moduleOp.getAttrOfType<UnitAttr>(
            gpu::GPUDialect::getKernelModuleAttrName())) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    }
  });

  SPIRVTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  populateGPUToSPIRVPatterns(context, typeConverter, patterns, workGroupSize);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });

  if (failed(applyFullConversion(kernelModules, target, patterns,
                                 &typeConverter))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createConvertGPUToSPIRVPass(ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<GPUToSPIRVPass>(workGroupSize);
}

static PassRegistration<GPUToSPIRVPass>
    pass("convert-gpu-to-spirv", "Convert GPU dialect to SPIR-V dialect");
