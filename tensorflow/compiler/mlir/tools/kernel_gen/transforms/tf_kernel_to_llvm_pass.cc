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

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

class TFKernelToLLVMPass : public TFKernelToLLVMPassBase<TFKernelToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter type_converter(ctx);
    type_converter.addConversion([&](tf_framework::OpKernelContextType type) {
      return LLVM::LLVMType::getInt8PtrTy(ctx);
    });

    // Populate patterns.
    OwningRewritePatternList patterns;

    populateStdExpandOpsPatterns(ctx, patterns);
    populateStdToLLVMConversionPatterns(type_converter, patterns);
    tf_framework::PopulateTFFrameworkToLLVMConversionPatterns(&type_converter,
                                                              &patterns);
    populateGpuToLLVMConversionPatterns(type_converter, patterns, "gpu.binary");

    // Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<gpu::GPUDialect, StandardOpsDialect,
                             tf_framework::TFFrameworkDialect>();
    target.addIllegalOp<LLVM::DialectCastOp>();

    if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateTFKernelToLLVMPass() {
  return std::make_unique<TFKernelToLLVMPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
