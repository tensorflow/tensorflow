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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace lmhlo {
namespace {

class TestLhloToLLVMPass
    : public ::mlir::PassWrapper<TestLhloToLLVMPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    PopulateLhloToLLVMConversionPatterns(
        LowerToLLVMOptions::getDefaultOptions(), &converter, &patterns);
    mlir::populateLoopToStdConversionPatterns(patterns, &getContext());

    mlir::populateAffineToStdConversionPatterns(patterns, m.getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addIllegalDialect<LmhloDialect>();

    if (failed(applyFullConversion(m, target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<TestLhloToLLVMPass> legalize_lhlo_pass(
    "test-lhlo-legalize-to-llvm", "Legalize from LHLO dialect to LLVM.");

}  // namespace lmhlo
}  // namespace mlir
