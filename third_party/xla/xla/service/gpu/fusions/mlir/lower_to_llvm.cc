/* Copyright 2024 The OpenXLA Authors.

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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_LOWERTOLLVMPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

class LowerToLLVMPass : public impl::LowerToLLVMPassBase<LowerToLLVMPass> {
 public:
  using LowerToLLVMPassBase::LowerToLLVMPassBase;

  void runOnOperation() override {
    // Populate type conversions.
    mlir::LLVMTypeConverter type_converter(getOperation().getContext());
    mlir::LLVMConversionTarget target(*getOperation().getContext());

    // Populate patterns.
    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
    mlir::populateGpuToNVVMConversionPatterns(type_converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                          patterns);
    mlir::populateComplexToLLVMConversionPatterns(type_converter, patterns);
    mlir::populateMathToLLVMConversionPatterns(type_converter, patterns);

    //  Setup target.
    mlir::configureGpuToNVVMConversionLegality(target);
    target.addIllegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                             mlir::complex::ComplexDialect,
                             mlir::math::MathDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

}  // namespace gpu
}  // namespace xla
