/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "transforms/passes.h"

namespace mlir {
namespace hlo {
namespace {

#define GEN_PASS_DEF_HLOMATHLEGALIZATIONPASS
#include "transforms/passes.h.inc"

constexpr bool kEnableApproximations = true;

struct HloMathLegalizationPass
    : public impl::HloMathLegalizationPassBase<HloMathLegalizationPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::vector::VectorDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    LLVMTypeConverter converter(ctx);

    populateMathToLLVMConversionPatterns(converter, patterns);
    int32_t libmLog1pBenefit = kEnableApproximations ? 0 : 2;
    // MathToLibm patterns are a last resort, so they have a 0 benefit (except
    // for log1p if approximations are disabled, because it has accuracy issues
    // near 0 if implemented naively).
    populateMathToLibmConversionPatterns(patterns, 0, {libmLog1pBenefit});

    ConversionTarget target(getContext());
    target.addIllegalDialect<math::MathDialect>();
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                           func::FuncDialect, vector::VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createMathLegalizationPass() {
  return std::make_unique<HloMathLegalizationPass>();
}

}  // namespace hlo
}  // namespace mlir
