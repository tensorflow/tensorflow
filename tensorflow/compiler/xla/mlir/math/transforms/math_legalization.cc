/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MathToLibm/MathToLibm.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h"

namespace xla {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_MATHLEGALIZATIONPASS
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h.inc"

struct MathLegalizationPass
    : public impl::MathLegalizationPassBase<MathLegalizationPass> {
  explicit MathLegalizationPass(bool enable_approximations) {
    enable_approximations_ = enable_approximations;
  }
  void runOnOperation() override;
};

void MathLegalizationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());

  populateMathToLLVMConversionPatterns(converter, patterns);
  int32_t libm_log1p_benefit = enable_approximations_ ? 0 : 2;
  // MathToLibm patterns are a last resort, so they have a 0 benefit (except
  // for log1p if approximations are disabled, because it has accuracy issues
  // near 0 if implemented naively).
  populateMathToLibmConversionPatterns(patterns, 0, {libm_log1p_benefit});

  ConversionTarget target(getContext());
  target.addIllegalDialect<math::MathDialect>();
  target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                         func::FuncDialect, vector::VectorDialect>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateMathLegalizationPass(
    bool enable_approximations) {
  return std::make_unique<MathLegalizationPass>(enable_approximations);
}

}  // namespace xla
