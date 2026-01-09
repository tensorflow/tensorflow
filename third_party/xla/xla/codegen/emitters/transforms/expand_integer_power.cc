/* Copyright 2025 The OpenXLA Authors.

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

#include "mhlo/IR/hlo_ops.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"

namespace xla::emitters {

#define GEN_PASS_DEF_EXPANDINTEGERPOWERPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

mlir::LogicalResult ExpandIntegerPower(mlir::math::IPowIOp op,
                                       mlir::PatternRewriter& rewriter) {
  llvm::SmallVector<mlir::Type> result_types(op->getResultTypes());
  llvm::SmallVector<mlir::Type> arg_types(op->getOperandTypes());
  mlir::Value result =
      mlir::mhlo::impl::mapMhloOpToStdScalarOp<mlir::mhlo::PowOp>(
          op.getLoc(), result_types, arg_types, {op->getOperands()},
          op->getAttrs(), &rewriter);

  rewriter.replaceOp(op, result);
  return mlir::success();
}

class ExpandIntegerPowerPass
    : public impl::ExpandIntegerPowerPassBase<ExpandIntegerPowerPass> {
 public:
  using ExpandIntegerPowerPassBase::ExpandIntegerPowerPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.add(ExpandIntegerPower);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandIntegerPowerPass() {
  return std::make_unique<ExpandIntegerPowerPass>();
}

}  // namespace xla::emitters
