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

#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep

namespace xla::xtile {

#define GEN_PASS_DEF_STABLEHLOLOWERTOARITHPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

template <typename StableHloOp, typename FloatArithOp, typename IntArithOp,
          typename UnsignedIntArithOp = IntArithOp>
class LowerStableHloOpToArith : public mlir::OpRewritePattern<StableHloOp> {
 public:
  using mlir::OpRewritePattern<StableHloOp>::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      StableHloOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = mlir::getElementTypeOrSelf(op.getResult().getType());
    if (result_type.isFloat()) {
      rewriter.replaceOpWithNewOp<FloatArithOp>(op, op.getOperands());
    } else {
      mlir::Operation* new_op = nullptr;
      bool should_guard_ub =
          mlir::isa<mlir::stablehlo::DivOp, mlir::stablehlo::RemOp>(op);

      if (result_type.isUnsignedInteger()) {
        llvm::SmallVector<mlir::Value> signless_operands;
        signless_operands.reserve(op.getOperands().size());
        mlir::Type operand_type = op.getOperands().front().getType();
        for (mlir::Value operand : op.getOperands()) {
          signless_operands.push_back(
              ::xla::xtile::UnsignedIntegerToSignlessInteger(rewriter,
                                                             operand));
        }
        new_op = UnsignedIntArithOp::create(rewriter, op.getLoc(),
                                            signless_operands.front().getType(),
                                            signless_operands);

        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            op, op.getResult().getType(), new_op->getResult(0));
      } else {
        new_op = rewriter.replaceOpWithNewOp<IntArithOp>(op, op.getOperands());
      }

      // Special case for division with zero.
      if (should_guard_ub) {
        new_op->setAttr("xla.guard_ub", rewriter.getUnitAttr());
      }
    }
    return mlir::success();
  }
};

struct StablehloLowerToArithPass
    : public impl::StablehloLowerToArithPassBase<StablehloLowerToArithPass> {
  using StablehloLowerToArithPassBase::StablehloLowerToArithPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<
        LowerStableHloOpToArith<mlir::stablehlo::AddOp, mlir::arith::AddFOp,
                                mlir::arith::AddIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::SubtractOp,
                                mlir::arith::SubFOp, mlir::arith::SubIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MulOp, mlir::arith::MulFOp,
                                mlir::arith::MulIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::AndOp, mlir::arith::AndIOp,
                                mlir::arith::AndIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::OrOp, mlir::arith::OrIOp,
                                mlir::arith::OrIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::XorOp, mlir::arith::XOrIOp,
                                mlir::arith::XOrIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::DivOp, mlir::arith::DivFOp,
                                mlir::arith::DivSIOp, mlir::arith::DivUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::RemOp, mlir::arith::RemFOp,
                                mlir::arith::RemSIOp, mlir::arith::RemUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MaxOp, mlir::arith::MaximumFOp,
                                mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MinOp, mlir::arith::MinimumFOp,
                                mlir::arith::MinSIOp, mlir::arith::MinUIOp>>(
        mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::xtile
