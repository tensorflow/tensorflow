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

#include <utility>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"  // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_TRITONXLAFOLDRESHAPEAROUNDFORLOOPPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class FoldReshapeAroundForLoop : public mlir::OpRewritePattern<scf::ForOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      scf::ForOp for_op, mlir::PatternRewriter& rewriter) const override {
    auto yield_op = mlir::cast<scf::YieldOp>(for_op.getBody()->getTerminator());
    ValueRange yield_operands = yield_op.getOperands();

    for (OpOperand& init_arg : for_op.getInitArgsMutable()) {
      unsigned i = init_arg.getOperandNumber() - for_op.getNumControlOperands();

      ttir::ReshapeOp reshape_op =
          yield_operands[i].getDefiningOp<ttir::ReshapeOp>();
      if (!reshape_op) {
        continue;  // Not yielding a reshape result.
      }
      if (reshape_op.getSrc().getType() == yield_operands[i].getType()) {
        continue;  // The reshape is a no-op.
      }
      if (reshape_op.getSrc().getType().getRank() >=
          reshape_op.getType().getRank()) {
        continue;  // The reshape is not rank increasing.
      }

      Value inner_yield_val = reshape_op.getOperand();
      const Location op_loc = for_op.getLoc();

      // Sink rank reduction for initialization.
      Value new_init = ttir::ReshapeOp::create(
          rewriter, op_loc, inner_yield_val.getType(), init_arg.get());

      // Update the yield of the original loop to provide the un-reshaped
      // value. This prevents 'replaceAndCastForOpIterArg' from cloning the
      // original reshape into the new loop body.
      rewriter.modifyOpInPlace(
          yield_op, [&]() { yield_op->setOperand(i, inner_yield_val); });

      // Use the SCF utility to handle structural rewrite and cast injection.
      SmallVector<Value> new_results = mlir::scf::replaceAndCastForOpIterArg(
          rewriter, for_op, init_arg, new_init,
          [](OpBuilder& b, Location loc, Type type, Value val) -> Value {
            if (val.getType() == type) {
              return val;
            }
            return ttir::ReshapeOp::create(b, loc, type, val);
          });

      rewriter.replaceOp(for_op, new_results);
      return mlir::success();
    }
    return mlir::failure();
  }
};

class TritonXLAFoldReshapeAroundForLoopPass
    : public impl::TritonXLAFoldReshapeAroundForLoopPassBase<
          TritonXLAFoldReshapeAroundForLoopPass> {
 public:
  using TritonXLAFoldReshapeAroundForLoopPassBase::
      TritonXLAFoldReshapeAroundForLoopPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<FoldReshapeAroundForLoop>(mlir_context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace

}  // namespace mlir::triton::xla
