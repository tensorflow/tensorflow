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

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_LINALGELEMENTWISETOVECTORPASS
#define GEN_PASS_DEF_LINALGELEMENTWISETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

class ElementwiseToVectorPattern
    : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::LinalgOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!mlir::linalg::isElementwise(op)) {
      return rewriter.notifyMatchFailure(op, "Op is not elementwise");
    }

    // Is this possible?
    if (op.getDpsInits().empty()) {
      return rewriter.notifyMatchFailure(op, "op has no outputs");
    }

    auto result_type =
        mlir::dyn_cast<mlir::ShapedType>(op.getDpsInits().front().getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }
    if (!result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "only static shapes are supported");
    }

    // The default linalg vectorization is very naive and just replaces the
    // elementwise op with a transfer_read -> super_vector -> transfer_write,
    // but this works as a first pass.
    // TODO(willfroom): replace this with explicit loops on natural vector
    // sizes.
    mlir::FailureOr<mlir::linalg::VectorizationResult> result =
        mlir::linalg::vectorize(rewriter, op);

    if (mlir::failed(result)) {
      return rewriter.notifyMatchFailure(op, "could not vectorize");
    }

    rewriter.replaceOp(op, result->replacements);
    return mlir::success();
  }
};

struct LinalgElementwiseToVectorPass
    : public impl::LinalgElementwiseToVectorPassBase<
          LinalgElementwiseToVectorPass> {
  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<ElementwiseToVectorPattern>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLinalgElementwiseToVectorPass() {
  return std::make_unique<LinalgElementwiseToVectorPass>();
}

}  // namespace xla::cpu
