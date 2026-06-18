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

#include <memory>
#include <utility>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_TENSORLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class LowerBitcast : public mlir::OpRewritePattern<tensor::BitcastOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      tensor::BitcastOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Value source = op.getSource();
    if (op.getSource().getType().getRank() == 0) {
      source = mlir::tensor::ExtractOp::create(rewriter, op.getLoc(), source);
    }

    mlir::TensorType tensor_result_type = op.getResult().getType();
    bool is_0d_result = tensor_result_type.getRank() == 0;
    mlir::Type triton_result_type =
        is_0d_result ? tensor_result_type.getElementType() : tensor_result_type;

    auto bitcast = ttir::BitcastOp::create(rewriter, op.getLoc(),
                                           triton_result_type, source);

    mlir::Value result = bitcast.getResult();
    if (is_0d_result) {
      result = mlir::tensor::FromElementsOp::create(rewriter, op.getLoc(),
                                                    tensor_result_type, result);
    }

    rewriter.replaceOp(op, result);

    return mlir::success();
  }
};

// TODO(basioli): Consider fusing this with the stablehlo lowering pass into a
// single xtile to triton lowering pass.
class TensorLowerToTritonPass
    : public impl::TensorLowerToTritonPassBase<TensorLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerBitcast>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTensorLowerToTritonPass() {
  return std::make_unique<TensorLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
