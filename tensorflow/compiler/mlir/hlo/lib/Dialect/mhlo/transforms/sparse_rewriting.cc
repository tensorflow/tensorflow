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

// This file implements a set of sparse MHLO rewriting rules.

#include <utility>

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

/// Approves subsuming sparse types into operation.
// TODO(b/231360416): replace this list with "supports sparsity" trait?
static bool canFuseWithSparseConvert(Operation *op) {
  return isa<sparse_tensor::ConvertOp>(op) || isa<AbsOp>(op) ||
         isa<CeilOp>(op) || isa<ConvertOp>(op) || isa<CosOp>(op) ||
         isa<Expm1Op>(op) || isa<FloorOp>(op) || isa<ImagOp>(op) ||
         isa<LogOp>(op) || isa<Log1pOp>(op) || isa<NegOp>(op) ||
         isa<RealOp>(op) || isa<RoundOp>(op) || isa<SignOp>(op) ||
         isa<SinOp>(op) || isa<SqrtOp>(op) || isa<TanhOp>(op) ||
         isa<AddOp>(op) || isa<DivOp>(op) || isa<MulOp>(op) || isa<RemOp>(op) ||
         isa<SubOp>(op);
}

/// Fuses a sparse tensor type from a conversion into a mhlo operation
/// where possible, essentially rewriting something like:
///    %0 = mhlo.sign %arg : tensor<100xf64>
///    %1 = sparse_tensor.convert %0 : tensor<100xf64> to tensor<100xf64, #SV>
///    ... = ... %1 ...
/// into:
///    %0 = mhlo.sign %arg : (tensor<100xf64>) -> tensor<100xf64, #SV>
///    ... = ... %0 ...
/// This eventually yields better sparse code, since the intermediate
/// results do not need to be explicitly generated.
struct SparseConvertConverter
    : public OpRewritePattern<sparse_tensor::ConvertOp> {
  explicit SparseConvertConverter(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(sparse_tensor::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    if (Operation *def = op.source().getDefiningOp()) {
      if (def->hasOneUse() && canFuseWithSparseConvert(def)) {
        def->getResult(0).setType(op->getResultTypes()[0]);
        rewriter.replaceOp(op, def->getResult(0));
        return success();
      }
    }
    return failure();
  }
};

struct SparseRewritingPass
    : public SparseRewritingPassBase<SparseRewritingPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSparseRewritingPatterns(&patterns, &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateSparseRewritingPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
  patterns->add<SparseConvertConverter>(ctx);
}

std::unique_ptr<OperationPass<func::FuncOp>> createSparseRewritingPass() {
  return std::make_unique<SparseRewritingPass>();
}

}  // namespace mhlo
}  // namespace mlir
