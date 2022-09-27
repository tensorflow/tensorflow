/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_DETENSORIZELINALG
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::AffineMap;
using mlir::ConversionPatternRewriter;
using mlir::failure;
using mlir::LogicalResult;
using mlir::OpConversionPattern;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::success;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;
using mlir::linalg::GenericOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::FromElementsOp;

bool IsNotZeroRankTensor(RankedTensorType tensor_type) {
  return !tensor_type || tensor_type.getRank() > 0;
}

/// A conversion patttern for detensoring Linalg ops.
struct DetensorizeLinalgOp : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenericOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::SmallVector<AffineMap, 3> indexing_maps = op.getIndexingMapsArray();

    mlir::SmallVector<Value, 3> inputs;
    bool found_zero_dim_tensor = false;
    for (auto& en : llvm::enumerate(op.getInputOperands())) {
      auto tensor_type =
          en.value()->get().getType().dyn_cast<RankedTensorType>();
      if (IsNotZeroRankTensor(tensor_type)) {
        inputs.push_back(en.value()->get());
        continue;
      }
      found_zero_dim_tensor = true;
      indexing_maps[en.index()] =
          AffineMap::get(op.getNumLoops(), 0, llvm::None, op.getContext());
      inputs.push_back(rewriter.create<ExtractOp>(loc, en.value()->get(),
                                                  mlir::ValueRange{}));
    }
    if (!found_zero_dim_tensor) return failure();

    auto linalg_op = rewriter.create<GenericOp>(
        loc, op.getResultTypes(), inputs, op.outputs(),
        rewriter.getAffineMapArrayAttr(indexing_maps), op.iterator_types(),
        mlir::StringAttr(), mlir::StringAttr());
    mlir::Region& region = linalg_op.region();
    rewriter.inlineRegionBefore(op.getBodyRegion(), region, region.end());
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

struct DetensorizeLinalgPass
    : public impl::DetensorizeLinalgBase<DetensorizeLinalgPass> {
  DetensorizeLinalgPass() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto* context = &getContext();

    mlir::ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });
    target.addDynamicallyLegalOp<GenericOp>([&](GenericOp op) {
      return llvm::all_of(TypeRange{op.inputs()}, [&](Type type) {
        return IsNotZeroRankTensor(type.dyn_cast<RankedTensorType>());
      });
    });

    // Detensorize.
    mlir::RewritePatternSet patterns(context);
    patterns.add<DetensorizeLinalgOp>(context);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();

    // Canonicalize.
    mlir::RewritePatternSet canonicalization_patterns(context);
    FromElementsOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            func, std::move(canonicalization_patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDetensorizeLinalgPass() {
  return std::make_unique<DetensorizeLinalgPass>();
}

}  // namespace tensorflow
