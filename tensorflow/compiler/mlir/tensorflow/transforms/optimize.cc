/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <iostream>

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"

namespace mlir {
namespace TF {
namespace {

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_optimize.inc"

// Returns a TF Constant tensor with the passed in values.
TF::ConstOp GetI64ConstantTensor(PatternRewriter &rewriter,
                                 ArrayRef<int64_t> values, Location location) {
  auto cst_attr = rewriter.getI64TensorAttr(values);
  return rewriter.create<TF::ConstOp>(location, cst_attr.getType(), cst_attr);
}

// Rewrites broadcast->reshape to a reshape->broadcast that reduces
// the rank of the input and output of the broadcast.
class SimplifyBroadcastReshape : public OpRewritePattern<BroadcastToOp> {
  using OpRewritePattern<BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    // Only rewrite if the Broadcast has only one consumer.
    if (!op.output().hasOneUse()) return failure();

    Operation *user = *op.output().getUsers().begin();

    auto reshape_op = llvm::dyn_cast_or_null<ReshapeOp>(user);
    if (!reshape_op) return failure();

    auto reshape_type = reshape_op.output().getType().cast<ShapedType>();

    if (!reshape_type.hasStaticShape()) return failure();
    ArrayRef<int64_t> reshape_shape = reshape_type.getShape();

    auto input_type = op.input().getType().cast<ShapedType>();
    auto output_type = op.output().getType().cast<ShapedType>();

    if (!input_type.hasRank() || !output_type.hasRank()) return failure();

    // The pattern attempts to reduce the rank of the input to BroadcastTo.
    // Thus, we fail to match if the consuming reshape rank is larger.
    ArrayRef<int64_t> input_shape = input_type.getShape();
    if (reshape_shape.size() > input_shape.size()) return failure();

    // Extend the input shape with leading 1s to match the broadcast shape.
    ArrayRef<int64_t> broadcast_shape = output_type.getShape();
    SmallVector<int64_t, 4> input_shape_extended;
    input_shape_extended.append(broadcast_shape.size() - input_shape.size(), 1);
    input_shape_extended.append(input_shape.begin(), input_shape.end());

    // Collect non-unit dims and corresponding dim in the input shape.
    SmallVector<int64_t, 4> input_carryover_dims;
    SmallVector<int64_t, 4> non_unit_dims;

    for (int i = 0; i < input_shape_extended.size(); i++) {
      int64_t dim = broadcast_shape[i];
      if (dim != 1) {
        non_unit_dims.push_back(dim);
        input_carryover_dims.push_back(input_shape_extended[i]);
      }
    }

    // If the reshape rank is less than the number of non-unit dimensions
    // of the broadcast, then the reshape collapses non-unit dimensions.
    // TODO(rahulsp) : Handle this case with more careful checks.
    if (reshape_shape.size() < non_unit_dims.size()) return failure();

    SmallVector<int64_t, 4> old_reshape_non_unit_dims;
    SmallVector<int64_t, 4> new_reshape_dims;
    int new_reshape_dim_idx = 0;
    for (int64_t dim : reshape_shape) {
      int new_reshape_dim = 1;
      if (dim != 1) {
        old_reshape_non_unit_dims.push_back(dim);
        if (new_reshape_dim_idx < input_carryover_dims.size()) {
          new_reshape_dim = input_carryover_dims[new_reshape_dim_idx];
          new_reshape_dim_idx++;
        }
      }
      new_reshape_dims.push_back(new_reshape_dim);
    }

    if (non_unit_dims != old_reshape_non_unit_dims) return failure();

    if (failed(VerifyShapeOfReshapeOp(new_reshape_dims))) return failure();

    Type el_ty = getElementTypeOrSelf(op.getType());
    TF::ConstOp new_reshape_shape = GetI64ConstantTensor(
        rewriter, ArrayRef<int64_t>(new_reshape_dims), op.getLoc());
    auto new_reshape_type = RankedTensorType::get(new_reshape_dims, el_ty);
    ReshapeOp new_reshape =
        rewriter.create<ReshapeOp>(new_reshape_shape.getLoc(), new_reshape_type,
                                   op.input(), new_reshape_shape);
    TF::ConstOp new_broadcast_shape =
        GetI64ConstantTensor(rewriter, reshape_shape, op.getLoc());
    rewriter.replaceOpWithNewOp<BroadcastToOp>(
        reshape_op, reshape_op.output().getType(), new_reshape,
        new_broadcast_shape);
    return success();
  }
};

// Canonicalize operations in functions.
struct TensorFlowOptimizePass
    : public TensorFlowOptimizePassBase<TensorFlowOptimizePass> {
  LogicalResult initialize(MLIRContext *context) override {
    OwningRewritePatternList pattern_list(context);
    populateWithGenerated(pattern_list);
    pattern_list.insert<SimplifyBroadcastReshape>(context);
    patterns = std::move(pattern_list);
    return success();
  }

  void runOnFunction() override {
    auto func = getFunction();
    if (failed(applyPatternsAndFoldGreedily(func, patterns)))
      signalPassFailure();
  }

  FrozenRewritePatternSet patterns;
};

}  // namespace

void CreateTFStandardPipeline(OpPassManager &pm,
                              const StandardPipelineOptions &options) {
  OpPassManager &func_pm = pm.nest<FuncOp>();

  // First operates on the executor dialect:
  // - remove dead islands.
  // - fuse islands as much as possible.
  // - materialize the eventual "pass-through" ops by inlining their content.
  func_pm.addPass(tf_executor::CreateTFExecutorGraphPruningPass());
  func_pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());
  func_pm.addPass(CreateMaterializePassthroughOpPass());
  if (options.form_clusters)
    func_pm.addPass(TFDevice::CreateClusterFormationPass());

  // Hopefully there is a single island left, or there wasn't any to begin with.
  // We now run the optimizer which operates mostly inside islands.
  func_pm.addPass(createCanonicalizerPass());
  pm.addPass(CreateTFShapeInferencePass());
  if (options.enable_inliner) {
    pm.addPass(createInlinerPass());
  }
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(CreateTFOptimizePass());
  pm.addNestedPass<FuncOp>(createCSEPass());
}

std::unique_ptr<OperationPass<FuncOp>> CreateTFOptimizePass() {
  return std::make_unique<TensorFlowOptimizePass>();
}

void RegisterTFOptimizePassPipeline() {
  // Registers a pipeline builder function for the default
  // canonicalize/optimizer.
  static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
      "tf-standard-pipeline",
      "Run all the passes involved in transforming/optimizing the graph after "
      "importing into MLIR, without any target specialization.",
      CreateTFStandardPipeline);
}

}  // namespace TF
}  // namespace mlir
