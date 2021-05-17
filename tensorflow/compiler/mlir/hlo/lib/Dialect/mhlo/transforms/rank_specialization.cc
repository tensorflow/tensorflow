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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

/// Needed to build `llvm::SmallSet`s of `mlir::Value`s.
static bool operator<(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
}

namespace mhlo {
namespace {

/// Identify clusters of operations that can be rank-specialized together. The
/// required traits for clustered operations are:
///   - Element-wise: All operations in the group must be element-wise. This
///     allows to reshape operands before applying the operations as well as
///     reshaping the result to the desired shape afterwards. This way, we can,
///     e.g., apply unary ops to a completely flattened operand and restore the
///     original shape afterwards.
///   - Broadcasting semantics: All operations must implement broadcasting
///     semantics. Most importantly, this allows extending operand shapes such
///     that they match in rank. Operations that require all their operands to
///     be of the same shape also fulfill this requirement.
///   - Shape reification: All operations must implement
///     `InferShapedTypeOpInterface`. This is later needed to compute and to
///     restore the desired result shape.

bool IsClusterable(Operation *op) {
  if (!llvm::isa<InferShapedTypeOpInterface>(op)) return false;
  if (op->getNumOperands() == 0) return false;
  return (op->hasTrait<OpTrait::Elementwise>() &&
          op->hasTrait<OpTrait::SameOperandsAndResultShape>()) ||
         (op->hasTrait<chlo::OpTrait::BroadcastingElementwise>() &&
          op->hasTrait<chlo::OpTrait::Broadcasting>());
}

struct RankSpecializationClusterPattern : public RewritePattern {
  explicit RankSpecializationClusterPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only apply to operations that have not been clustered yet.
    if (op->getParentOfType<chlo::RankSpecializationClusterOp>()) {
      return failure();
    }

    // Only cluster when rank specialization is needed.
    if (!IsClusterable(op) || !llvm::any_of(op->getOperandTypes(), [](Type ty) {
          return ty.isa<UnrankedTensorType>();
        })) {
      return failure();
    }

    // Collect all collectively rank specializable ops.
    SmallVector<Operation *, 16> cluster;
    llvm::SmallSet<Value, 16> operand_set;
    llvm::SmallSet<Value, 16> result_set;

    Operation *root_op = op;
    while (root_op->getNextNode() != nullptr &&
           IsClusterable(root_op->getNextNode()))
      root_op = root_op->getNextNode();

    Operation *it = root_op;
    while (it != nullptr && IsClusterable(it)) {
      // Find results that escape the cluster.
      for (OpOperand &use : it->getUses()) {
        if (!llvm::is_contained(cluster, use.getOwner()))
          result_set.insert(use.get());
      }

      // Update cluster operands.
      for (OpResult v : it->getResults()) operand_set.erase(Value(v));
      for (OpOperand &v : it->getOpOperands()) operand_set.insert(v.get());

      cluster.push_back(it);
      it = it->getPrevNode();
    }

    // Create `RankSpecializationClusterOp`.
    auto operands = llvm::to_vector<16>(operand_set);
    auto results = llvm::to_vector<16>(result_set);
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(result_set, [](Value v) { return v.getType(); }));
    Location loc = op->getLoc();
    auto cluster_op = rewriter.create<chlo::RankSpecializationClusterOp>(
        loc, result_types, operands);

    // Create body block.
    auto operand_types = llvm::to_vector<16>(
        llvm::map_range(operand_set, [](Value v) { return v.getType(); }));
    Block *block = rewriter.createBlock(&cluster_op.body(), {}, operand_types);

    // Copy operations into the body.
    BlockAndValueMapping bvm;
    for (auto it : llvm::zip(operands, block->getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    rewriter.setInsertionPointToStart(block);
    for (Operation *it : llvm::reverse(cluster)) rewriter.clone(*it, bvm);

    // Create `RankSpecializationClusterYieldOp`.
    auto mapped_results = llvm::to_vector<16>(
        llvm::map_range(results, [&](Value v) { return bvm.lookup(v); }));
    rewriter.create<chlo::RankSpecializationClusterYieldOp>(loc,
                                                            mapped_results);

    // Replace original ops with the new results.
    for (auto it : llvm::zip(results, cluster_op.results()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    for (Operation *it : cluster) {
      if (it->getUses().empty()) {
        rewriter.eraseOp(it);
        continue;
      }
      auto replacements = llvm::to_vector<16>(llvm::map_range(
          it->getResults(), [&](Value v) { return bvm.lookup(v); }));
      rewriter.replaceOp(it, replacements);
    }

    return success();
  }
};

struct RankSpecializationClusterPass
    : public PassWrapper<RankSpecializationClusterPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::HloClientDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::PopulateRankSpecializationClusterPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

/// Lower rank specialization cluster to SCF.

bool IsScalarTensorType(Type ty) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();
  return ranked_ty && ranked_ty.getRank() == 0;
}

Type DeriveRankedTensorTypes(Type ty, int64_t rank) {
  auto unranked_ty = ty.dyn_cast<UnrankedTensorType>();
  if (!unranked_ty) return ty;
  SmallVector<int64_t, 8> shape(rank, ShapedType::kDynamicSize);
  return RankedTensorType::get(shape, unranked_ty.getElementType());
}

Type DeriveUnrankedTensorTypes(Type ty) {
  if (auto ranked_ty = ty.dyn_cast<RankedTensorType>())
    return UnrankedTensorType::get(ranked_ty.getElementType());
  return ty;
}

Optional<Value> FindUniqueNonScalar(ValueRange values) {
  Value unique_non_scalar;
  for (Value v : values) {
    if (!IsScalarTensorType(v.getType())) {
      if (unique_non_scalar) return llvm::None;
      unique_non_scalar = v;
    }
  }
  if (!unique_non_scalar) return llvm::None;
  return unique_non_scalar;
}

SmallVector<Value, 8> MaterializeRankedOperations(
    OpBuilder &b, Location loc, BlockAndValueMapping &bvm,
    chlo::RankSpecializationClusterOp &op, int64_t target_rank) {
  // Create ranked operations.
  for (Operation &nested_op : op.getBody()->without_terminator()) {
    auto mapped_operands = llvm::to_vector<4>(llvm::map_range(
        nested_op.getOperands(), [&](Value v) { return bvm.lookup(v); }));
    auto ranked_result_types = llvm::to_vector<2>(llvm::map_range(
        nested_op.getResultTypes(),
        [&](Type ty) { return DeriveRankedTensorTypes(ty, target_rank); }));
    OperationState ranked_op_state(loc, nested_op.getName().getStringRef(),
                                   mapped_operands, ranked_result_types,
                                   nested_op.getAttrs());
    Operation *ranked_op = b.createOperation(ranked_op_state);
    for (auto it : llvm::zip(nested_op.getResults(), ranked_op->getResults()))
      bvm.map(std::get<0>(it), std::get<1>(it));
  }

  // Collect ranked results.
  auto yield_op = llvm::cast<chlo::RankSpecializationClusterYieldOp>(
      op.getBody()->getTerminator());
  return llvm::to_vector<8>(llvm::map_range(
      yield_op.results(), [&](Value v) { return bvm.lookup(v); }));
}

SmallVector<Value, 8> MaterializeFinalReshape(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    ValueRange unshaped_results) {
  // Compute result shape.
  auto non_scalar_operands = llvm::make_filter_range(
      op.operands(), [](Value v) { return !IsScalarTensorType(v.getType()); });
  SmallVector<Value, 8> results;
  auto operand_shapes =
      llvm::to_vector<8>(llvm::map_range(non_scalar_operands, [&](Value v) {
        return b.create<shape::ShapeOfOp>(loc, v).result();
      }));
  auto shape = b.create<shape::BroadcastOp>(
      loc, shape::getExtentTensorType(b.getContext()), operand_shapes);

  // Reshape results.
  return llvm::to_vector<8>(
      llvm::map_range(unshaped_results, [&](Value unshaped) {
        return b
            .create<mhlo::DynamicReshapeOp>(
                loc, DeriveUnrankedTensorTypes(unshaped.getType()), unshaped,
                shape)
            .result();
      }));
}

struct LowerSingleNonScalarOperandPattern
    : public OpRewritePattern<chlo::RankSpecializationClusterOp> {
  using OpRewritePattern<chlo::RankSpecializationClusterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::RankSpecializationClusterOp op,
                                PatternRewriter &rewriter) const override {
    // Only apply this pattern if we can statically know that all operands have
    // the same shape or are scalars, i.e. all but one operands are scalars.
    Optional<Value> non_scalar_operand = FindUniqueNonScalar(op.operands());
    if (!non_scalar_operand) return failure();

    // Flatten the non-scalar operand.
    Location loc = op.getLoc();
    Value flat_shape = rewriter.create<tensor::FromElementsOp>(
        loc,
        rewriter
            .create<shape::NumElementsOp>(
                loc, rewriter.getIndexType(),
                rewriter.create<shape::ShapeOfOp>(loc, *non_scalar_operand))
            .result());
    Value flat_non_scalar_operand = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, DeriveRankedTensorTypes(non_scalar_operand->getType(), /*rank=*/1),
        *non_scalar_operand, flat_shape);

    // Materialize ranked variants for the element-wise operations.
    BlockAndValueMapping bvm;
    for (auto it : llvm::zip(op.getBody()->getArguments(), op.operands())) {
      Value operand = std::get<1>(it);
      bvm.map(std::get<0>(it), operand == *non_scalar_operand
                                   ? flat_non_scalar_operand
                                   : operand);
    }
    SmallVector<Value, 8> unshaped_results =
        MaterializeRankedOperations(rewriter, loc, bvm, op, /*target_rank=*/1);

    // Restore the results' expected shape.
    SmallVector<Value, 8> results =
        MaterializeFinalReshape(rewriter, loc, op, unshaped_results);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct RankSpecializationToSCFPass
    : public PassWrapper<RankSpecializationToSCFPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::HloClientDialect,
                    shape::ShapeDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateRankSpecializationToSCFPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateRankSpecializationClusterPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<RankSpecializationClusterPattern>(context);
}

void PopulateRankSpecializationToSCFPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<LowerSingleNonScalarOperandPattern>(context);
}

std::unique_ptr<FunctionPass> createRankSpecializationClusterPass() {
  return std::make_unique<RankSpecializationClusterPass>();
}

std::unique_ptr<FunctionPass> createRankSpecializationToSCFPass() {
  return std::make_unique<RankSpecializationToSCFPass>();
}

}  // namespace mhlo
}  // namespace mlir
