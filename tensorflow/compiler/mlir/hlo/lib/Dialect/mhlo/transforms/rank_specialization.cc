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
#include "mlir/Dialect/SCF/SCF.h"
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

bool IsScalarShapeType(Type ty) {
  return ty.cast<RankedTensorType>().getDimSize(0) == 0;
}

Type DeriveRankedTensorTypes(Type ty, int64_t rank) {
  auto tensor_ty = ty.dyn_cast<TensorType>();
  if (!tensor_ty) return ty;
  SmallVector<int64_t, 8> shape(rank, ShapedType::kDynamicSize);
  return RankedTensorType::get(shape, tensor_ty.getElementType());
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
    chlo::RankSpecializationClusterOp op) {
  // Create ranked operations.
  for (Operation &nested_op : op.getBody()->without_terminator()) {
    auto mapped_operands = llvm::to_vector<4>(llvm::map_range(
        nested_op.getOperands(), [&](Value v) { return bvm.lookup(v); }));
    int64_t target_rank = 0;
    for (Value v : mapped_operands) {
      target_rank =
          std::max(target_rank, v.getType().cast<RankedTensorType>().getRank());
    }
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

Value MaterializeScalarRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t non_scalar_idx,
    function_ref<void(OpBuilder &, Location)> else_builder_fn) {
  // Materialize predicate: All operands except one are scalars.
  Value one = b.create<ConstantIndexOp>(loc, 1);
  Value all_others_are_scalar;
  for (auto it : llvm::enumerate(shapes)) {
    if (it.index() == non_scalar_idx) continue;
    // For statically known scalars, there is no need to test.
    if (IsScalarTensorType(op.getOperand(it.index()).getType())) continue;
    auto literal =
        b.create<CmpIOp>(loc, CmpIPredicate::eq,
                         b.create<shape::NumElementsOp>(loc, it.value()), one);
    all_others_are_scalar =
        all_others_are_scalar
            ? b.create<mlir::AndOp>(loc, all_others_are_scalar, literal)
                  .getResult()
            : literal.result();
  }

  auto if_op = b.create<scf::IfOp>(
      loc, op->getResultTypes(), all_others_are_scalar,
      [&](OpBuilder &b, Location loc) {
        // Flatten the non-scalar operand.
        Value flat_shape = b.create<tensor::FromElementsOp>(
            loc, b.create<shape::NumElementsOp>(loc, b.getIndexType(),
                                                shapes[non_scalar_idx])
                     .result());
        Value non_scalar_operand = op.operands()[non_scalar_idx];
        Value flat_non_scalar_operand = b.create<mhlo::DynamicReshapeOp>(
            loc,
            DeriveRankedTensorTypes(non_scalar_operand.getType(), /*rank=*/1),
            non_scalar_operand, flat_shape);

        // Derive ranked operands.
        auto ranked_operands =
            llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
              if (v == non_scalar_operand) return flat_non_scalar_operand;
              return b
                  .create<mhlo::ReshapeOp>(
                      loc, DeriveRankedTensorTypes(v.getType(), /*rank=*/0), v)
                  .getResult();
            }));

        // Materialize ranked variants for the element-wise operations.
        BlockAndValueMapping bvm;
        for (auto it : llvm::zip(op.getBody()->getArguments(), ranked_operands))
          bvm.map(std::get<0>(it), std::get<1>(it));
        Value unshaped_result =
            MaterializeRankedOperations(b, loc, bvm, op).front();

        // Return as unranked tensor for compatibility with the other cases.
        b.create<scf::YieldOp>(
            loc, b.create<tensor::CastOp>(
                      loc, DeriveUnrankedTensorTypes(unshaped_result.getType()),
                      unshaped_result)
                     .dest());
      },
      else_builder_fn);

  return if_op.results().front();
}

Value MaterializeEqualShapesRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes,
    function_ref<void(OpBuilder &, Location)> else_builder_fn) {
  // Materialize all shapes equal predicate.
  Value all_shapes_eq_or_scalar;
  auto non_scalar_shapes = llvm::to_vector<8>(llvm::make_filter_range(
      shapes, [](Value v) { return !IsScalarShapeType(v.getType()); }));
  assert(
      non_scalar_shapes.size() >= 2 &&
      "Equal shapes strategy requires at least two non-scalar operand shapes.");
  for (Value s : llvm::drop_begin(non_scalar_shapes)) {
    auto literal =
        b.create<shape::ShapeEqOp>(loc, non_scalar_shapes.front(), s);
    all_shapes_eq_or_scalar =
        all_shapes_eq_or_scalar
            ? b.create<mlir::AndOp>(loc, all_shapes_eq_or_scalar, literal)
                  .result()
            : literal;
  }

  auto if_op = b.create<scf::IfOp>(
      loc, op->getResultTypes(), all_shapes_eq_or_scalar,
      [&](OpBuilder &b, Location loc) {
        // Flatten non-scalar operands.
        Value shape = non_scalar_shapes.front();
        for (Value s : llvm::drop_begin(non_scalar_shapes)) {
          shape = b.create<shape::AnyOp>(loc, shape.getType(),
                                         ValueRange{shape, s});
        }
        Value flat_shape = b.create<tensor::FromElementsOp>(
            loc, b.create<shape::NumElementsOp>(loc, b.getIndexType(), shape)
                     .result());
        auto flat_operands =
            llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
              if (IsScalarTensorType(v.getType())) return v;
              return b
                  .create<mhlo::DynamicReshapeOp>(
                      loc, DeriveRankedTensorTypes(v.getType(), /*rank=*/1), v,
                      flat_shape)
                  .result();
            }));

        // Materialize ranked variants for the element-wise operations.
        BlockAndValueMapping bvm;
        for (auto it : llvm::zip(op.getBody()->getArguments(), flat_operands))
          bvm.map(std::get<0>(it), std::get<1>(it));
        Value unshaped_result =
            MaterializeRankedOperations(b, loc, bvm, op).front();

        // Return as unranked tensor for compatibility with the other cases.
        b.create<scf::YieldOp>(
            loc, b.create<tensor::CastOp>(
                      loc, DeriveUnrankedTensorTypes(unshaped_result.getType()),
                      unshaped_result)
                     .dest());
      },
      else_builder_fn);

  return if_op.results().front();
}

Value MaterializeTargetRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t target_rank) {
  // Reshape operands to match the target rank.
  RankedTensorType extent_tensor_ty =
      shape::getExtentTensorType(b.getContext(), target_rank);
  Value all_ones_shape = b.create<shape::ConstShapeOp>(
      loc, extent_tensor_ty,
      mlir::DenseIntElementsAttr::get(extent_tensor_ty,
                                      SmallVector<int64_t, 6>(target_rank, 1)));
  SmallVector<Value, 8> ranked_operands;
  for (auto it : llvm::zip(op.operands(), shapes)) {
    Value operand, shape;
    std::tie(operand, shape) = it;
    if (operand.getType().isa<RankedTensorType>()) {
      ranked_operands.push_back(operand);
      continue;
    }
    Value ranked_shape = b.create<tensor::CastOp>(
        loc, extent_tensor_ty,
        b.create<shape::BroadcastOp>(loc,
                                     shape::getExtentTensorType(b.getContext()),
                                     shape, all_ones_shape,
                                     /*error=*/nullptr));
    ranked_operands.push_back(b.create<mhlo::DynamicReshapeOp>(
        loc, DeriveRankedTensorTypes(operand.getType(), target_rank), operand,
        ranked_shape));
  }

  // Materialize ranked versions of the element-wise operations.
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(op.body().front().getArguments(), ranked_operands))
    bvm.map(std::get<0>(it), std::get<1>(it));

  // Return as unranked for compatibility with other target ranks.
  auto unshaped_result = MaterializeRankedOperations(b, loc, bvm, op).front();
  return b.create<tensor::CastOp>(
      loc, DeriveUnrankedTensorTypes(unshaped_result.getType()),
      unshaped_result);
}

Value RecusivelyMaterializeTargetRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, Value max_rank,
    int64_t min_target_rank, int64_t max_target_rank) {
  Value condition =
      b.create<CmpIOp>(loc, CmpIPredicate::ule, max_rank,
                       b.create<ConstantIndexOp>(loc, min_target_rank));

  // If only a unique target rank is left, we can lower to an assert instead
  // of the usual if operation.
  if (min_target_rank == max_target_rank) {
    b.create<AssertOp>(loc, condition,
                       "Input for dynamic binary or n-ary op lowering was of "
                       "a rank greater than " +
                           std::to_string(max_target_rank));
    return MaterializeTargetRankSpecializationCase(b, loc, op, shapes,
                                                   min_target_rank);
  }

  // Materialize IR for the smallest considered target rank.
  auto if_op = b.create<scf::IfOp>(loc, op->getResultTypes(), condition,
                                   /*withElseRegion=*/true);
  auto then_builder = if_op.getThenBodyBuilder();
  then_builder.create<scf::YieldOp>(
      loc, MaterializeTargetRankSpecializationCase(then_builder, loc, op,
                                                   shapes, min_target_rank));

  // Recurse for all remaining target ranks.
  auto else_builder = if_op.getElseBodyBuilder();
  else_builder.create<scf::YieldOp>(
      loc, RecusivelyMaterializeTargetRankSpecializationCases(
               else_builder, loc, op, shapes, max_rank, min_target_rank + 1,
               max_target_rank));

  return if_op.results().front();
}

Value MaterializeGenericRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes) {
  // Get the minimum broadcast shapes of the operands.
  auto non_scalar_shapes = llvm::to_vector<8>(llvm::make_filter_range(
      shapes, [](Value v) { return !IsScalarShapeType(v.getType()); }));
  auto min_bcast_shapes_op = b.create<chlo::MinimumBroadcastShapesOp>(
      loc,
      SmallVector<Type, 8>(non_scalar_shapes.size(),
                           shape::getExtentTensorType(b.getContext())),
      non_scalar_shapes);

  // Find the maximum rank among the reduced operand shapes.
  Value max_rank;
  for (Value shape : min_bcast_shapes_op.results()) {
    Value rank = b.create<shape::RankOp>(loc, b.getIndexType(), shape);
    if (!max_rank) {
      max_rank = rank;
    } else {
      max_rank = b.create<mlir::SelectOp>(
          loc, b.create<CmpIOp>(loc, CmpIPredicate::sgt, max_rank, rank),
          max_rank, rank);
    }
  }

  // Collect reduced shapes.
  SmallVector<Value, 8> reduced_shapes;
  auto it = min_bcast_shapes_op.result_begin();
  for (Value s : shapes) {
    if (IsScalarShapeType(s.getType())) {
      reduced_shapes.push_back(s);
    } else {
      reduced_shapes.push_back(*it++);
    }
  }

  // Materialize rank specialization for ranks 1, ..., 8.
  // TODO(frgossen): For clusters w/o a select operation, consider only ranks
  // 1, ..., 5.
  const int64_t kMinTargetRank = 1;
  const int64_t kMaxTargetRank = 8;
  return RecusivelyMaterializeTargetRankSpecializationCases(
      b, loc, op, reduced_shapes, max_rank, kMinTargetRank, kMaxTargetRank);
}

Value MaterializeDefaultRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes) {
  return MaterializeEqualShapesRankSpecializationCase(
      b, loc, op, shapes, [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(
            loc, MaterializeGenericRankSpecializationCases(b, loc, op, shapes));
      });
}

SmallVector<Value, 8> MaterializeRankSpecializationForSingleNonScalarOperand(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    Value non_scalar_operand) {
  // Flatten the non-scalar operand.
  Value flat_shape = b.create<tensor::FromElementsOp>(
      loc, b.create<shape::NumElementsOp>(
                loc, b.getIndexType(),
                b.create<shape::ShapeOfOp>(loc, non_scalar_operand))
               .result());
  Value flat_non_scalar_operand = b.create<mhlo::DynamicReshapeOp>(
      loc, DeriveRankedTensorTypes(non_scalar_operand.getType(), /*rank=*/1),
      non_scalar_operand, flat_shape);

  // Materialize ranked variants for the element-wise operations.
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(op.getBody()->getArguments(), op.operands())) {
    Value operand;
    Value bb_arg;
    std::tie(bb_arg, operand) = it;
    bvm.map(bb_arg,
            operand == non_scalar_operand ? flat_non_scalar_operand : operand);
  }
  SmallVector<Value, 8> unshaped_results =
      MaterializeRankedOperations(b, loc, bvm, op);

  // Restore the results' expected shape.
  return MaterializeFinalReshape(b, loc, op, unshaped_results);
}

Value MaterializeRankSpecializationForTwoNonScalarOperands(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    ValueRange non_scalar_operands) {
  assert(non_scalar_operands.size() == 2);

  auto shapes = llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
    return b.create<shape::ShapeOfOp>(loc, v).result();
  }));
  auto non_scalar_lhs = llvm::find(op.operands(), non_scalar_operands[0]);
  auto non_scalar_rhs = llvm::find(op.operands(), non_scalar_operands[1]);

  // Materialize all the different cases.
  Value unshaped_result = MaterializeScalarRankSpecializationCase(
      b, loc, op, shapes, non_scalar_rhs.getIndex(),
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(
            loc, MaterializeScalarRankSpecializationCase(
                     b, loc, op, shapes, non_scalar_lhs.getIndex(),
                     [&](OpBuilder &b, Location loc) {
                       b.create<scf::YieldOp>(
                           loc, MaterializeDefaultRankSpecializationCases(
                                    b, loc, op, shapes));
                     }));
      });

  // Materialize final reshape once and for all rank specialization cases.
  return MaterializeFinalReshape(b, loc, op, unshaped_result).front();
}

// Materialize rank generic rank specialization.
Value MaterializeDefaultRankSpecialization(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op) {
  auto shapes = llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
    return b.create<shape::ShapeOfOp>(loc, v).result();
  }));

  // Materialize all the different cases.
  Value unshaped_result =
      MaterializeDefaultRankSpecializationCases(b, loc, op, shapes);

  // Materialize final reshape once and for all rank specialization cases.
  return MaterializeFinalReshape(b, loc, op, unshaped_result).front();
}

struct LowerRankSpecializationClusterPattern
    : public OpRewritePattern<chlo::RankSpecializationClusterOp> {
  using OpRewritePattern<chlo::RankSpecializationClusterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::RankSpecializationClusterOp op,
                                PatternRewriter &rewriter) const override {
    // Restoring the result shape currently relies on all operands being used
    // for a single result. The result shape is then the broadcasted shape of
    // all operands.
    if (op.getNumResults() != 1) return failure();

    // If there is only a single non-scalar operand, we can flatten that operand
    // completely.
    Location loc = op.getLoc();
    auto non_scalar_operands =
        llvm::to_vector<2>(llvm::make_filter_range(op.operands(), [](Value v) {
          return !IsScalarTensorType(v.getType());
        }));
    if (non_scalar_operands.size() == 1) {
      rewriter.replaceOp(op,
                         MaterializeRankSpecializationForSingleNonScalarOperand(
                             rewriter, loc, op, non_scalar_operands.front()));
      return success();
    }

    // If there are exactly two unranked operands and all others are known to be
    // scalars, we can consider two extra cases: If either of the unranked
    // operands turns out to be a scalar at runtime, we can, again, apply the
    // trick for a single non-scalar operand.
    if (non_scalar_operands.size() == 2 &&
        llvm::all_of(non_scalar_operands, [](Value v) {
          return v.getType().isa<UnrankedTensorType>();
        })) {
      rewriter.replaceOp(op,
                         MaterializeRankSpecializationForTwoNonScalarOperands(
                             rewriter, loc, op, non_scalar_operands));
      return success();
    }

    // For all other cases, reshape the operands to match in rank, apply the
    // operation, and restore the expected shape.
    rewriter.replaceOp(op,
                       MaterializeDefaultRankSpecialization(rewriter, loc, op));
    return success();
  }
};

struct RankSpecializationToSCFPass
    : public PassWrapper<RankSpecializationToSCFPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::HloClientDialect,
                    shape::ShapeDialect, scf::SCFDialect>();
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
  patterns->insert<LowerRankSpecializationClusterPattern>(context);
}

std::unique_ptr<FunctionPass> createRankSpecializationClusterPass() {
  return std::make_unique<RankSpecializationClusterPass>();
}

std::unique_ptr<FunctionPass> createRankSpecializationToSCFPass() {
  return std::make_unique<RankSpecializationToSCFPass>();
}

}  // namespace mhlo
}  // namespace mlir
