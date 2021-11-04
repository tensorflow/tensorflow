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

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
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

/// Needed to build `llvm::SmallSet`s and `llvm::EquivalenceClasses` of
/// `mlir::Value`s.
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
  return (op->hasTrait<mlir::OpTrait::Elementwise>() &&
          op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) ||
         op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>();
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

struct MergeRankSpecializationClusterOpsPattern
    : public OpRewritePattern<chlo::RankSpecializationClusterOp> {
  using OpRewritePattern<chlo::RankSpecializationClusterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::RankSpecializationClusterOp op,
                                PatternRewriter &rewriter) const override {
    auto preceding_op =
        llvm::dyn_cast_or_null<chlo::RankSpecializationClusterOp>(
            op->getPrevNode());
    if (!preceding_op) return failure();
    Block *body = op.getBody();
    Block *preceding_body = preceding_op.getBody();
    auto yield_op = llvm::dyn_cast<chlo::RankSpecializationClusterYieldOp>(
        op.getBody()->getTerminator());
    auto preceding_yield_op =
        llvm::dyn_cast<chlo::RankSpecializationClusterYieldOp>(
            preceding_op.getBody()->getTerminator());

    // Merge cluster operands. Consider only those operands of the second
    // cluster that do not originate in the preceding cluster.
    SmallVector<Value, 8> new_operands;
    for (Value v : preceding_op.operands()) new_operands.push_back(v);
    for (Value v : op.operands()) {
      if (v.getDefiningOp() != preceding_op &&
          !llvm::is_contained(preceding_op.operands(), v)) {
        new_operands.push_back(v);
      }
    }

    // Merge cluster results. Consider only those results of the preceding
    // cluster that are not exclusively used as operands to the second cluster.
    SmallVector<Value, 8> new_unmapped_results;
    for (auto it :
         llvm::zip(preceding_op.results(), preceding_yield_op.results())) {
      Value result, inner_result;
      std::tie(result, inner_result) = it;
      if (!llvm::all_of(result.getUsers(),
                        [&](Operation *user) { return user == op; })) {
        new_unmapped_results.push_back(inner_result);
      }
    }
    for (Value v : yield_op.results()) new_unmapped_results.push_back(v);

    // Create merged cluster op.
    rewriter.setInsertionPoint(preceding_op);
    auto loc = op.getLoc();
    auto result_types = llvm::to_vector<16>(llvm::map_range(
        new_unmapped_results, [](Value v) { return v.getType(); }));
    auto new_op = rewriter.create<chlo::RankSpecializationClusterOp>(
        loc, result_types, new_operands);
    auto operand_types = llvm::to_vector<16>(
        llvm::map_range(new_operands, [](Value v) { return v.getType(); }));
    Block *new_body = rewriter.createBlock(&new_op.body(), {}, operand_types);
    rewriter.setInsertionPointToStart(new_body);

    // Map operands and copy operations of the preceding cluster into the new
    // body.
    BlockAndValueMapping bvm;
    for (auto it : llvm::enumerate(preceding_body->getArguments()))
      bvm.map(it.value(), new_body->getArgument(it.index()));
    for (Operation &nested_op : preceding_body->without_terminator())
      rewriter.clone(nested_op, bvm);

    // Map operands and copy operations of the second cluster. If they result
    // from the preceeding cluster, we can simply map the corresponding value
    // internally.
    for (auto it : llvm::zip(body->getArguments(), op.operands())) {
      Value block_arg, operand;
      std::tie(block_arg, operand) = it;
      if (operand.getDefiningOp() == preceding_op) {
        auto where = llvm::find(preceding_op.results(), operand);
        assert(where.getBase() != nullptr && "expected to find ");
        bvm.map(block_arg,
                bvm.lookup(preceding_yield_op.getOperand(where.getIndex())));
      } else {
        auto where = llvm::find(new_op.operands(), operand);
        bvm.map(block_arg, new_body->getArgument(where.getIndex()));
      }
    }
    for (Operation &nested_op : body->without_terminator()) {
      rewriter.clone(nested_op, bvm);
    }

    // Yield inner results.
    rewriter.create<chlo::RankSpecializationClusterYieldOp>(
        loc,
        llvm::to_vector<16>(llvm::map_range(new_unmapped_results, [&](Value v) {
          return bvm.lookupOrDefault(v);
        })));

    // Replace the two cluster ops with the new corresponding results.
    SmallVector<Value, 8> preceding_op_replacements;
    int64_t i = 0;
    for (Value result : preceding_op.results()) {
      Value replacement = nullptr;
      if (!llvm::all_of(result.getUsers(),
                        [&](Operation *user) { return user == op; })) {
        replacement = new_op->getResult(i++);
      }
      preceding_op_replacements.push_back(replacement);
    }
    ValueRange op_replacements = new_op.results().take_back(op.getNumResults());
    rewriter.replaceOp(op, op_replacements);
    rewriter.replaceOp(preceding_op, preceding_op_replacements);

    return success();
  }
};

struct RankSpecializationClusterPass
    : public RankSpecializationClusterPassBase<RankSpecializationClusterPass> {
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
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op, ValueRange unshaped_results) {
  auto yield_op = llvm::cast<chlo::RankSpecializationClusterYieldOp>(
      op.getBody()->getTerminator());
  assert(unshaped_results.size() == 1 && yield_op.results().size() == 1 &&
         "Currently, rank specialization supports only one result.");

  // Reify result shape.
  Operation *last_op_before_shape_reification = op->getPrevNode();
  SmallVector<Value, 1> result_shape;
  Value original_result = yield_op.results().front();
  auto original_result_iface =
      llvm::cast<InferShapedTypeOpInterface>(original_result.getDefiningOp());
  if (failed(original_result_iface.reifyReturnTypeShapes(
          rewriter, original_result_iface->getOperands(), result_shape))) {
    return {};
  }

  // Materialize final reshape.
  Value unshaped_result = unshaped_results.front();
  Value result = rewriter.create<mhlo::DynamicReshapeOp>(
      loc, DeriveUnrankedTensorTypes(unshaped_result.getType()),
      unshaped_result, result_shape.front());

  // Reify shapes until they are independent of operations in the original
  // cluster.
  {
    Operation *it = result_shape.front().getDefiningOp();
    while (it != nullptr && it != last_op_before_shape_reification) {
      bool advanced = false;
      if (auto shape_of_op = llvm::dyn_cast<shape::ShapeOfOp>(it)) {
        Operation *def = shape_of_op.getArg().getDefiningOp();
        if (def && def->getBlock() == op.getBody()) {
          // Resolve `shape_of` op because it still depends on operation in the
          // original cluster.
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(shape_of_op);
          SmallVector<Value, 1> tmp_shape;
          auto iface = llvm::cast<InferShapedTypeOpInterface>(def);
          if (failed(iface.reifyReturnTypeShapes(rewriter, iface->getOperands(),
                                                 tmp_shape)))
            return {};
          rewriter.replaceOp(shape_of_op, tmp_shape.front());

          // Continue, including the newly created operations.
          it = tmp_shape.front().getDefiningOp();
          advanced = true;
        }
      }

      // Skip op, otherwise.
      if (!advanced) it = it->getPrevNode();
    }
  }

  // Replace all remaining uses of the original cluster's block args.
  for (auto it : llvm::zip(op.operands(), op.getBody()->getArguments())) {
    Value operand, barg;
    std::tie(operand, barg) = it;
    barg.replaceUsesWithIf(operand, [&](OpOperand &operand) {
      return operand.getOwner()->getBlock() != op.getBody();
    });
  }

  return {result};
}

Value MaterializeFlatShape(OpBuilder &b, Location loc, ValueRange same_shapes) {
  assert(!same_shapes.empty() && "Expected at least one shape.");
  Value shape = same_shapes.size() == 1
                    ? same_shapes.front()
                    : b.create<shape::AnyOp>(loc, same_shapes.front().getType(),
                                             same_shapes);
  return b.create<tensor::FromElementsOp>(
      loc,
      b.create<shape::NumElementsOp>(loc, b.getIndexType(), shape).getResult());
}

Value MaterializeScalarRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, ValueRange non_scalars_of_same_shape,
    function_ref<void(OpBuilder &, Location)> else_builder_fn) {
  // Materialize predicate: All operands are scalars, except the expected
  // non-scalars.
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value all_others_are_scalar;
  for (auto it : llvm::zip(op.operands(), shapes)) {
    Value operand, shape;
    std::tie(operand, shape) = it;
    if (llvm::is_contained(non_scalars_of_same_shape, operand) ||
        IsScalarTensorType(operand.getType())) {
      continue;
    }
    auto literal = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        b.create<shape::NumElementsOp>(loc, shape), one);
    all_others_are_scalar =
        all_others_are_scalar
            ? b.create<mlir::arith::AndIOp>(loc, all_others_are_scalar, literal)
                  .getResult()
            : literal.getResult();
  }

  auto if_op = b.create<scf::IfOp>(
      loc, op->getResultTypes(), all_others_are_scalar,
      [&](OpBuilder &b, Location loc) {
        // Compute flat non-scalar shape.
        SmallVector<Value, 4> non_scalar_shapes;
        for (auto it : llvm::zip(op.operands(), shapes)) {
          Value operand, shape;
          std::tie(operand, shape) = it;
          if (llvm::is_contained(non_scalars_of_same_shape, operand))
            non_scalar_shapes.push_back(shape);
        }
        Value flat_shape = MaterializeFlatShape(b, loc, non_scalar_shapes);

        // Derive ranked operands.
        auto ranked_operands =
            llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
              if (IsScalarTensorType(v.getType())) return v;
              if (!llvm::is_contained(non_scalars_of_same_shape, v)) {
                return b
                    .create<mhlo::ReshapeOp>(
                        loc, DeriveRankedTensorTypes(v.getType(), /*rank=*/0),
                        v)
                    .getResult();
              }
              return b
                  .create<mhlo::DynamicReshapeOp>(
                      loc, DeriveRankedTensorTypes(v.getType(), /*rank=*/1), v,
                      flat_shape)
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
    all_shapes_eq_or_scalar = all_shapes_eq_or_scalar
                                  ? b.create<mlir::arith::AndIOp>(
                                         loc, all_shapes_eq_or_scalar, literal)
                                        .getResult()
                                  : literal;
  }

  auto if_op = b.create<scf::IfOp>(
      loc, op->getResultTypes(), all_shapes_eq_or_scalar,
      [&](OpBuilder &b, Location loc) {
        // Flatten non-scalar operands.
        Value flat_shape = MaterializeFlatShape(b, loc, non_scalar_shapes);
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
  // Reshape unranked operands to match the target rank.
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
  Value condition = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ule, max_rank,
      b.create<arith::ConstantIndexOp>(loc, min_target_rank));

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
    const SmallVector<Value, 8> &shapes, int64_t max_target_rank) {
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
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, max_rank,
                                  rank),
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

  // Materialize rank specialization for ranks 1, ...
  return RecusivelyMaterializeTargetRankSpecializationCases(
      b, loc, op, reduced_shapes, max_rank, /*min_target_rank=*/1,
      max_target_rank);
}

Value MaterializeDefaultRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t max_target_rank) {
  return MaterializeEqualShapesRankSpecializationCase(
      b, loc, op, shapes, [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, MaterializeGenericRankSpecializationCases(
                                        b, loc, op, shapes, max_target_rank));
      });
}

SmallVector<Value, 8>
MaterializeRankSpecializationForSingleNonScalarShapeEquivalenceClass(
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op,
    ValueRange non_scalars_of_same_shape) {
  // Compute flat operand shape.
  auto non_scalar_shapes = llvm::to_vector<4>(
      llvm::map_range(non_scalars_of_same_shape, [&](Value v) {
        return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
      }));
  Value flat_shape = MaterializeFlatShape(rewriter, loc, non_scalar_shapes);

  // Materialize ranked variants for the element-wise operations.
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(op.getBody()->getArguments(), op.operands())) {
    Value operand;
    Value bb_arg;
    std::tie(bb_arg, operand) = it;
    if (!IsScalarTensorType(operand.getType())) {
      assert(llvm::is_contained(non_scalars_of_same_shape, operand) &&
             "Expected all non-scalars in the same shape equivalence class.");
      operand = rewriter.create<mhlo::DynamicReshapeOp>(
          loc, DeriveRankedTensorTypes(operand.getType(), /*rank=*/1), operand,
          flat_shape);
    }
    bvm.map(bb_arg, operand);
  }
  SmallVector<Value, 8> unshaped_results =
      MaterializeRankedOperations(rewriter, loc, bvm, op);

  // Restore the results' expected shape.
  Value shape = non_scalar_shapes.front();
  return llvm::to_vector<8>(llvm::map_range(unshaped_results, [&](Value v) {
    return rewriter
        .create<mhlo::DynamicReshapeOp>(
            loc, DeriveUnrankedTensorTypes(v.getType()), v, shape)
        .result();
  }));
}

Value MaterializeRankSpecializationForTwoNonScalarShapeEquivalenceClasses(
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op,
    SmallVector<SmallVector<Value, 4>, 4> non_scalar_eqs,
    int64_t max_target_rank) {
  assert(non_scalar_eqs.size() == 2 &&
         "Expect two non-scalar equivalence classes.");
  auto shapes = llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
    return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
  }));
  ValueRange lhs_non_scalar_eqs = non_scalar_eqs[0];
  ValueRange rhs_non_scalar_eqs = non_scalar_eqs[1];

  // Materialize all the different cases.
  Value unshaped_result = MaterializeScalarRankSpecializationCase(
      rewriter, loc, op, shapes, rhs_non_scalar_eqs,
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(
            loc, MaterializeScalarRankSpecializationCase(
                     b, loc, op, shapes, lhs_non_scalar_eqs,
                     [&](OpBuilder &b, Location loc) {
                       b.create<scf::YieldOp>(
                           loc, MaterializeDefaultRankSpecializationCases(
                                    b, loc, op, shapes, max_target_rank));
                     }));
      });

  // Materialize final reshape once and for all rank specialization cases.
  return MaterializeFinalReshape(rewriter, loc, op, unshaped_result).front();
}

// Materialize rank generic rank specialization.
Value MaterializeDefaultRankSpecialization(PatternRewriter &rewriter,
                                           Location loc,
                                           chlo::RankSpecializationClusterOp op,
                                           int64_t max_target_rank) {
  auto shapes = llvm::to_vector<8>(llvm::map_range(op.operands(), [&](Value v) {
    return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
  }));

  // Materialize all the different cases.
  Value unshaped_result = MaterializeDefaultRankSpecializationCases(
      rewriter, loc, op, shapes, max_target_rank);

  // Materialize final reshape once and for all rank specialization cases.
  return MaterializeFinalReshape(rewriter, loc, op, unshaped_result).front();
}

// This is a very limited form of shape inference. It is correct but incomplete.
SmallVector<SmallVector<Value, 4>, 4> FindNonScalarShapeEquivalences(
    chlo::RankSpecializationClusterOp op) {
  llvm::EquivalenceClasses<Value> eqs;

  // Bridge the equivalences between operands and block arguments.
  for (auto it : llvm::zip(op.operands(), op.getBody()->getArguments()))
    eqs.unionSets(std::get<0>(it), std::get<1>(it));

  // Find equalities through `SameOperandsAndResultShape` trait.
  auto union_sets = [&](ValueRange vs) {
    if (vs.empty()) return;
    Value repr = vs.front();
    for (Value v : vs.drop_front()) eqs.unionSets(repr, v);
  };
  for (Operation &nested_op : op.getBody()->without_terminator()) {
    if (nested_op.hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      union_sets(nested_op.getOperands());
      union_sets(nested_op.getResults());
      if (!nested_op.getOperands().empty() && !nested_op.getResults().empty())
        eqs.unionSets(nested_op.getResult(0), nested_op.getOperand(0));
    }
  }

  // Find shape equalities through surrounding constraints.
  if (auto assuming_op = op->getParentOfType<shape::AssumingOp>()) {
    SmallVector<Operation *, 8> queue;
    auto append_if_not_null = [&](Operation *op) {
      if (op != nullptr) queue.push_back(op);
    };
    append_if_not_null(assuming_op.getWitness().getDefiningOp());
    while (!queue.empty()) {
      Operation *it = queue.pop_back_val();
      if (auto assuming_all_op = llvm::dyn_cast<shape::AssumingAllOp>(it)) {
        for (Value v : assuming_all_op.getInputs())
          append_if_not_null(v.getDefiningOp());
      } else if (auto cstr_eq_op = llvm::dyn_cast<shape::CstrEqOp>(it)) {
        Value ref_arg;
        for (Value v : cstr_eq_op.getShapes()) {
          if (auto shape_of_op =
                  dyn_cast_or_null<shape::ShapeOfOp>(v.getDefiningOp())) {
            if (!ref_arg) {
              ref_arg = shape_of_op.getArg();
            } else {
              eqs.unionSets(ref_arg, shape_of_op.getArg());
            }
          }
        }
      }
    }
  }

  // Find equalities through special knowledge of ops.
  // TODO(frgossen): Remove this when these shape equalities can be inferred
  // from surrounding shape constraints.
  for (Operation &nested_op : op.getBody()->without_terminator()) {
    if (auto select_op = llvm::dyn_cast<mhlo::SelectOp>(nested_op)) {
      union_sets(
          {select_op.on_true(), select_op.on_false(), select_op.getResult()});
    } else if (auto clamp_op = llvm::dyn_cast<mhlo::ClampOp>(nested_op)) {
      union_sets({clamp_op.operand(), clamp_op.getResult()});
    }
  }

  // Convert to a list-like equivalence class representation.
  SmallVector<SmallVector<Value, 4>, 4> non_scalar_eqs;
  for (Value v : op.operands()) {
    if (IsScalarTensorType(v.getType())) continue;
    bool inserted = false;
    for (auto &eq_class : non_scalar_eqs) {
      if (eqs.isEquivalent(eq_class.front(), v)) {
        eq_class.push_back(v);
        inserted = true;
        break;
      }
    }
    if (!inserted) non_scalar_eqs.push_back(SmallVector<Value, 4>({v}));
  }

  return non_scalar_eqs;
}

struct LowerRankSpecializationClusterPattern
    : public OpRewritePattern<chlo::RankSpecializationClusterOp> {
  LowerRankSpecializationClusterPattern(MLIRContext *ctx,
                                        int64_t max_target_rank)
      : OpRewritePattern<chlo::RankSpecializationClusterOp>(ctx, /*benefit=*/1),
        max_target_rank(max_target_rank) {}

  LogicalResult matchAndRewrite(chlo::RankSpecializationClusterOp op,
                                PatternRewriter &rewriter) const override {
    // Restoring the result shape currently relies on all operands being used
    // for a single result. The result shape is then the broadcasted shape of
    // all operands.
    if (op.getNumResults() != 1) return failure();

    // If there is only a single non-scalar shape equivalence class, we can
    // flatten that operands completely.
    SmallVector<SmallVector<Value, 4>, 4> non_scalar_eqs =
        FindNonScalarShapeEquivalences(op);
    Location loc = op.getLoc();
    if (non_scalar_eqs.size() == 1) {
      rewriter.replaceOp(
          op,
          MaterializeRankSpecializationForSingleNonScalarShapeEquivalenceClass(
              rewriter, loc, op, non_scalar_eqs.front()));
      return success();
    }

    // If there are exactly two non-scalar shape equivalence classes, we can
    // consider two extra cases: If either of the operand classes turns out to
    // be all-scalars at runtime, we can, again, flatten all operands.
    if (non_scalar_eqs.size() == 2) {
      rewriter.replaceOp(
          op,
          MaterializeRankSpecializationForTwoNonScalarShapeEquivalenceClasses(
              rewriter, loc, op, non_scalar_eqs, max_target_rank));
      return success();
    }

    // For all other cases, reshape the operands to match in rank, apply the
    // operation, and restore the expected shape.
    rewriter.replaceOp(op, MaterializeDefaultRankSpecialization(
                               rewriter, loc, op, max_target_rank));
    return success();
  }

 private:
  int64_t max_target_rank;
};

struct RankSpecializationToSCFPass
    : public RankSpecializationToSCFPassBase<RankSpecializationToSCFPass> {
  explicit RankSpecializationToSCFPass(int64_t max_target_rank)
      : RankSpecializationToSCFPassBase<
            RankSpecializationToSCFPass>::RankSpecializationToSCFPassBase() {
    this->max_target_rank_ = max_target_rank;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::HloClientDialect,
                    shape::ShapeDialect, scf::SCFDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateRankSpecializationToSCFPatterns(ctx, &patterns,
                                            this->max_target_rank_);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateRankSpecializationClusterPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<MergeRankSpecializationClusterOpsPattern,
                   RankSpecializationClusterPattern>(context);
}

void PopulateRankSpecializationToSCFPatterns(MLIRContext *context,
                                             OwningRewritePatternList *patterns,
                                             int64_t max_target_rank) {
  patterns->insert<LowerRankSpecializationClusterPattern>(context,
                                                          max_target_rank);
  shape::BroadcastOp::getCanonicalizationPatterns(*patterns, context);
  shape::ShapeOfOp::getCanonicalizationPatterns(*patterns, context);
  shape::AnyOp::getCanonicalizationPatterns(*patterns, context);
}

std::unique_ptr<FunctionPass> createRankSpecializationClusterPass() {
  return std::make_unique<RankSpecializationClusterPass>();
}

std::unique_ptr<FunctionPass> createRankSpecializationToSCFPass(
    int64_t max_target_rank) {
  return std::make_unique<RankSpecializationToSCFPass>(max_target_rank);
}

}  // namespace mhlo
}  // namespace mlir
