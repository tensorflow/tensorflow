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
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
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
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {

/// Needed to build `llvm::SmallSet`s and `llvm::EquivalenceClasses` of
/// `mlir::Value`s.
static bool operator<(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
}

namespace mhlo {

#define GEN_PASS_DEF_RANKSPECIALIZATIONCLUSTERPASS
#define GEN_PASS_DEF_RANKSPECIALIZATIONTOSCFPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

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

bool isClusterable(Operation *op) {
  if (!llvm::isa<InferShapedTypeOpInterface>(op)) return false;
  if (op->getNumOperands() == 0) return false;
  return (op->hasTrait<mlir::OpTrait::Elementwise>() &&
          op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) ||
         op->hasTrait<hlo::OpTrait::BroadcastingElementwise>();
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
    if (!isClusterable(op) || !llvm::any_of(op->getOperandTypes(), [](Type ty) {
          return ty.isa<UnrankedTensorType>();
        })) {
      return failure();
    }

    // Collect all collectively rank specializable ops.
    SmallVector<Operation *, 16> cluster;
    llvm::SmallSet<Value, 16> operandSet;
    llvm::SmallSet<Value, 16> resultSet;

    Operation *rootOp = op;
    while (rootOp->getNextNode() != nullptr &&
           isClusterable(rootOp->getNextNode()))
      rootOp = rootOp->getNextNode();

    Operation *it = rootOp;
    while (it != nullptr && isClusterable(it)) {
      // Find results that escape the cluster.
      for (OpOperand &use : it->getUses()) {
        if (!llvm::is_contained(cluster, use.getOwner()))
          resultSet.insert(use.get());
      }

      // Update cluster operands.
      for (OpResult v : it->getResults()) operandSet.erase(Value(v));
      for (OpOperand &v : it->getOpOperands()) operandSet.insert(v.get());

      cluster.push_back(it);
      it = it->getPrevNode();
    }

    // Create `RankSpecializationClusterOp`.
    auto operands = llvm::to_vector<16>(operandSet);
    auto results = llvm::to_vector<16>(resultSet);
    auto resultTypes = llvm::to_vector<16>(
        llvm::map_range(resultSet, [](Value v) { return v.getType(); }));
    Location loc = op->getLoc();
    auto clusterOp = rewriter.create<chlo::RankSpecializationClusterOp>(
        loc, resultTypes, operands);

    // Create body block.
    auto operandTypes = llvm::to_vector<16>(
        llvm::map_range(operandSet, [](Value v) { return v.getType(); }));
    Block *block =
        rewriter.createBlock(&clusterOp.getBody(), {}, operandTypes,
                             SmallVector<Location>(operandTypes.size(), loc));

    // Copy operations into the body.
    BlockAndValueMapping bvm;
    for (auto it : llvm::zip(operands, block->getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    rewriter.setInsertionPointToStart(block);
    for (Operation *it : llvm::reverse(cluster)) rewriter.clone(*it, bvm);

    // Create `RankSpecializationClusterYieldOp`.
    auto mappedResults = llvm::to_vector<16>(
        llvm::map_range(results, [&](Value v) { return bvm.lookup(v); }));
    rewriter.create<chlo::RankSpecializationClusterYieldOp>(loc, mappedResults);

    // Replace original ops with the new results.
    for (auto it : llvm::zip(results, clusterOp.getResults()))
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
    auto precedingOp =
        llvm::dyn_cast_or_null<chlo::RankSpecializationClusterOp>(
            op->getPrevNode());
    if (!precedingOp) return failure();
    Block *body = op.SingleBlock::getBody();
    Block *precedingBody = precedingOp.SingleBlock::getBody();
    auto yieldOp = llvm::dyn_cast<chlo::RankSpecializationClusterYieldOp>(
        op.SingleBlock::getBody()->getTerminator());
    auto precedingYieldOp =
        llvm::dyn_cast<chlo::RankSpecializationClusterYieldOp>(
            precedingOp.SingleBlock::getBody()->getTerminator());

    // Merge cluster operands. Consider only those operands of the second
    // cluster that do not originate in the preceding cluster.
    SmallVector<Value, 8> newOperands;
    for (Value v : precedingOp.getOperands()) newOperands.push_back(v);
    for (Value v : op.getOperands()) {
      if (v.getDefiningOp() != precedingOp &&
          !llvm::is_contained(precedingOp.getOperands(), v)) {
        newOperands.push_back(v);
      }
    }

    // Merge cluster results. Consider only those results of the preceding
    // cluster that are not exclusively used as operands to the second cluster.
    SmallVector<Value, 8> newUnmappedResults;
    for (auto it :
         llvm::zip(precedingOp.getResults(), precedingYieldOp.getResults())) {
      Value result, innerResult;
      std::tie(result, innerResult) = it;
      if (!llvm::all_of(result.getUsers(),
                        [&](Operation *user) { return user == op; })) {
        newUnmappedResults.push_back(innerResult);
      }
    }
    for (Value v : yieldOp.getResults()) newUnmappedResults.push_back(v);

    // Create merged cluster op.
    rewriter.setInsertionPoint(precedingOp);
    auto loc = op.getLoc();
    auto resultTypes = llvm::to_vector<16>(llvm::map_range(
        newUnmappedResults, [](Value v) { return v.getType(); }));
    auto newOp = rewriter.create<chlo::RankSpecializationClusterOp>(
        loc, resultTypes, newOperands);
    auto operandTypes = llvm::to_vector<16>(
        llvm::map_range(newOperands, [](Value v) { return v.getType(); }));
    Block *newBody =
        rewriter.createBlock(&newOp.getBody(), {}, operandTypes,
                             SmallVector<Location>(operandTypes.size(), loc));
    rewriter.setInsertionPointToStart(newBody);

    // Map operands and copy operations of the preceding cluster into the new
    // body.
    BlockAndValueMapping bvm;
    for (const auto &it : llvm::enumerate(precedingBody->getArguments()))
      bvm.map(it.value(), newBody->getArgument(it.index()));
    for (Operation &nestedOp : precedingBody->without_terminator())
      rewriter.clone(nestedOp, bvm);

    // Map operands and copy operations of the second cluster. If they result
    // from the preceeding cluster, we can simply map the corresponding value
    // internally.
    for (auto it : llvm::zip(body->getArguments(), op.getOperands())) {
      Value blockArg, operand;
      std::tie(blockArg, operand) = it;
      if (operand.getDefiningOp() == precedingOp) {
        auto where = llvm::find(precedingOp.getResults(), operand);
        assert(where.getBase() != nullptr && "expected to find ");
        bvm.map(blockArg,
                bvm.lookup(precedingYieldOp.getOperand(where.getIndex())));
      } else {
        auto where = llvm::find(newOp.getOperands(), operand);
        bvm.map(blockArg, newBody->getArgument(where.getIndex()));
      }
    }
    for (Operation &nestedOp : body->without_terminator()) {
      rewriter.clone(nestedOp, bvm);
    }

    // Yield inner results.
    rewriter.create<chlo::RankSpecializationClusterYieldOp>(
        loc,
        llvm::to_vector<16>(llvm::map_range(newUnmappedResults, [&](Value v) {
          return bvm.lookupOrDefault(v);
        })));

    // Replace the two cluster ops with the new corresponding results.
    SmallVector<Value, 8> precedingOpReplacements;
    int64_t i = 0;
    for (Value result : precedingOp.getResults()) {
      Value replacement = nullptr;
      if (!llvm::all_of(result.getUsers(),
                        [&](Operation *user) { return user == op; })) {
        replacement = newOp->getResult(i++);
      }
      precedingOpReplacements.push_back(replacement);
    }
    ValueRange opReplacements =
        newOp.getResults().take_back(op.getNumResults());
    rewriter.replaceOp(op, opReplacements);
    rewriter.replaceOp(precedingOp, precedingOpReplacements);

    return success();
  }
};

struct RankSpecializationClusterPass
    : public impl::RankSpecializationClusterPassBase<
          RankSpecializationClusterPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::ChloDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::populateRankSpecializationClusterPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

/// Lower rank specialization cluster to SCF.

bool isScalarTensorType(Type ty) {
  auto rankedTy = ty.dyn_cast<RankedTensorType>();
  return rankedTy && rankedTy.getRank() == 0;
}

bool isScalarShapeType(Type ty) {
  return ty.cast<RankedTensorType>().getDimSize(0) == 0;
}

Type deriveRankedTensorTypes(Type ty, int64_t rank) {
  auto tensorTy = ty.dyn_cast<TensorType>();
  if (!tensorTy) return ty;
  SmallVector<int64_t, 8> shape(rank, ShapedType::kDynamic);
  return RankedTensorType::get(shape, tensorTy.getElementType());
}

Type deriveUnrankedTensorTypes(Type ty) {
  if (auto rankedTy = ty.dyn_cast<RankedTensorType>())
    return UnrankedTensorType::get(rankedTy.getElementType());
  return ty;
}

SmallVector<Value, 8> materializeRankedOperations(
    OpBuilder &b, Location loc, BlockAndValueMapping &bvm,
    chlo::RankSpecializationClusterOp op) {
  // Create ranked operations.
  for (Operation &nestedOp : op.SingleBlock::getBody()->without_terminator()) {
    auto mappedOperands = llvm::to_vector<4>(llvm::map_range(
        nestedOp.getOperands(), [&](Value v) { return bvm.lookup(v); }));
    int64_t targetRank = 0;
    for (Value v : mappedOperands) {
      targetRank =
          std::max(targetRank, v.getType().cast<RankedTensorType>().getRank());
    }
    auto rankedResultTypes = llvm::to_vector<2>(
        llvm::map_range(nestedOp.getResultTypes(), [targetRank](Type ty) {
          return deriveRankedTensorTypes(ty, targetRank);
        }));
    OperationState rankedOpState(loc, nestedOp.getName().getStringRef(),
                                 mappedOperands, rankedResultTypes,
                                 nestedOp.getAttrs());
    Operation *rankedOp = b.create(rankedOpState);
    for (auto it : llvm::zip(nestedOp.getResults(), rankedOp->getResults()))
      bvm.map(std::get<0>(it), std::get<1>(it));
  }

  // Collect ranked results.
  auto yieldOp = llvm::cast<chlo::RankSpecializationClusterYieldOp>(
      op.SingleBlock::getBody()->getTerminator());
  return llvm::to_vector<8>(llvm::map_range(
      yieldOp.getResults(), [&](Value v) { return bvm.lookup(v); }));
}

SmallVector<Value, 8> materializeFinalReshape(
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op, ValueRange unshapedResults) {
  auto yieldOp = llvm::cast<chlo::RankSpecializationClusterYieldOp>(
      op.SingleBlock::getBody()->getTerminator());
  assert(unshapedResults.size() == 1 && yieldOp.getResults().size() == 1 &&
         "Currently, rank specialization supports only one result.");

  // Reify result shape.
  Operation *lastOpBeforeShapeReification = op->getPrevNode();
  SmallVector<Value, 1> resultShape;
  Value originalResult = yieldOp.getResults().front();
  auto originalResultIface =
      llvm::cast<InferShapedTypeOpInterface>(originalResult.getDefiningOp());
  if (failed(originalResultIface.reifyReturnTypeShapes(
          rewriter, originalResultIface->getOperands(), resultShape))) {
    return {};
  }

  // Materialize final reshape.
  Value unshapedResult = unshapedResults.front();
  Value result = rewriter.create<mhlo::DynamicReshapeOp>(
      loc, deriveUnrankedTensorTypes(unshapedResult.getType()), unshapedResult,
      resultShape.front());

  // Reify shapes until they are independent of operations in the original
  // cluster.
  {
    Operation *it = resultShape.front().getDefiningOp();
    while (it != nullptr && it != lastOpBeforeShapeReification) {
      bool advanced = false;
      if (auto shapeOfOp = llvm::dyn_cast<shape::ShapeOfOp>(it)) {
        Operation *def = shapeOfOp.getArg().getDefiningOp();
        if (def && def->getBlock() == op.SingleBlock::getBody()) {
          // Resolve `shape_of` op because it still depends on operation in the
          // original cluster.
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(shapeOfOp);
          SmallVector<Value, 1> tmpShape;
          auto iface = llvm::cast<InferShapedTypeOpInterface>(def);
          if (failed(iface.reifyReturnTypeShapes(rewriter, iface->getOperands(),
                                                 tmpShape)))
            return {};
          rewriter.replaceOp(shapeOfOp, tmpShape.front());

          // Continue, including the newly created operations.
          it = tmpShape.front().getDefiningOp();
          advanced = true;
        }
      }

      // Skip op, otherwise.
      if (!advanced) it = it->getPrevNode();
    }
  }

  // Replace all remaining uses of the original cluster's block args.
  for (auto it :
       llvm::zip(op.getOperands(), op.SingleBlock::getBody()->getArguments())) {
    Value operand, barg;
    std::tie(operand, barg) = it;
    barg.replaceUsesWithIf(operand, [&](OpOperand &operand) {
      return operand.getOwner()->getBlock() != op.SingleBlock::getBody();
    });
  }

  return {result};
}

Value materializeFlatShape(OpBuilder &b, Location loc, ValueRange sameShapes) {
  assert(!sameShapes.empty() && "Expected at least one shape.");
  Value shape = sameShapes.size() == 1
                    ? sameShapes.front()
                    : b.create<shape::AnyOp>(loc, sameShapes.front().getType(),
                                             sameShapes);
  return b.create<tensor::FromElementsOp>(
      loc,
      b.create<shape::NumElementsOp>(loc, b.getIndexType(), shape).getResult());
}

Value materializeScalarRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, ValueRange nonScalarsOfSameShape,
    function_ref<void(OpBuilder &, Location)> elseBuilderFn) {
  // Materialize predicate: All operands are scalars, except the expected
  // non-scalars.
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value allOthersAreScalar;
  for (auto it : llvm::zip(op.getOperands(), shapes)) {
    Value operand, shape;
    std::tie(operand, shape) = it;
    if (llvm::is_contained(nonScalarsOfSameShape, operand) ||
        isScalarTensorType(operand.getType())) {
      continue;
    }
    auto literal = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        b.create<shape::NumElementsOp>(loc, shape), one);
    allOthersAreScalar =
        allOthersAreScalar
            ? b.create<mlir::arith::AndIOp>(loc, allOthersAreScalar, literal)
                  .getResult()
            : literal.getResult();
  }

  auto ifOp = b.create<scf::IfOp>(
      loc, op->getResultTypes(), allOthersAreScalar,
      [&](OpBuilder &b, Location loc) {
        // Compute flat non-scalar shape.
        SmallVector<Value, 4> nonScalarShapes;
        for (auto it : llvm::zip(op.getOperands(), shapes)) {
          Value operand, shape;
          std::tie(operand, shape) = it;
          if (llvm::is_contained(nonScalarsOfSameShape, operand))
            nonScalarShapes.push_back(shape);
        }
        Value flatShape = materializeFlatShape(b, loc, nonScalarShapes);

        // Derive ranked operands.
        auto rankedOperands = llvm::to_vector<8>(
            llvm::map_range(op.getOperands(), [&](Value v) -> Value {
              if (isScalarTensorType(v.getType())) return v;
              if (!llvm::is_contained(nonScalarsOfSameShape, v)) {
                return b
                    .create<mhlo::ReshapeOp>(
                        loc, deriveRankedTensorTypes(v.getType(), /*rank=*/0),
                        v)
                    .getResult();
              }
              return b
                  .create<mhlo::DynamicReshapeOp>(
                      loc, deriveRankedTensorTypes(v.getType(), /*rank=*/1), v,
                      flatShape)
                  .getResult();
            }));

        // Materialize ranked variants for the element-wise operations.
        BlockAndValueMapping bvm;
        for (auto it : llvm::zip(op.SingleBlock::getBody()->getArguments(),
                                 rankedOperands))
          bvm.map(std::get<0>(it), std::get<1>(it));
        Value unshapedResult =
            materializeRankedOperations(b, loc, bvm, op).front();

        // Return as unranked tensor for compatibility with the other cases.
        b.create<scf::YieldOp>(
            loc, b.create<tensor::CastOp>(
                      loc, deriveUnrankedTensorTypes(unshapedResult.getType()),
                      unshapedResult)
                     .getDest());
      },
      elseBuilderFn);

  return ifOp.getResults().front();
}

Value materializeEqualShapesRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes,
    function_ref<void(OpBuilder &, Location)> elseBuilderFn) {
  // Materialize all shapes equal predicate.
  Value allShapesEqOrScalar;
  auto nonScalarShapes = llvm::to_vector<8>(llvm::make_filter_range(
      shapes, [](Value v) { return !isScalarShapeType(v.getType()); }));
  assert(
      nonScalarShapes.size() >= 2 &&
      "Equal shapes strategy requires at least two non-scalar operand shapes.");
  for (Value s : llvm::drop_begin(nonScalarShapes)) {
    auto literal = b.create<shape::ShapeEqOp>(loc, nonScalarShapes.front(), s);
    allShapesEqOrScalar =
        allShapesEqOrScalar
            ? b.create<mlir::arith::AndIOp>(loc, allShapesEqOrScalar, literal)
                  .getResult()
            : literal;
  }

  auto ifOp = b.create<scf::IfOp>(
      loc, op->getResultTypes(), allShapesEqOrScalar,
      [&](OpBuilder &b, Location loc) {
        // Flatten non-scalar operands.
        Value flatShape = materializeFlatShape(b, loc, nonScalarShapes);
        auto flatOperands = llvm::to_vector<8>(
            llvm::map_range(op.getOperands(), [&](Value v) -> Value {
              if (isScalarTensorType(v.getType())) return v;
              return b.create<mhlo::DynamicReshapeOp>(
                  loc, deriveRankedTensorTypes(v.getType(), /*rank=*/1), v,
                  flatShape);
            }));

        // Materialize ranked variants for the element-wise operations.
        BlockAndValueMapping bvm;
        for (auto it :
             llvm::zip(op.SingleBlock::getBody()->getArguments(), flatOperands))
          bvm.map(std::get<0>(it), std::get<1>(it));
        Value unshapedResult =
            materializeRankedOperations(b, loc, bvm, op).front();

        // Return as unranked tensor for compatibility with the other cases.
        b.create<scf::YieldOp>(
            loc, b.create<tensor::CastOp>(
                      loc, deriveUnrankedTensorTypes(unshapedResult.getType()),
                      unshapedResult)
                     .getDest());
      },
      elseBuilderFn);

  return ifOp.getResults().front();
}

Value materializeTargetRankSpecializationCase(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t targetRank) {
  // Reshape unranked operands to match the target rank.
  RankedTensorType extentTensorTy =
      shape::getExtentTensorType(b.getContext(), targetRank);
  Value allOnesShape = b.create<shape::ConstShapeOp>(
      loc, extentTensorTy,
      mlir::DenseIntElementsAttr::get(extentTensorTy,
                                      SmallVector<int64_t, 6>(targetRank, 1)));
  SmallVector<Value, 8> rankedOperands;
  for (auto it : llvm::zip(op.getOperands(), shapes)) {
    Value operand, shape;
    std::tie(operand, shape) = it;
    if (operand.getType().isa<RankedTensorType>()) {
      rankedOperands.push_back(operand);
      continue;
    }
    Value rankedShape = b.create<tensor::CastOp>(
        loc, extentTensorTy,
        b.create<shape::BroadcastOp>(loc,
                                     shape::getExtentTensorType(b.getContext()),
                                     shape, allOnesShape,
                                     /*error=*/nullptr));
    rankedOperands.push_back(b.create<mhlo::DynamicReshapeOp>(
        loc, deriveRankedTensorTypes(operand.getType(), targetRank), operand,
        rankedShape));
  }

  // Materialize ranked versions of the element-wise operations.
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(op.getBody().front().getArguments(), rankedOperands))
    bvm.map(std::get<0>(it), std::get<1>(it));

  // Return as unranked for compatibility with other target ranks.
  auto unshapedResult = materializeRankedOperations(b, loc, bvm, op).front();
  return b.create<tensor::CastOp>(
      loc, deriveUnrankedTensorTypes(unshapedResult.getType()), unshapedResult);
}

Value recusivelyMaterializeTargetRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, Value maxRank, int64_t minTargetRank,
    int64_t maxTargetRank) {
  Value condition = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ule, maxRank,
      b.create<arith::ConstantIndexOp>(loc, minTargetRank));

  // If only a unique target rank is left, we can lower to an assert instead
  // of the usual if operation.
  if (minTargetRank == maxTargetRank) {
    b.create<cf::AssertOp>(
        loc, condition,
        "Input for dynamic binary or n-ary op lowering was of "
        "a rank greater than " +
            std::to_string(maxTargetRank));
    return materializeTargetRankSpecializationCase(b, loc, op, shapes,
                                                   minTargetRank);
  }

  // Materialize IR for the smallest considered target rank.
  auto ifOp = b.create<scf::IfOp>(loc, op->getResultTypes(), condition,
                                  /*withElseRegion=*/true);
  auto thenBuilder = ifOp.getThenBodyBuilder();
  thenBuilder.create<scf::YieldOp>(
      loc, materializeTargetRankSpecializationCase(thenBuilder, loc, op, shapes,
                                                   minTargetRank));

  // Recurse for all remaining target ranks.
  auto elseBuilder = ifOp.getElseBodyBuilder();
  elseBuilder.create<scf::YieldOp>(
      loc, recusivelyMaterializeTargetRankSpecializationCases(
               elseBuilder, loc, op, shapes, maxRank, minTargetRank + 1,
               maxTargetRank));

  return ifOp.getResults().front();
}

Value materializeGenericRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t maxTargetRank) {
  // Get the minimum broadcast shapes of the operands.
  auto nonScalarShapes = llvm::to_vector<8>(llvm::make_filter_range(
      shapes, [](Value v) { return !isScalarShapeType(v.getType()); }));
  auto minBcastShapesOp = b.create<chlo::MinimumBroadcastShapesOp>(
      loc,
      SmallVector<Type, 8>(nonScalarShapes.size(),
                           shape::getExtentTensorType(b.getContext())),
      nonScalarShapes);

  // Find the maximum rank among the reduced operand shapes.
  Value maxRank;
  for (Value shape : minBcastShapesOp.getResults()) {
    Value rank = b.create<shape::RankOp>(loc, b.getIndexType(), shape);
    if (!maxRank) {
      maxRank = rank;
    } else {
      maxRank = b.create<mlir::arith::SelectOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, maxRank,
                                  rank),
          maxRank, rank);
    }
  }

  // Collect reduced shapes.
  SmallVector<Value, 8> reducedShapes;
  auto it = minBcastShapesOp.result_begin();
  for (Value s : shapes) {
    if (isScalarShapeType(s.getType())) {
      reducedShapes.push_back(s);
    } else {
      reducedShapes.push_back(*it++);
    }
  }

  // Materialize rank specialization for ranks 1, ...
  return recusivelyMaterializeTargetRankSpecializationCases(
      b, loc, op, reducedShapes, maxRank, /*minTargetRank=*/1, maxTargetRank);
}

Value materializeDefaultRankSpecializationCases(
    OpBuilder &b, Location loc, chlo::RankSpecializationClusterOp op,
    const SmallVector<Value, 8> &shapes, int64_t maxTargetRank) {
  return materializeEqualShapesRankSpecializationCase(
      b, loc, op, shapes, [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, materializeGenericRankSpecializationCases(
                                        b, loc, op, shapes, maxTargetRank));
      });
}

SmallVector<Value, 8>
materializeRankSpecializationForSingleNonScalarShapeEquivalenceClass(
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op, ValueRange nonScalarsOfSameShape) {
  // Compute flat operand shape.
  auto nonScalarShapes =
      llvm::to_vector<4>(llvm::map_range(nonScalarsOfSameShape, [&](Value v) {
        return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
      }));
  Value flatShape = materializeFlatShape(rewriter, loc, nonScalarShapes);

  // Materialize ranked variants for the element-wise operations.
  BlockAndValueMapping bvm;
  for (auto it :
       llvm::zip(op.SingleBlock::getBody()->getArguments(), op.getOperands())) {
    Value operand;
    Value bbArg;
    std::tie(bbArg, operand) = it;
    if (!isScalarTensorType(operand.getType())) {
      assert(llvm::is_contained(nonScalarsOfSameShape, operand) &&
             "Expected all non-scalars in the same shape equivalence class.");
      operand = rewriter.create<mhlo::DynamicReshapeOp>(
          loc, deriveRankedTensorTypes(operand.getType(), /*rank=*/1), operand,
          flatShape);
    }
    bvm.map(bbArg, operand);
  }
  SmallVector<Value, 8> unshapedResults =
      materializeRankedOperations(rewriter, loc, bvm, op);

  // Restore the results' expected shape.
  Value shape = nonScalarShapes.front();
  return llvm::to_vector<8>(
      llvm::map_range(unshapedResults, [&](Value v) -> Value {
        return rewriter.create<mhlo::DynamicReshapeOp>(
            loc, deriveUnrankedTensorTypes(v.getType()), v, shape);
      }));
}

Value materializeRankSpecializationForTwoNonScalarShapeEquivalenceClasses(
    PatternRewriter &rewriter, Location loc,
    chlo::RankSpecializationClusterOp op,
    SmallVector<SmallVector<Value, 4>, 4> nonScalarEqs, int64_t maxTargetRank) {
  assert(nonScalarEqs.size() == 2 &&
         "Expect two non-scalar equivalence classes.");
  auto shapes =
      llvm::to_vector<8>(llvm::map_range(op.getOperands(), [&](Value v) {
        return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
      }));
  ValueRange lhsNonScalarEqs = nonScalarEqs[0];
  ValueRange rhsNonScalarEqs = nonScalarEqs[1];

  // Materialize all the different cases.
  Value unshapedResult = materializeScalarRankSpecializationCase(
      rewriter, loc, op, shapes, rhsNonScalarEqs,
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(
            loc, materializeScalarRankSpecializationCase(
                     b, loc, op, shapes, lhsNonScalarEqs,
                     [&](OpBuilder &b, Location loc) {
                       b.create<scf::YieldOp>(
                           loc, materializeDefaultRankSpecializationCases(
                                    b, loc, op, shapes, maxTargetRank));
                     }));
      });

  // Materialize final reshape once and for all rank specialization cases.
  return materializeFinalReshape(rewriter, loc, op, unshapedResult).front();
}

// Materialize rank generic rank specialization.
Value materializeDefaultRankSpecialization(PatternRewriter &rewriter,
                                           Location loc,
                                           chlo::RankSpecializationClusterOp op,
                                           int64_t maxTargetRank) {
  auto shapes =
      llvm::to_vector<8>(llvm::map_range(op.getOperands(), [&](Value v) {
        return rewriter.create<shape::ShapeOfOp>(loc, v).getResult();
      }));

  // Materialize all the different cases.
  Value unshapedResult = materializeDefaultRankSpecializationCases(
      rewriter, loc, op, shapes, maxTargetRank);

  // Materialize final reshape once and for all rank specialization cases.
  return materializeFinalReshape(rewriter, loc, op, unshapedResult).front();
}

// This is a very limited form of shape inference. It is correct but incomplete.
SmallVector<SmallVector<Value, 4>, 4> findNonScalarShapeEquivalences(
    chlo::RankSpecializationClusterOp op) {
  llvm::EquivalenceClasses<Value> eqs;

  // Bridge the equivalences between operands and block arguments.
  for (auto it :
       llvm::zip(op.getOperands(), op.SingleBlock::getBody()->getArguments()))
    eqs.unionSets(std::get<0>(it), std::get<1>(it));

  // Find equalities through `SameOperandsAndResultShape` trait.
  auto unionSets = [&](ValueRange vs) {
    if (vs.empty()) return;
    Value repr = vs.front();
    for (Value v : vs.drop_front()) eqs.unionSets(repr, v);
  };
  for (Operation &nestedOp : op.SingleBlock::getBody()->without_terminator()) {
    if (nestedOp.hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      unionSets(nestedOp.getOperands());
      unionSets(nestedOp.getResults());
      if (!nestedOp.getOperands().empty() && !nestedOp.getResults().empty())
        eqs.unionSets(nestedOp.getResult(0), nestedOp.getOperand(0));
    }
  }

  // Find shape equalities through surrounding constraints.
  if (auto assumingOp = op->getParentOfType<shape::AssumingOp>()) {
    SmallVector<Operation *, 8> queue;
    auto appendIfNotNull = [&](Operation *op) {
      if (op != nullptr) queue.push_back(op);
    };
    appendIfNotNull(assumingOp.getWitness().getDefiningOp());
    while (!queue.empty()) {
      Operation *it = queue.pop_back_val();
      if (auto assumingAllOp = llvm::dyn_cast<shape::AssumingAllOp>(it)) {
        for (Value v : assumingAllOp.getInputs())
          appendIfNotNull(v.getDefiningOp());
      } else if (auto cstrEqOp = llvm::dyn_cast<shape::CstrEqOp>(it)) {
        Value refArg;
        for (Value v : cstrEqOp.getShapes()) {
          if (auto shapeOfOp =
                  dyn_cast_or_null<shape::ShapeOfOp>(v.getDefiningOp())) {
            if (!refArg) {
              refArg = shapeOfOp.getArg();
            } else {
              eqs.unionSets(refArg, shapeOfOp.getArg());
            }
          }
        }
      }
    }
  }

  // Find equalities through special knowledge of ops.
  // TODO(frgossen): Remove this when these shape equalities can be inferred
  // from surrounding shape constraints.
  for (Operation &nestedOp : op.SingleBlock::getBody()->without_terminator()) {
    if (auto selectOp = llvm::dyn_cast<mhlo::SelectOp>(nestedOp)) {
      unionSets(
          {selectOp.getOnTrue(), selectOp.getOnFalse(), selectOp.getResult()});
    } else if (auto clampOp = llvm::dyn_cast<mhlo::ClampOp>(nestedOp)) {
      unionSets({clampOp.getOperand(), clampOp.getResult()});
    }
  }

  // Convert to a list-like equivalence class representation.
  SmallVector<SmallVector<Value, 4>, 4> nonScalarEqs;
  for (Value v : op.getOperands()) {
    if (isScalarTensorType(v.getType())) continue;
    bool inserted = false;
    for (auto &eqClass : nonScalarEqs) {
      if (eqs.isEquivalent(eqClass.front(), v)) {
        eqClass.push_back(v);
        inserted = true;
        break;
      }
    }
    if (!inserted) nonScalarEqs.push_back(SmallVector<Value, 4>({v}));
  }

  return nonScalarEqs;
}

struct LowerRankSpecializationClusterPattern
    : public OpRewritePattern<chlo::RankSpecializationClusterOp> {
  LowerRankSpecializationClusterPattern(MLIRContext *ctx, int64_t maxTargetRank)
      : OpRewritePattern<chlo::RankSpecializationClusterOp>(ctx, /*benefit=*/1),
        maxTargetRank(maxTargetRank) {}

  LogicalResult matchAndRewrite(chlo::RankSpecializationClusterOp op,
                                PatternRewriter &rewriter) const override {
    // Restoring the result shape currently relies on all operands being used
    // for a single result. The result shape is then the broadcasted shape of
    // all operands.
    if (op.getNumResults() != 1) return failure();

    // If there is only a single non-scalar shape equivalence class, we can
    // flatten that operands completely.
    SmallVector<SmallVector<Value, 4>, 4> nonScalarEqs =
        findNonScalarShapeEquivalences(op);
    Location loc = op.getLoc();
    if (nonScalarEqs.size() == 1) {
      rewriter.replaceOp(
          op,
          materializeRankSpecializationForSingleNonScalarShapeEquivalenceClass(
              rewriter, loc, op, nonScalarEqs.front()));
      return success();
    }

    // If there are exactly two non-scalar shape equivalence classes, we can
    // consider two extra cases: If either of the operand classes turns out to
    // be all-scalars at runtime, we can, again, flatten all operands.
    if (nonScalarEqs.size() == 2) {
      rewriter.replaceOp(
          op,
          materializeRankSpecializationForTwoNonScalarShapeEquivalenceClasses(
              rewriter, loc, op, nonScalarEqs, maxTargetRank));
      return success();
    }

    // For all other cases, reshape the operands to match in rank, apply the
    // operation, and restore the expected shape.
    rewriter.replaceOp(op, materializeDefaultRankSpecialization(
                               rewriter, loc, op, maxTargetRank));
    return success();
  }

 private:
  int64_t maxTargetRank;
};

struct RankSpecializationToSCFPass
    : public impl::RankSpecializationToSCFPassBase<
          RankSpecializationToSCFPass> {
  explicit RankSpecializationToSCFPass(int64_t maxTargetRank)
      : RankSpecializationToSCFPassBase<
            RankSpecializationToSCFPass>::RankSpecializationToSCFPassBase() {
    this->max_target_rank_ = maxTargetRank;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::ChloDialect, func::FuncDialect,
                    shape::ShapeDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateRankSpecializationToSCFPatterns(ctx, &patterns,
                                            this->max_target_rank_);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateRankSpecializationClusterPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns) {
  patterns->add<MergeRankSpecializationClusterOpsPattern,
                RankSpecializationClusterPattern>(context);
}

void populateRankSpecializationToSCFPatterns(MLIRContext *context,
                                             RewritePatternSet *patterns,
                                             int64_t maxTargetRank) {
  patterns->add<LowerRankSpecializationClusterPattern>(context, maxTargetRank);
  shape::BroadcastOp::getCanonicalizationPatterns(*patterns, context);
  shape::ShapeOfOp::getCanonicalizationPatterns(*patterns, context);
  shape::AnyOp::getCanonicalizationPatterns(*patterns, context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createRankSpecializationClusterPass() {
  return std::make_unique<RankSpecializationClusterPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createRankSpecializationToSCFPass(
    int64_t maxTargetRank) {
  return std::make_unique<RankSpecializationToSCFPass>(maxTargetRank);
}

}  // namespace mhlo
}  // namespace mlir
