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

#include <algorithm>
#include <utility>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// To avoid duplicate broadcasts, we collect all the intended broadcasts ahead
// of realizing any broadcasts in the IR. These are broadcasted versions of
// values that we are interested in, and they are uniquely characterized by a
// `BroadcastIntent` value.
struct BroadcastIntent {
  RankedTensorType resultType;
  Value targetValue;
  Value outputDimensions;
  Attribute broadcastDimensions;
  bool operator==(BroadcastIntent rhs) const {
    return resultType == rhs.resultType && targetValue == rhs.targetValue &&
           outputDimensions == rhs.outputDimensions &&
           broadcastDimensions == rhs.broadcastDimensions;
  }
  bool operator!=(BroadcastIntent rhs) const { return !(*this == rhs); }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::mhlo::BroadcastIntent> {
  static mlir::mhlo::BroadcastIntent getEmptyKey() {
    return {DenseMapInfo<mlir::RankedTensorType>::getEmptyKey(),
            DenseMapInfo<mlir::Value>::getEmptyKey(),
            DenseMapInfo<mlir::Value>::getEmptyKey(),
            DenseMapInfo<mlir::Attribute>::getEmptyKey()};
  }
  static mlir::mhlo::BroadcastIntent getTombstoneKey() {
    return {DenseMapInfo<mlir::RankedTensorType>::getTombstoneKey(),
            DenseMapInfo<mlir::Value>::getTombstoneKey(),
            DenseMapInfo<mlir::Value>::getTombstoneKey(),
            DenseMapInfo<mlir::Attribute>::getTombstoneKey()};
  }
  static unsigned getHashValue(const mlir::mhlo::BroadcastIntent &intent) {
    return hash_combine(
        DenseMapInfo<mlir::RankedTensorType>::getHashValue(intent.resultType),
        DenseMapInfo<mlir::Value>::getHashValue(intent.targetValue),
        DenseMapInfo<mlir::Value>::getHashValue(intent.outputDimensions),
        DenseMapInfo<mlir::Attribute>::getHashValue(
            intent.broadcastDimensions));
  }
  static bool isEqual(const mlir::mhlo::BroadcastIntent &lhs,
                      const mlir::mhlo::BroadcastIntent &rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

namespace mlir {
namespace mhlo {
namespace {

bool allowsForElementwiseBroadcastPropagation(Operation *op) {
  if (op && op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() &&
      op->hasTrait<mlir::OpTrait::Elementwise>() && op->getNumResults() == 1) {
    return true;
  }
  if (op && op->hasTrait<mlir::mhlo::OpTrait::BroadcastingElementwise>() &&
      op->getNumResults() == 1) {
    return true;
  }
  return false;
}

bool allowsForBroadcastPropagation(Operation *op) {
  return llvm::isa_and_nonnull<DynamicBroadcastInDimOp>(op) ||
         allowsForElementwiseBroadcastPropagation(op);
}

DenseIntElementsAttr composeBroadcastDimensionsAttr(OpBuilder &builder,
                                                    DenseIntElementsAttr a,
                                                    DenseIntElementsAttr b) {
  SmallVector<int64_t> bVec =
      llvm::to_vector(llvm::map_range(b, [](const APInt &it) {
        return static_cast<int64_t>(it.getLimitedValue());
      }));
  SmallVector<int64_t> composedVec = llvm::to_vector(llvm::map_range(
      a, [bVec](const APInt &it) { return bVec[it.getLimitedValue()]; }));
  return builder.getI64TensorAttr(composedVec);
}

// Find all the broadcast intents and their dependencies. Start analyzing from
// the root an collect all broadcast intents that can help broadcast propagation
// from there.
void findBroadcastIntents(
    DynamicBroadcastInDimOp root, Block *parentBlock,
    BroadcastIntent &rootBcastIntent,
    SmallVector<BroadcastIntent> &bcastIntents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcastIntentDependencies) {
  OpBuilder builder(root.getContext());

  // Use the result vector of broadcast intents as a worklist. The set of
  // broadcast intents helps to ensure their uniqueness.
  DenseSet<BroadcastIntent> bcastIntentsSet;
  auto addToWorklistIfNew = [&](BroadcastIntent bcastIntent) {
    if (!bcastIntentsSet.count(bcastIntent)) {
      bcastIntentsSet.insert(bcastIntent);
      bcastIntents.push_back(bcastIntent);
    }
  };

  // Derive the broadcast intent associated with the root broadcast operation.
  // Add it to the worklist to seed the analysis.
  rootBcastIntent = {root.getResult().getType().cast<RankedTensorType>(),
                     root.operand(), root.output_dimensions(),
                     root.broadcast_dimensions()};
  addToWorklistIfNew(rootBcastIntent);

  // We use result vector of broadcast intents as a worklist, the first `i`
  // intents of which have been processed.
  for (int i = 0; i < bcastIntents.size(); ++i) {
    BroadcastIntent it = bcastIntents[i];
    Operation *producerOp = it.targetValue.getDefiningOp();

    // We can propagate broadcasts over (broadcasting) element-wise operations
    // and dynamic_broadcast_in_dim ops with the restriction that they must be
    // in the same block as they may depend on assuming regions.
    if (!producerOp || producerOp->getBlock() != parentBlock ||
        !allowsForBroadcastPropagation(producerOp)) {
      continue;
    }

    // We can skip broadcasting producers (dynamic_broadcast_in_dim ops) if we
    // compose their broadcasting dimensions.
    if (auto producerBcastOp =
            llvm::dyn_cast<DynamicBroadcastInDimOp>(producerOp)) {
      DenseIntElementsAttr composedBcastDims = composeBroadcastDimensionsAttr(
          builder, producerBcastOp.broadcast_dimensions(),
          it.broadcastDimensions.cast<DenseIntElementsAttr>());
      BroadcastIntent bcastedOperandIntent = {
          it.resultType, producerBcastOp.operand(), it.outputDimensions,
          composedBcastDims};

      // Record dependency and "recur".
      bcastIntentDependencies[it] = {bcastedOperandIntent};
      addToWorklistIfNew(bcastedOperandIntent);
      continue;
    }

    // We can propagate broadcasts over (broadcasting) element-wise operations.
    // Instead of broadcasting the result of such an op, we can broadcast the
    // operands and apply the element-wise operation to them.
    assert(allowsForElementwiseBroadcastPropagation(producerOp));
    bcastIntentDependencies[it] = {};
    for (auto operand : producerOp->getOperands()) {
      auto operandTy = operand.getType().cast<RankedTensorType>();
      auto operandBcastDims = operandTy.getRank() == 0
                                  ? builder.getI64TensorAttr({})
                                  : it.broadcastDimensions;
      auto bcastedOperandTy = RankedTensorType::get(it.resultType.getShape(),
                                                    operandTy.getElementType());
      BroadcastIntent bcastedOperandIntent = {
          bcastedOperandTy, operand, it.outputDimensions, operandBcastDims};

      // Record dependency and "recur".
      bcastIntentDependencies[it].push_back(bcastedOperandIntent);
      addToWorklistIfNew(bcastedOperandIntent);
    }
  }
}

void sortBroadcastIntentsInReverseTopologicalOrder(
    SmallVector<BroadcastIntent> &bcastIntentsVec, Block *parentBlock) {
  // Sort broadcast intents in reverse topological order of the producer ops. We
  // can use the positions in the block for this. All broadcast intents outside
  // the block (e.g. arguments) will be sorted towards the front.
  // This ordering is independent of the output dimensions as dependencies can
  // only occur between broadcast intents of the same output dimension.
  std::sort(bcastIntentsVec.begin(), bcastIntentsVec.end(),
            [parentBlock](const BroadcastIntent &a, const BroadcastIntent &b) {
              Operation *producerOpA = a.targetValue.getDefiningOp();
              Operation *producerOpB = b.targetValue.getDefiningOp();
              bool aInBlock = producerOpA != nullptr &&
                              producerOpA->getBlock() == parentBlock;
              bool bInBlock = producerOpB != nullptr &&
                              producerOpB->getBlock() == parentBlock;
              if (aInBlock && bInBlock) {
                return producerOpA->isBeforeInBlock(producerOpB);
              }
              return !aInBlock && bInBlock;
            });
}

void setInsertionPointToEarliestPointWithAllValuesAvailable(
    PatternRewriter &rewriter, Block *block, ValueRange values) {
  Operation *lastDef = nullptr;
  for (Value v : values) {
    Operation *def = v.getDefiningOp();
    if (def && def->getBlock() == block) {
      if (!lastDef || lastDef->isBeforeInBlock(def)) lastDef = def;
    }
  }
  if (lastDef) {
    rewriter.setInsertionPointAfter(lastDef);
  } else {
    rewriter.setInsertionPointToStart(block);
  }
}

DenseMap<BroadcastIntent, Value> realizeBroadcastIntents(
    SmallVector<BroadcastIntent> &sortedBcastIntents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcastIntentDependencies,
    Block *parentBlock, PatternRewriter &rewriter) {
  // Realize broadcast intents in order. They must be sorted so that their
  // dependencies are realized before them.
  DenseMap<BroadcastIntent, Value> realizations;
  for (auto it : sortedBcastIntents) {
    Operation *producerOp = it.targetValue.getDefiningOp();
    assert(!realizations.count(it) && "expect unrealized broadcast intent");
    auto deps = bcastIntentDependencies.find(it);

    // If we cannot propagate broadcasts further, materialize them as a
    // dynamic_broadcast_in_dim op.
    if (!producerOp || producerOp->getBlock() != parentBlock ||
        !allowsForBroadcastPropagation(producerOp)) {
      assert(deps == bcastIntentDependencies.end() && "expect no dependencies");
      setInsertionPointToEarliestPointWithAllValuesAvailable(
          rewriter, parentBlock,
          ValueRange{it.targetValue, it.outputDimensions});
      realizations[it] = rewriter.create<DynamicBroadcastInDimOp>(
          it.targetValue.getLoc(), it.resultType, it.targetValue,
          it.outputDimensions,
          it.broadcastDimensions.cast<DenseIntElementsAttr>());
      continue;
    }

    // For broadcast propagation across dynamic_broadcast_in_dim ops, the
    // broadcasted value is already materialized. Forward it.
    if (auto producerBcastOp =
            llvm::dyn_cast_or_null<DynamicBroadcastInDimOp>(producerOp)) {
      assert(deps != bcastIntentDependencies.end() &&
             deps->second.size() == 1 && "expect one dependency");
      auto bcastedOperand = realizations.find(deps->second.front());
      assert(bcastedOperand != realizations.end());
      realizations[it] = Value(bcastedOperand->second);
      continue;
    }

    // Othwerwise, realize broadcast intent for a (broadcasting) element-wise
    // operation based on the broadcasted operands.
    assert(allowsForElementwiseBroadcastPropagation(producerOp) &&
           "expect broadcast propagation over an (broadcasting) element-wise "
           "operation");
    assert(deps != bcastIntentDependencies.end() &&
           deps->second.size() == producerOp->getNumOperands() &&
           "expect one dependency per operand");
    auto bcastedOperands = llvm::to_vector(
        llvm::map_range(deps->second, [&](BroadcastIntent operandIntent) {
          auto bcastedOperand = realizations.find(operandIntent);
          assert(bcastedOperand != realizations.end() &&
                 "expect dependencies to be realized earlier");
          return bcastedOperand->second;
        }));
    setInsertionPointToEarliestPointWithAllValuesAvailable(
        rewriter, parentBlock, bcastedOperands);
    OperationState newProducerOpState(
        producerOp->getLoc(), producerOp->getName().getStringRef(),
        bcastedOperands, it.resultType, producerOp->getAttrs());
    Operation *newProducerOp = rewriter.create(newProducerOpState);
    assert(newProducerOp->getNumResults() == 1 && "expect exactly one result");
    realizations[it] = newProducerOp->getResults().front();
  }

  return realizations;
}

void transitivelyEraseUnusedSideEffectFreeOps(Operation *root,
                                              PatternRewriter &rewriter) {
  // Find ops to erase.
  SmallPtrSet<Operation *, 16> opsToEraseSet;
  SmallVector<Operation *, 16> opsToErase;
  SmallVector<Operation *, 16> worklist = {root};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // Erase ops only once.
    if (opsToEraseSet.count(op)) continue;

    // Erase only operations that are unused and free of side effects.
    if (!MemoryEffectOpInterface::hasNoEffect(op) ||
        !llvm::all_of(op->getUsers(), [opsToEraseSet](Operation *user) {
          return opsToEraseSet.count(user);
        })) {
      continue;
    }

    // Erase and "recur".
    opsToEraseSet.insert(op);
    opsToErase.push_back(op);
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) worklist.push_back(def);
    }
  }

  // Finally, erase the ops in the order of their uses.
  for (Operation *op : opsToErase) rewriter.eraseOp(op);
}

LogicalResult propagateBroadcast(DynamicBroadcastInDimOp root,
                                 Block *parentBlock,
                                 PatternRewriter &rewriter) {
  // We can move broadcasts up over (i) (broadcasting) element-wise operations
  // and (i) dynamic_broadcast_in_dim ops. This way, we propagate them through
  // the IR to perform them early. Instead of broadcasting the result of such an
  // op, we can broadcast the operands and apply the element-wise operation to
  // them.
  //
  // To avoid exponential growth of the IR, we will do this in two phases:
  //   1) First, we collect all the unique broadcast intents. These are
  //      broadcasted versions of values that we are interested in. They may
  //      later be materialized as an explicit broadcast or they can be the
  //      direct result of an operation over which a broadcast was propagated.
  //   2) Then, we fulfill every broadcast intent in reverse topological order
  //      to ensure that their dependencies (the broadcasted operands) are
  //      available.

  // Find the unique broadcast intents.
  BroadcastIntent rootBcastIntent;
  SmallVector<BroadcastIntent> bcastIntents;
  DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
      bcastIntentDependencies;
  findBroadcastIntents(root, parentBlock, rootBcastIntent, bcastIntents,
                       bcastIntentDependencies);

  // Fail if there is nothing but the root intent, i.e. if there is nothing to
  // rewrite here.
  if (bcastIntents.size() <= 1) {
    assert(bcastIntents.front() == rootBcastIntent && "expect root intent");
    return failure();
  }

  // Sort the broadcast intents in reverse topological order so that they can be
  // materialized and every depency is available when needed.
  sortBroadcastIntentsInReverseTopologicalOrder(bcastIntents, parentBlock);

  // Realize broadcast intents.
  DenseMap<BroadcastIntent, Value> realizations = realizeBroadcastIntents(
      bcastIntents, bcastIntentDependencies, parentBlock, rewriter);

  // Find the operations that may become redundant after replacing the root
  // operation. This allows us to transitively erase unused side effect-free
  // operations that result from this rewrite (after the root operation is no
  // longer accessible).
  SmallVector<Operation *> possiblyUnused;
  for (auto operand : root->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) possiblyUnused.push_back(def);
  }

  // Replace the root operation with its broadcast intent's realization.
  rewriter.replaceOp(root, realizations[rootBcastIntent]);

  // Erase all the operations that have become redundant as a result of this
  // rewrite.
  for (Operation *op : possiblyUnused) {
    transitivelyEraseUnusedSideEffectFreeOps(op, rewriter);
  }

  return success();
}

struct BroadcastPropagationPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern<DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    return propagateBroadcast(op, op->getBlock(), rewriter);
  }
};

struct BroadcastPropagationPass
    : public BroadcastPropagationPassBase<BroadcastPropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Collect patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<BroadcastPropagationPattern>(ctx);

    // Apply broadcast propagation in reverse order to start propagation at
    // the root of broadcast chains. This avoids duplicate work.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = false;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createBroadcastPropagationPass() {
  return std::make_unique<BroadcastPropagationPass>();
}

}  // namespace mhlo
}  // namespace mlir
