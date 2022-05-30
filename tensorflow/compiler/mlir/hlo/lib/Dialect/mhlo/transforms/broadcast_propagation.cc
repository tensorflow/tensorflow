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
  RankedTensorType result_type;
  Value target_value;
  Value output_dimensions;
  Attribute broadcast_dimensions;
  bool operator==(BroadcastIntent rhs) const {
    return result_type == rhs.result_type && target_value == rhs.target_value &&
           output_dimensions == rhs.output_dimensions &&
           broadcast_dimensions == rhs.broadcast_dimensions;
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
        DenseMapInfo<mlir::RankedTensorType>::getHashValue(intent.result_type),
        DenseMapInfo<mlir::Value>::getHashValue(intent.target_value),
        DenseMapInfo<mlir::Value>::getHashValue(intent.output_dimensions),
        DenseMapInfo<mlir::Attribute>::getHashValue(
            intent.broadcast_dimensions));
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

bool AllowsForElementwiseBroadcastPropagation(Operation *op) {
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

bool AllowsForBroadcastPropagation(Operation *op) {
  return llvm::isa_and_nonnull<DynamicBroadcastInDimOp>(op) ||
         AllowsForElementwiseBroadcastPropagation(op);
}

DenseIntElementsAttr ComposeBroadcastDimensionsAttr(OpBuilder &builder,
                                                    DenseIntElementsAttr a,
                                                    DenseIntElementsAttr b) {
  SmallVector<int64_t> b_vec =
      llvm::to_vector(llvm::map_range(b, [](const APInt &it) {
        return static_cast<int64_t>(it.getLimitedValue());
      }));
  SmallVector<int64_t> composed_vec = llvm::to_vector(llvm::map_range(
      a, [&](const APInt &it) { return b_vec[it.getLimitedValue()]; }));
  return builder.getI64TensorAttr(composed_vec);
}

// Find all the broadcast intents and their dependencies. Start analyzing from
// the root an collect all broadcast intents that can help broadcast propagation
// from there.
void FindBroadcastIntents(
    DynamicBroadcastInDimOp root, Block *parent_block,
    BroadcastIntent &root_bcast_intent,
    SmallVector<BroadcastIntent> &bcast_intents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcast_intent_dependencies) {
  OpBuilder builder(root.getContext());

  // Use the result vector of broadcast intents as a worklist. The set of
  // broadcast intents helps to ensure their uniqueness.
  DenseSet<BroadcastIntent> bcast_intents_set;
  auto add_to_worklist_if_new = [&](BroadcastIntent bcast_intent) {
    if (!bcast_intents_set.count(bcast_intent)) {
      bcast_intents_set.insert(bcast_intent);
      bcast_intents.push_back(bcast_intent);
    }
  };

  // Derive the broadcast intent associated with the root broadcast operation.
  // Add it to the worklist to seed the analysis.
  root_bcast_intent = {root.getResult().getType().cast<RankedTensorType>(),
                       root.operand(), root.output_dimensions(),
                       root.broadcast_dimensions()};
  add_to_worklist_if_new(root_bcast_intent);

  // We use result vector of broadcast intents as a worklist, the first `i`
  // intents of which have been processed.
  for (int i = 0; i < bcast_intents.size(); ++i) {
    BroadcastIntent it = bcast_intents[i];
    Operation *producer_op = it.target_value.getDefiningOp();

    // We can propagate broadcasts over (broadcasting) element-wise operations
    // and dynamic_broadcast_in_dim ops with the restriction that they must be
    // in the same block as they may depend on assuming regions.
    if (!producer_op || producer_op->getBlock() != parent_block ||
        !AllowsForBroadcastPropagation(producer_op)) {
      continue;
    }

    // We can skip broadcasting producers (dynamic_broadcast_in_dim ops) if we
    // compose their broadcasting dimensions.
    if (auto producer_bcast_op =
            llvm::dyn_cast<DynamicBroadcastInDimOp>(producer_op)) {
      DenseIntElementsAttr composed_bcast_dims = ComposeBroadcastDimensionsAttr(
          builder, producer_bcast_op.broadcast_dimensions(),
          it.broadcast_dimensions.cast<DenseIntElementsAttr>());
      BroadcastIntent bcasted_operand_intent = {
          it.result_type, producer_bcast_op.operand(), it.output_dimensions,
          composed_bcast_dims};

      // Record dependency and "recur".
      bcast_intent_dependencies[it] = {bcasted_operand_intent};
      add_to_worklist_if_new(bcasted_operand_intent);
      continue;
    }

    // We can propagate broadcasts over (broadcasting) element-wise operations.
    // Instead of broadcasting the result of such an op, we can broadcast the
    // operands and apply the element-wise operation to them.
    assert(AllowsForElementwiseBroadcastPropagation(producer_op));
    bcast_intent_dependencies[it] = {};
    for (auto operand : producer_op->getOperands()) {
      auto operand_ty = operand.getType().cast<RankedTensorType>();
      auto operand_bcast_dims = operand_ty.getRank() == 0
                                    ? builder.getI64TensorAttr({})
                                    : it.broadcast_dimensions;
      auto bcasted_operand_ty = RankedTensorType::get(
          it.result_type.getShape(), operand_ty.getElementType());
      BroadcastIntent bcasted_operand_intent = {bcasted_operand_ty, operand,
                                                it.output_dimensions,
                                                operand_bcast_dims};

      // Record dependency and "recur".
      bcast_intent_dependencies[it].push_back(bcasted_operand_intent);
      add_to_worklist_if_new(bcasted_operand_intent);
    }
  }
}

void SortBroadcastIntentsInReverseTopologicalOrder(
    SmallVector<BroadcastIntent> &bcast_intents_vec, Block *parent_block) {
  // Sort broadcast intents in reverse topological order of the producer ops. We
  // can use the positions in the block for this. All broadcast intents outside
  // the block (e.g. arguments) will be sorted towards the front.
  // This ordering is independent of the output dimensions as dependencies can
  // only occur between broadcast intents of the same output dimension.
  std::sort(bcast_intents_vec.begin(), bcast_intents_vec.end(),
            [&](const BroadcastIntent &a, const BroadcastIntent &b) {
              Operation *producer_op_a = a.target_value.getDefiningOp();
              Operation *producer_op_b = b.target_value.getDefiningOp();
              bool a_in_block = producer_op_a != nullptr &&
                                producer_op_a->getBlock() == parent_block;
              bool b_in_block = producer_op_b != nullptr &&
                                producer_op_b->getBlock() == parent_block;
              if (a_in_block && b_in_block) {
                return producer_op_a->isBeforeInBlock(producer_op_b);
              }
              return !a_in_block && b_in_block;
            });
}

void SetInsertionPointToEarliestPointWithAllValuesAvailable(
    PatternRewriter &rewriter, Block *block, ValueRange values) {
  Operation *last_def = nullptr;
  for (Value v : values) {
    Operation *def = v.getDefiningOp();
    if (def && def->getBlock() == block) {
      if (!last_def || last_def->isBeforeInBlock(def)) last_def = def;
    }
  }
  if (last_def) {
    rewriter.setInsertionPointAfter(last_def);
  } else {
    rewriter.setInsertionPointToStart(block);
  }
}

DenseMap<BroadcastIntent, Value> RealizeBroadcastIntents(
    SmallVector<BroadcastIntent> &sorted_bcast_intents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcast_intent_dependencies,
    Block *parent_block, PatternRewriter &rewriter) {
  // Realize broadcast intents in order. They must be sorted so that their
  // dependencies are realized before them.
  DenseMap<BroadcastIntent, Value> realizations;
  for (auto it : sorted_bcast_intents) {
    Operation *producer_op = it.target_value.getDefiningOp();
    assert(!realizations.count(it) && "expect unrealized broadcast intent");
    auto deps = bcast_intent_dependencies.find(it);

    // If we cannot propagate broadcasts further, materialize them as a
    // dynamic_broadcast_in_dim op.
    if (!producer_op || producer_op->getBlock() != parent_block ||
        !AllowsForBroadcastPropagation(producer_op)) {
      assert(deps == bcast_intent_dependencies.end() &&
             "expect no dependencies");
      SetInsertionPointToEarliestPointWithAllValuesAvailable(
          rewriter, parent_block,
          ValueRange{it.target_value, it.output_dimensions});
      realizations[it] = rewriter.create<DynamicBroadcastInDimOp>(
          it.target_value.getLoc(), it.result_type, it.target_value,
          it.output_dimensions,
          it.broadcast_dimensions.cast<DenseIntElementsAttr>());
      continue;
    }

    // For broadcast propagation across dynamic_broadcast_in_dim ops, the
    // broadcasted value is already materialized. Forward it.
    if (auto producer_bcast_op =
            llvm::dyn_cast_or_null<DynamicBroadcastInDimOp>(producer_op)) {
      assert(deps != bcast_intent_dependencies.end() &&
             deps->second.size() == 1 && "expect one dependency");
      auto bcasted_operand = realizations.find(deps->second.front());
      assert(bcasted_operand != realizations.end());
      realizations[it] = Value(bcasted_operand->second);
      continue;
    }

    // Othwerwise, realize broadcast intent for a (broadcasting) element-wise
    // operation based on the broadcasted operands.
    assert(AllowsForElementwiseBroadcastPropagation(producer_op) &&
           "expect broadcast propagation over an (broadcasting) element-wise "
           "operation");
    assert(deps != bcast_intent_dependencies.end() &&
           deps->second.size() == producer_op->getNumOperands() &&
           "expect one dependency per operand");
    auto bcasted_operands = llvm::to_vector(
        llvm::map_range(deps->second, [&](BroadcastIntent operand_intent) {
          auto bcasted_operand = realizations.find(operand_intent);
          assert(bcasted_operand != realizations.end() &&
                 "expect dependencies to be realized earlier");
          return bcasted_operand->second;
        }));
    SetInsertionPointToEarliestPointWithAllValuesAvailable(
        rewriter, parent_block, bcasted_operands);
    OperationState new_producer_op_state(
        producer_op->getLoc(), producer_op->getName().getStringRef(),
        bcasted_operands, it.result_type, producer_op->getAttrs());
    Operation *new_producer_op = rewriter.create(new_producer_op_state);
    assert(new_producer_op->getNumResults() == 1 &&
           "expect exactly one result");
    realizations[it] = new_producer_op->getResults().front();
  }

  return realizations;
}

void TransitivelyEraseUnusedSideEffectFreeOps(Operation *root,
                                              PatternRewriter &rewriter) {
  // Find ops to erase.
  SmallPtrSet<Operation *, 16> ops_to_erase_set;
  SmallVector<Operation *, 16> ops_to_erase;
  SmallVector<Operation *, 16> worklist = {root};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // Erase ops only once.
    if (ops_to_erase_set.count(op)) continue;

    // Erase only operations that are unused and free of side effects.
    if (!MemoryEffectOpInterface::hasNoEffect(op) ||
        !llvm::all_of(op->getUsers(), [&](Operation *user) {
          return ops_to_erase_set.count(user);
        })) {
      continue;
    }

    // Erase and "recur".
    ops_to_erase_set.insert(op);
    ops_to_erase.push_back(op);
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) worklist.push_back(def);
    }
  }

  // Finally, erase the ops in the order of their uses.
  for (Operation *op : ops_to_erase) rewriter.eraseOp(op);
}

LogicalResult PropagateBroadcast(DynamicBroadcastInDimOp root,
                                 Block *parent_block,
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
  BroadcastIntent root_bcast_intent;
  SmallVector<BroadcastIntent> bcast_intents;
  DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
      bcast_intent_dependencies;
  FindBroadcastIntents(root, parent_block, root_bcast_intent, bcast_intents,
                       bcast_intent_dependencies);

  // Fail if there is nothing but the root intent, i.e. if there is nothing to
  // rewrite here.
  if (bcast_intents.size() <= 1) {
    assert(bcast_intents.front() == root_bcast_intent && "expect root intent");
    return failure();
  }

  // Sort the broadcast intents in reverse topological order so that they can be
  // materialized and every depency is available when needed.
  SortBroadcastIntentsInReverseTopologicalOrder(bcast_intents, parent_block);

  // Realize broadcast intents.
  DenseMap<BroadcastIntent, Value> realizations = RealizeBroadcastIntents(
      bcast_intents, bcast_intent_dependencies, parent_block, rewriter);

  // Find the operations that may become redundant after replacing the root
  // operation. This allows us to transitively erase unused side effect-free
  // operations that result from this rewrite (after the root operation is no
  // longer accessible).
  SmallVector<Operation *> possibly_unused;
  for (auto operand : root->getOperands()) {
    if (Operation *def = operand.getDefiningOp())
      possibly_unused.push_back(def);
  }

  // Replace the root operation with its broadcast intent's realization.
  rewriter.replaceOp(root, realizations[root_bcast_intent]);

  // Erase all the operations that have become redundant as a result of this
  // rewrite.
  for (Operation *op : possibly_unused) {
    TransitivelyEraseUnusedSideEffectFreeOps(op, rewriter);
  }

  return success();
}

struct BroadcastPropagationPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern<DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    return PropagateBroadcast(op, op->getBlock(), rewriter);
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
