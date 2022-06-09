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

#include <algorithm>
#include <functional>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mhlo {
namespace {

enum class CstrBroadcastableOperandKind {
  kValue = 0,
  kShapeOfValue = 1,
};

struct CstrBroadcastableOperand {
  static CstrBroadcastableOperand valueOf(BlockArgument barg) {
    return {CstrBroadcastableOperandKind::kValue, barg};
  }
  static CstrBroadcastableOperand shapeOf(BlockArgument barg) {
    return {CstrBroadcastableOperandKind::kShapeOfValue, barg};
  }

  // An arbitrary but well define order.
  inline bool operator<(const CstrBroadcastableOperand &rhs) const {
    if (kind != rhs.kind) return kind < rhs.kind;
    return value.getArgNumber() < rhs.value.getArgNumber();
  }
  inline bool operator>(const CstrBroadcastableOperand &rhs) const {
    return rhs < *this;
  }
  inline bool operator<=(const CstrBroadcastableOperand &rhs) const {
    return !(*this > rhs);
  }
  inline bool operator>=(const CstrBroadcastableOperand &rhs) const {
    return !(*this < rhs);
  }

  // Equality.
  inline bool operator==(const CstrBroadcastableOperand &rhs) const {
    return kind == rhs.kind && value == rhs.value;
  }
  inline bool operator!=(const CstrBroadcastableOperand &rhs) const {
    return !(*this == rhs);
  }

  CstrBroadcastableOperandKind kind;
  BlockArgument value;
};

struct CstrBroadcastableIntent {
  explicit CstrBroadcastableIntent(Location loc) : loc(loc) {}

  // A well defined order that sorts weaker constraints to the front.
  inline bool operator<(const CstrBroadcastableIntent &rhs) const {
    // Sort weaker constraints to the front.
    if (operands.size() != rhs.operands.size())
      return operands.size() < rhs.operands.size();

    return operands < rhs.operands;
  }
  inline bool operator>(const CstrBroadcastableIntent &rhs) const {
    return rhs < *this;
  }
  inline bool operator<=(const CstrBroadcastableIntent &rhs) const {
    return !(*this > rhs);
  }
  inline bool operator>=(const CstrBroadcastableIntent &rhs) const {
    return !(*this < rhs);
  }

  inline bool operator==(const CstrBroadcastableIntent &rhs) const {
    return operands == rhs.operands;
  }
  inline bool operator!=(const CstrBroadcastableIntent &rhs) const {
    return !(*this == rhs);
  }

  Location loc;
  SmallVector<CstrBroadcastableOperand> operands;
};

void canonicalizeBroadcastabilityCstrs(
    SmallVector<CstrBroadcastableIntent> &broadcastabilityCstrs) {
  // Sort inner constraint arguments and eliminate duplicates.
  for (auto &it : broadcastabilityCstrs) {
    llvm::sort(it.operands);
    auto *newEnd =
        llvm::unique(it.operands, [](auto a, auto b) { return a == b; });
    it.operands.erase(newEnd, it.operands.end());
  }

  // Sort broadcastability constraints and sort the strongest to the front.
  llvm::sort(broadcastabilityCstrs, std::greater<>());

  // Remove broadcastability constraints if they are implied by stronger
  // constraints.
  for (int i = 0; i < broadcastabilityCstrs.size(); i++) {
    CstrBroadcastableIntent &strongCstr = broadcastabilityCstrs[i];
    auto *newEnd = std::remove_if(
        broadcastabilityCstrs.begin() + i + 1, broadcastabilityCstrs.end(),
        [strongCstr](CstrBroadcastableIntent weakerCstr) {
          assert(weakerCstr.operands.size() <= strongCstr.operands.size() &&
                 "only look at possibly weaker broadcastability constraints");
          return std::includes(
              strongCstr.operands.begin(), strongCstr.operands.end(),
              weakerCstr.operands.begin(), weakerCstr.operands.end());
        });
    broadcastabilityCstrs.erase(newEnd, broadcastabilityCstrs.end());
  }
}

void eliminateDuplicateBlockArguments(SmallVector<BlockArgument> &bargs) {
  llvm::sort(bargs, [](auto a, auto b) {
    return a.getArgNumber() < b.getArgNumber();
  });
  auto *newEnd = llvm::unique(bargs, [](auto a, auto b) { return a == b; });
  bargs.erase(newEnd, bargs.end());
}

void inlineAssumingRegions(Block *theBlock) {
  theBlock->walk([](shape::AssumingOp aop) {
    Block *body = aop.getBody();
    auto yop = llvm::cast<shape::AssumingYieldOp>(body->getTerminator());
    aop->getBlock()->getOperations().splice(aop->getIterator(),
                                            body->getOperations());
    aop.replaceAllUsesWith(yop.getOperands());
    yop.erase();
    aop.erase();
  });
}

Value materializeFusedConstraints(
    Location loc, OpBuilder &builder, SmallVector<BlockArgument> &argumentCstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastabilityCstrs) {
  // Ensure to materialize shape_of only once.
  DenseMap<Value, Value> shapeOfMaterializations;
  auto getShapeOfMaterialization = [&](Value arg) {
    auto it = shapeOfMaterializations.find(arg);
    if (it != shapeOfMaterializations.end()) return it->second;
    auto shapeOf = builder.create<shape::ShapeOfOp>(loc, arg).getResult();
    shapeOfMaterializations[arg] = shapeOf;
    return shapeOf;
  };

  SmallVector<Value> witnesses;
  witnesses.reserve(argumentCstrs.size() + broadcastabilityCstrs.size());

  // Carry over the argument witnesses.
  for (BlockArgument it : argumentCstrs) witnesses.push_back(it);

  // Materialize broadcastability constraints.
  for (const CstrBroadcastableIntent &it : broadcastabilityCstrs) {
    auto shapes = llvm::to_vector<8>(llvm::map_range(
        it.operands,
        [getShapeOfMaterialization](const CstrBroadcastableOperand &operand) {
          if (operand.kind == CstrBroadcastableOperandKind::kShapeOfValue) {
            return getShapeOfMaterialization(operand.value);
          }
          assert(operand.kind == CstrBroadcastableOperandKind::kValue);
          Value shape = operand.value;
          return shape;
        }));
    auto cstr = builder.create<shape::CstrBroadcastableOp>(it.loc, shapes);
    witnesses.push_back(cstr);
  }
  if (witnesses.size() == 1) return witnesses.front();
  return builder.create<shape::AssumingAllOp>(loc, witnesses);
}

void materializeBlockGlobalConstraintFusion(
    Location loc, OpBuilder &builder, Block *theBlock,
    llvm::SmallSetVector<Operation *, 16> &toBeErased,
    SmallVector<BlockArgument> &argumentCstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastabilityCstrs) {
  // Eliminate the old assuming regions and inline their ops into the main
  // function body.
  inlineAssumingRegions(theBlock);

  // Delete ops that are known to have become redundant by inlining of assuming
  // regions.
  for (auto *it : toBeErased) it->erase();

  // Materialize fused constraints at the beginning of the function.
  builder.setInsertionPointToStart(theBlock);
  Value fusedCstr = materializeFusedConstraints(loc, builder, argumentCstrs,
                                                broadcastabilityCstrs);

  // Create fused assuming region with empty body.
  Operation *theBlockTerminator = theBlock->getTerminator();
  auto fusedAop = builder.create<shape::AssumingOp>(
      loc, theBlockTerminator->getOperandTypes(), fusedCstr);
  auto *fusedAopBody = new Block;
  fusedAop.getDoRegion().getBlocks().push_back(fusedAopBody);

  // Splice all the original block's operations into the fused assuming region's
  // body (except for the block terminator).
  auto &dstBlocks = fusedAopBody->getOperations();
  dstBlocks.splice(dstBlocks.begin(), theBlock->getOperations(),
                   builder.getInsertionPoint(),
                   theBlockTerminator->getIterator());

  // Yield results from the assuming region and pass them on to the original
  // block terminator.
  builder.setInsertionPointToEnd(fusedAopBody);
  builder.create<shape::AssumingYieldOp>(loc,
                                         theBlockTerminator->getOperands());
  theBlockTerminator->setOperands(fusedAop.getResults());
}

bool isRemainingUse(OpOperand &use, Block *the_block,
                    llvm::SmallSetVector<Operation *, 16> &considerDead) {
  Operation *op = use.getOwner();

  // Not a real use if user is considered dead.
  if (considerDead.count(op)) return false;

  // Assuming regions in the regarded block are not a real use as they will be
  // inlined.
  if (auto aop = llvm::dyn_cast<shape::AssumingOp>(op))
    return aop->getBlock() == the_block;

  // Look through assuming regions' yield ops.
  if (auto yop = llvm::dyn_cast<shape::AssumingYieldOp>(op)) {
    auto aop = yop->getParentOfType<shape::AssumingOp>();
    auto outerResult = aop.getResults()[use.getOperandNumber()];
    return llvm::all_of(outerResult.getUses(), [&](auto &outerUse) {
      return isRemainingUse(outerUse, the_block, considerDead);
    });
  }

  // Otherwise, consider it a real use.
  return true;
}

void tryFlagForErase(Block *the_block, Operation *op,
                     llvm::SmallSetVector<Operation *, 16> &toBeErased) {
  if (llvm::none_of(op->getUses(), [&](auto &use) {
        return isRemainingUse(use, the_block, toBeErased);
      })) {
    toBeErased.insert(op);
  }
}

bool isWithinBlock(Operation *op, Block *theBlock) {
  while (op != nullptr && op->getBlock() != theBlock) op = op->getParentOp();
  return op != nullptr;
}

LogicalResult analyzeBroadcastableConstraint(
    shape::CstrBroadcastableOp cstrBcastable, Block *theBlock,
    llvm::SmallSetVector<Operation *, 16> &toBeErased,
    SmallVector<CstrBroadcastableOperand> &transitiveBcastableCstrOperands) {
  SmallVector<Value> worklist = cstrBcastable.getShapes();
  while (!worklist.empty()) {
    Value shape = worklist.pop_back_val();
    Operation *def = shape.getDefiningOp();

    // For shapes without a definition, expect them to be an argument of the
    // regarded block.
    if (def == nullptr) {
      auto barg = shape.dyn_cast<BlockArgument>();
      if (!barg || barg.getParentBlock() != theBlock) return failure();
      transitiveBcastableCstrOperands.push_back(
          CstrBroadcastableOperand::valueOf(barg));
      continue;
    }

    // For shape_of ops, expect them to wrap an argument of the regarded block.
    // The shape reification pass helps achieve this, which should be run before
    // this pass.
    if (auto sof = llvm::dyn_cast<shape::ShapeOfOp>(def)) {
      if (!isWithinBlock(sof, theBlock)) return failure();
      tryFlagForErase(theBlock, def, toBeErased);
      auto barg = sof.getArg().dyn_cast<BlockArgument>();
      if (!barg) return failure();
      transitiveBcastableCstrOperands.push_back(
          CstrBroadcastableOperand::shapeOf(barg));
      continue;
    }

    // For broadcast ops, broadcastability of the operands is an implicit
    // requirement. We can online the operands.
    if (auto bcast = llvm::dyn_cast<shape::BroadcastOp>(def)) {
      if (!isWithinBlock(bcast, theBlock)) return failure();
      tryFlagForErase(theBlock, def, toBeErased);
      auto bcastShapes = bcast.getShapes();
      worklist.append(bcastShapes.begin(), bcastShapes.end());
      continue;
    }

    // Look into assuming ops to proceed.
    if (auto aop = llvm::dyn_cast<shape::AssumingOp>(def)) {
      if (!isWithinBlock(aop, theBlock)) return failure();
      auto yieldOp =
          llvm::cast<shape::AssumingYieldOp>(aop.getBody()->getTerminator());
      size_t i = llvm::find(aop.getResults(), shape).getIndex();
      Value innerShape = yieldOp.getOperand(i);
      worklist.push_back(innerShape);
      continue;
    }

    // Otherwise, bail.
    return failure();
  }

  return success();
}

LogicalResult analyzeBlockGlobalConstraints(
    Block *theBlock, llvm::SmallSetVector<Operation *, 16> &toBeErased,
    SmallVector<BlockArgument> &argumentCstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastabilityCstrs) {
  // Find all the assuming regions and start the search for reachable
  // constraints from there.
  SmallVector<Value> cstrWorklist;
  theBlock->walk(
      [&](shape::AssumingOp aop) { cstrWorklist.push_back(aop.getWitness()); });

  while (!cstrWorklist.empty()) {
    Value cstr = cstrWorklist.pop_back_val();
    Operation *def = cstr.getDefiningOp();

    // For witnesses without a definition, expect it to be an argument of the
    // regarded block.
    if (def == nullptr) {
      auto barg = cstr.dyn_cast<BlockArgument>();
      if (!barg || barg.getParentBlock() != theBlock) return failure();
      argumentCstrs.push_back(barg);
      continue;
    }

    // For conjunctions, continue with the operands.
    if (auto aaop = llvm::dyn_cast<shape::AssumingAllOp>(def)) {
      if (!isWithinBlock(aaop, theBlock)) return failure();
      tryFlagForErase(theBlock, def, toBeErased);
      auto aaopCstrs = aaop.getOperands();
      cstrWorklist.append(aaopCstrs.begin(), aaopCstrs.end());
      continue;
    }

    // For broadcastable constraints, find the transitively included shape
    // operands.
    if (auto cstrBcastable = llvm::dyn_cast<shape::CstrBroadcastableOp>(def)) {
      if (!isWithinBlock(cstrBcastable, theBlock)) return failure();
      tryFlagForErase(theBlock, def, toBeErased);
      CstrBroadcastableIntent bcastableIntent(cstrBcastable.getLoc());
      if (failed(analyzeBroadcastableConstraint(
              cstrBcastable, theBlock, toBeErased, bcastableIntent.operands))) {
        return failure();
      }
      broadcastabilityCstrs.push_back(bcastableIntent);
      continue;
    }

    // Look into assuming regions when running into them. They will be inlined
    // later.
    if (auto aop = llvm::dyn_cast<shape::AssumingOp>(def)) {
      if (!isWithinBlock(aop, theBlock)) return failure();
      size_t i = llvm::find(aop.getResults(), cstr).getIndex();
      auto yieldOp =
          llvm::cast<shape::AssumingYieldOp>(aop.getBody()->getTerminator());
      cstrWorklist.push_back(yieldOp.getOperand(i));
      continue;
    }

    // Otherwise, bail.
    return failure();
  }

  return success();
}

LogicalResult fuseBlockGlobalConstraints(Location loc, OpBuilder &builder,
                                         Block *theBlock) {
  // Analyze block-global constraints.
  SmallVector<BlockArgument> argumentCstrs;
  SmallVector<CstrBroadcastableIntent> broadcastabilityCstrs;
  llvm::SmallSetVector<Operation *, 16> toBeErased;
  if (failed(analyzeBlockGlobalConstraints(theBlock, toBeErased, argumentCstrs,
                                           broadcastabilityCstrs))) {
    return failure();
  }

  // Return early if there is nothing to do.
  if (argumentCstrs.empty() && broadcastabilityCstrs.empty()) {
    return success();
  }

  // Simplify constraints.
  eliminateDuplicateBlockArguments(argumentCstrs);
  canonicalizeBroadcastabilityCstrs(broadcastabilityCstrs);

  // Materialize constraint fusion.
  materializeBlockGlobalConstraintFusion(loc, builder, theBlock, toBeErased,
                                         argumentCstrs, broadcastabilityCstrs);

  return success();
}

struct ConstraintFusionPass
    : public ConstraintFusionPassBase<ConstraintFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    auto loc = f.getLoc();
    OpBuilder builder(&getContext());
    for (auto &block : f.getBody().getBlocks()) {
      if (failed(fuseBlockGlobalConstraints(loc, builder, &block)))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConstraintFusionPass() {
  return std::make_unique<ConstraintFusionPass>();
}

}  // namespace mhlo
}  // namespace mlir
