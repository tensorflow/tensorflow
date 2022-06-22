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

void CanonicalizeBroadcastabilityCstrs(
    SmallVector<CstrBroadcastableIntent> &broadcastability_cstrs) {
  // Sort inner constraint arguments and eliminate duplicates.
  for (auto &it : broadcastability_cstrs) {
    llvm::sort(it.operands);
    auto *new_end =
        llvm::unique(it.operands, [](auto a, auto b) { return a == b; });
    it.operands.erase(new_end, it.operands.end());
  }

  // Sort broadcastability constraints and sort the strongest to the front.
  llvm::sort(broadcastability_cstrs, std::greater<>());

  // Remove broadcastability constraints if they are implied by stronger
  // constraints.
  for (int i = 0; i < broadcastability_cstrs.size(); i++) {
    CstrBroadcastableIntent &strong_cstr = broadcastability_cstrs[i];
    auto *new_end = std::remove_if(
        broadcastability_cstrs.begin() + i + 1, broadcastability_cstrs.end(),
        [strong_cstr](CstrBroadcastableIntent weaker_cstr) {
          assert(weaker_cstr.operands.size() <= strong_cstr.operands.size() &&
                 "only look at possibly weaker broadcastability constraints");
          return std::includes(
              strong_cstr.operands.begin(), strong_cstr.operands.end(),
              weaker_cstr.operands.begin(), weaker_cstr.operands.end());
        });
    broadcastability_cstrs.erase(new_end, broadcastability_cstrs.end());
  }
}

void EliminateDuplicateBlockArguments(SmallVector<BlockArgument> &bargs) {
  llvm::sort(bargs, [](auto a, auto b) {
    return a.getArgNumber() < b.getArgNumber();
  });
  auto *new_end = llvm::unique(bargs, [](auto a, auto b) { return a == b; });
  bargs.erase(new_end, bargs.end());
}

void InlineAssumingRegions(Block *the_block) {
  the_block->walk([](shape::AssumingOp aop) {
    Block *body = aop.getBody();
    auto yop = llvm::cast<shape::AssumingYieldOp>(body->getTerminator());
    aop->getBlock()->getOperations().splice(aop->getIterator(),
                                            body->getOperations());
    aop.replaceAllUsesWith(yop.getOperands());
    yop.erase();
    aop.erase();
  });
}

Value MaterializeFusedConstraints(
    Location loc, OpBuilder &builder,
    SmallVector<BlockArgument> &argument_cstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastability_cstrs) {
  // Ensure to materialize shape_of only once.
  DenseMap<Value, Value> shape_of_materializations;
  auto get_shape_of_materialization = [&](Value arg) {
    auto it = shape_of_materializations.find(arg);
    if (it != shape_of_materializations.end()) return it->second;
    auto shape_of = builder.create<shape::ShapeOfOp>(loc, arg).getResult();
    shape_of_materializations[arg] = shape_of;
    return shape_of;
  };

  SmallVector<Value> witnesses;
  witnesses.reserve(argument_cstrs.size() + broadcastability_cstrs.size());

  // Carry over the argument witnesses.
  for (BlockArgument it : argument_cstrs) witnesses.push_back(it);

  // Materialize broadcastability constraints.
  for (const CstrBroadcastableIntent &it : broadcastability_cstrs) {
    auto shapes = llvm::to_vector<8>(llvm::map_range(
        it.operands, [&](const CstrBroadcastableOperand &operand) {
          if (operand.kind == CstrBroadcastableOperandKind::kShapeOfValue) {
            return get_shape_of_materialization(operand.value);
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

void MaterializeBlockGlobalConstraintFusion(
    Location loc, OpBuilder &builder, Block *the_block,
    llvm::SmallSetVector<Operation *, 16> &to_be_erased,
    SmallVector<BlockArgument> &argument_cstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastability_cstrs) {
  // Eliminate the old assuming regions and inline their ops into the main
  // function body.
  InlineAssumingRegions(the_block);

  // Delete ops that are known to have become redundant by inlining of assuming
  // regions.
  for (auto *it : to_be_erased) it->erase();

  // Materialize fused constraints at the beginning of the function.
  builder.setInsertionPointToStart(the_block);
  Value fused_cstr = MaterializeFusedConstraints(loc, builder, argument_cstrs,
                                                 broadcastability_cstrs);

  // Create fused assuming region with empty body.
  Operation *the_block_terminator = the_block->getTerminator();
  auto fused_aop = builder.create<shape::AssumingOp>(
      loc, the_block_terminator->getOperandTypes(), fused_cstr);
  auto *fused_aop_body = new Block;
  fused_aop.getDoRegion().getBlocks().push_back(fused_aop_body);

  // Splice all the original block's operations into the fused assuming region's
  // body (except for the block terminator).
  auto &dst_blocks = fused_aop_body->getOperations();
  dst_blocks.splice(dst_blocks.begin(), the_block->getOperations(),
                    builder.getInsertionPoint(),
                    the_block_terminator->getIterator());

  // Yield results from the assuming region and pass them on to the original
  // block terminator.
  builder.setInsertionPointToEnd(fused_aop_body);
  builder.create<shape::AssumingYieldOp>(loc,
                                         the_block_terminator->getOperands());
  the_block_terminator->setOperands(fused_aop.getResults());
}

bool IsRemainingUse(OpOperand &use, Block *the_block,
                    llvm::SmallSetVector<Operation *, 16> &consider_dead) {
  Operation *op = use.getOwner();

  // Not a real use if user is considered dead.
  if (consider_dead.count(op)) return false;

  // Assuming regions in the regarded block are not a real use as they will be
  // inlined.
  if (auto aop = llvm::dyn_cast<shape::AssumingOp>(op))
    return aop->getBlock() == the_block;

  // Look through assuming regions' yield ops.
  if (auto yop = llvm::dyn_cast<shape::AssumingYieldOp>(op)) {
    auto aop = yop->getParentOfType<shape::AssumingOp>();
    auto outer_result = aop.getResults()[use.getOperandNumber()];
    return llvm::all_of(outer_result.getUses(), [&](auto &outer_use) {
      return IsRemainingUse(outer_use, the_block, consider_dead);
    });
  }

  // Otherwise, consider it a real use.
  return true;
}

void TryFlagForErase(Block *the_block, Operation *op,
                     llvm::SmallSetVector<Operation *, 16> &to_be_erased) {
  if (llvm::none_of(op->getUses(), [&](auto &use) {
        return IsRemainingUse(use, the_block, to_be_erased);
      })) {
    to_be_erased.insert(op);
  }
}

bool IsWithinBlock(Operation *op, Block *the_block) {
  while (op != nullptr && op->getBlock() != the_block) op = op->getParentOp();
  return op != nullptr;
}

LogicalResult AnalyzeBroadcastableConstraint(
    shape::CstrBroadcastableOp cstr_bcastable, Block *the_block,
    llvm::SmallSetVector<Operation *, 16> &to_be_erased,
    SmallVector<CstrBroadcastableOperand> &transitive_bcastable_cstr_operands) {
  SmallVector<Value> worklist = cstr_bcastable.getShapes();
  while (!worklist.empty()) {
    Value shape = worklist.pop_back_val();
    Operation *def = shape.getDefiningOp();

    // For shapes without a definition, expect them to be an argument of the
    // regarded block.
    if (def == nullptr) {
      auto barg = shape.dyn_cast<BlockArgument>();
      if (!barg || barg.getParentBlock() != the_block) return failure();
      transitive_bcastable_cstr_operands.push_back(
          CstrBroadcastableOperand::valueOf(barg));
      continue;
    }

    // For shape_of ops, expect them to wrap an argument of the regarded block.
    // The shape reification pass helps achieve this, which should be run before
    // this pass.
    if (auto sof = llvm::dyn_cast<shape::ShapeOfOp>(def)) {
      if (!IsWithinBlock(sof, the_block)) return failure();
      TryFlagForErase(the_block, def, to_be_erased);
      auto barg = sof.getArg().dyn_cast<BlockArgument>();
      if (!barg) return failure();
      transitive_bcastable_cstr_operands.push_back(
          CstrBroadcastableOperand::shapeOf(barg));
      continue;
    }

    // For broadcast ops, broadcastability of the operands is an implicit
    // requirement. We can online the operands.
    if (auto bcast = llvm::dyn_cast<shape::BroadcastOp>(def)) {
      if (!IsWithinBlock(bcast, the_block)) return failure();
      TryFlagForErase(the_block, def, to_be_erased);
      auto bcast_shapes = bcast.getShapes();
      worklist.append(bcast_shapes.begin(), bcast_shapes.end());
      continue;
    }

    // Look into assuming ops to proceed.
    if (auto aop = llvm::dyn_cast<shape::AssumingOp>(def)) {
      if (!IsWithinBlock(aop, the_block)) return failure();
      auto yield_op =
          llvm::cast<shape::AssumingYieldOp>(aop.getBody()->getTerminator());
      size_t i = llvm::find(aop.getResults(), shape).getIndex();
      Value inner_shape = yield_op.getOperand(i);
      worklist.push_back(inner_shape);
      continue;
    }

    // Otherwise, bail.
    return failure();
  }

  return success();
}

LogicalResult AnalyzeBlockGlobalConstraints(
    Block *the_block, llvm::SmallSetVector<Operation *, 16> &to_be_erased,
    SmallVector<BlockArgument> &argument_cstrs,
    SmallVector<CstrBroadcastableIntent> &broadcastability_cstrs) {
  // Find all the assuming regions and start the search for reachable
  // constraints from there.
  SmallVector<Value> cstr_worklist;
  the_block->walk([&](shape::AssumingOp aop) {
    cstr_worklist.push_back(aop.getWitness());
  });

  while (!cstr_worklist.empty()) {
    Value cstr = cstr_worklist.pop_back_val();
    Operation *def = cstr.getDefiningOp();

    // For witnesses without a definition, expect it to be an argument of the
    // regarded block.
    if (def == nullptr) {
      auto barg = cstr.dyn_cast<BlockArgument>();
      if (!barg || barg.getParentBlock() != the_block) return failure();
      argument_cstrs.push_back(barg);
      continue;
    }

    // For conjunctions, continue with the operands.
    if (auto aaop = llvm::dyn_cast<shape::AssumingAllOp>(def)) {
      if (!IsWithinBlock(aaop, the_block)) return failure();
      TryFlagForErase(the_block, def, to_be_erased);
      auto aaop_cstrs = aaop.getOperands();
      cstr_worklist.append(aaop_cstrs.begin(), aaop_cstrs.end());
      continue;
    }

    // For broadcastable constraints, find the transitively included shape
    // operands.
    if (auto cstr_bcastable = llvm::dyn_cast<shape::CstrBroadcastableOp>(def)) {
      if (!IsWithinBlock(cstr_bcastable, the_block)) return failure();
      TryFlagForErase(the_block, def, to_be_erased);
      CstrBroadcastableIntent bcastable_intent(cstr_bcastable.getLoc());
      if (failed(AnalyzeBroadcastableConstraint(cstr_bcastable, the_block,
                                                to_be_erased,
                                                bcastable_intent.operands))) {
        return failure();
      }
      broadcastability_cstrs.push_back(bcastable_intent);
      continue;
    }

    // Look into assuming regions when running into them. They will be inlined
    // later.
    if (auto aop = llvm::dyn_cast<shape::AssumingOp>(def)) {
      if (!IsWithinBlock(aop, the_block)) return failure();
      size_t i = llvm::find(aop.getResults(), cstr).getIndex();
      auto yield_op =
          llvm::cast<shape::AssumingYieldOp>(aop.getBody()->getTerminator());
      cstr_worklist.push_back(yield_op.getOperand(i));
      continue;
    }

    // Otherwise, bail.
    return failure();
  }

  return success();
}

LogicalResult FuseBlockGlobalConstraints(Location loc, OpBuilder &builder,
                                         Block *the_block) {
  // Analyze block-global constraints.
  SmallVector<BlockArgument> argument_cstrs;
  SmallVector<CstrBroadcastableIntent> broadcastability_cstrs;
  llvm::SmallSetVector<Operation *, 16> to_be_erased;
  if (failed(AnalyzeBlockGlobalConstraints(
          the_block, to_be_erased, argument_cstrs, broadcastability_cstrs))) {
    return failure();
  }

  // Return early if there is nothing to do.
  if (argument_cstrs.empty() && broadcastability_cstrs.empty()) {
    return success();
  }

  // Simplify constraints.
  EliminateDuplicateBlockArguments(argument_cstrs);
  CanonicalizeBroadcastabilityCstrs(broadcastability_cstrs);

  // Materialize constraint fusion.
  MaterializeBlockGlobalConstraintFusion(loc, builder, the_block, to_be_erased,
                                         argument_cstrs,
                                         broadcastability_cstrs);

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
      if (failed(FuseBlockGlobalConstraints(loc, builder, &block)))
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
