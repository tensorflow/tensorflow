/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "mhlo-control-flow-to-scf"

namespace mlir {
namespace mhlo {

namespace {

/// Convert MHLO While to SCF.
void MatchAndRewrite(WhileOp whileOp);

/// Pass that converts MHLO control flow to SCF.
class ControlFlowToScfPass
    : public mlir::PassWrapper<ControlFlowToScfPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnFunction() override {
    getFunction().walk([&](WhileOp whileOp) { MatchAndRewrite(whileOp); });
  }
};

// TODO(jpienaar): Look into reformulating as a pattern.
void MatchAndRewrite(WhileOp whileOp) {
  // Handle pattern:
  //   x = start
  //   step = ...
  //   limit = ...
  //   while (x < limit) { ... x += step; }

  // Only handling multi value while loops at the moment.
  auto tupleOp = whileOp.getOperand().getDefiningOp<TupleOp>();
  if (!tupleOp) return;
  auto bodyReturn = whileOp.body()
                        .front()
                        .getTerminator()
                        ->getOperand(0)
                        .getDefiningOp<mhlo::TupleOp>();
  // Note: due to the shape restrictions on While, if the operand to While is a
  // tuple, then so is the return type of the body. But the verifier isn't
  // checking that at the moment, so just bail out here if this doesn't hold.
  if (!bodyReturn) return;

  Value result = whileOp.cond().front().getTerminator()->getOperand(0);
  // TODO(jpienaar): Expand to handle more than simple case with LT compare and
  // constant step.
  auto cmp = result.getDefiningOp<mhlo::CompareOp>();
  if (!cmp || cmp.comparison_direction() != "LT") return;

  const int kConstant = -1;
  auto getValueAndIndex = [&](Value val) -> std::pair<Value, int> {
    if (matchPattern(val, m_Constant())) return {val, kConstant};
    // If it is defined by a tuple, then the tuple has to have been fed in and
    // the external value is captured.
    if (auto gte = val.getDefiningOp<GetTupleElementOp>()) {
      if (!gte.getOperand().isa<mlir::BlockArgument>()) return {nullptr, 0};
      int index = gte.index();
      return {tupleOp.getOperand(index), index};
    }
    return {nullptr, 0};
  };

  using ValueIndex = std::pair<Value, int>;
  ValueIndex loopIndVar = getValueAndIndex(cmp.lhs());
  ValueIndex max = getValueAndIndex(cmp.rhs());
  if (!loopIndVar.first || !max.first) return;
  auto add =
      bodyReturn.getOperand(loopIndVar.second).getDefiningOp<mhlo::AddOp>();
  if (!add) return;
  ValueIndex step = getValueAndIndex(add.rhs());
  if (step.second != kConstant || !step.first) return;

  // Only handle case where tuple isn't propagated as is for now.
  // TODO(jpienaar): Remove this when a tuple is also created inside the loop
  // to propagate.
  for (auto* use : whileOp.body().front().getArgument(0).getUsers())
    if (!isa<GetTupleElementOp>(use)) return;

  LLVM_DEBUG(llvm::dbgs() << "Found for (" << whileOp.getLoc() << "):\n";
             llvm::dbgs() << "  loopIndVar = " << loopIndVar.second << " max = "
                          << max.second << " step = " << step.second << "\n";
             llvm::dbgs() << "  loopIndVar = " << loopIndVar.first << " max = "
                          << max.first << " step = " << step.first << "\n";);
  OpBuilder b(whileOp);
  // Inputs to new for loop.
  llvm::SmallVector<Value, 4> input;
  input.reserve(tupleOp.getNumOperands());
  for (auto r : tupleOp.getOperands().take_front(loopIndVar.second))
    input.push_back(r);
  for (auto r : tupleOp.getOperands().drop_front(loopIndVar.second + 1))
    input.push_back(r);

  auto tensorIndexType = RankedTensorType::get({}, b.getIndexType());
  auto getAsIndex = [&](Value val) {
    auto loc = whileOp.getLoc();
    return b.create<tensor::ExtractOp>(
        loc, b.create<IndexCastOp>(loc, tensorIndexType, val), ValueRange());
  };

  // SCF for uses index type, so converted these.
  auto forloopIndVar = getAsIndex(loopIndVar.first);
  auto forMax = getAsIndex(max.first);
  auto forStep = getAsIndex(step.first);
  auto forOp = b.create<mlir::scf::ForOp>(whileOp.getLoc(), forloopIndVar,
                                          forMax, forStep, input);
  // Transfer the body without the block arguments.
  forOp.getLoopBody().front().getOperations().splice(
      forOp.getLoopBody().front().getOperations().end(),
      whileOp.body().front().getOperations());

  b.setInsertionPointToStart(&forOp.getLoopBody().front());
  auto loopIndVarElType =
      loopIndVar.first.getType().cast<ShapedType>().getElementType();
  Value indVar = b.create<SplatOp>(
      whileOp.getLoc(), RankedTensorType::get({}, loopIndVarElType),
      b.create<IndexCastOp>(whileOp.getLoc(), loopIndVarElType,
                            forOp.getInductionVar()));
  // Update all block argument users to the SCF For args.
  for (auto* use :
       llvm::make_early_inc_range(whileOp.body().getArgument(0).getUsers())) {
    // TODO(jpienaar): Expand here too when we allow using the tuple in the
    // loop.
    auto gte = cast<GetTupleElementOp>(use);
    // If the loop induction var, then refer to the loop induction variable as
    // this operand is not updated.
    if (gte.index() == loopIndVar.second) {
      use->getResult(0).replaceAllUsesWith(indVar);
      use->erase();
      continue;
    }
    int index = gte.index();
    // If after the loop induction variable, then decrement as we don't include
    // the loop induction variable in the for iter operands.
    if (index > loopIndVar.second) --index;
    use->getResult(0).replaceAllUsesWith(forOp.getIterOperands()[index]);
    use->erase();
  }

  // Create new yield op without induction var update.
  SmallVector<Value, 4> newYieldOps;
  newYieldOps.reserve(bodyReturn.getNumOperands() - 1);
  for (auto r : bodyReturn.getOperands().take_front(loopIndVar.second))
    newYieldOps.push_back(r);
  for (auto r : bodyReturn.getOperands().drop_front(loopIndVar.second + 1))
    newYieldOps.push_back(r);
  // Delete return & tuple op.
  forOp.getLoopBody().front().back().erase();
  forOp.getLoopBody().front().back().erase();
  b.setInsertionPointToEnd(&forOp.getLoopBody().front());
  b.create<scf::YieldOp>(whileOp.getLoc(), newYieldOps);

  // Recombine output tuple with max value of induction variable.
  llvm::SmallVector<Value, 4> loopOut;
  loopOut.reserve(forOp.getNumResults() + 1);
  for (auto r : forOp.getResults().take_front(loopIndVar.second))
    loopOut.push_back(r);
  loopOut.push_back(max.first);
  for (auto r : forOp.getResults().drop_front(loopIndVar.second))
    loopOut.push_back(r);
  b.setInsertionPoint(whileOp);
  auto newRes = b.create<mhlo::TupleOp>(whileOp.getLoc(), loopOut);
  whileOp.replaceAllUsesWith(newRes.getOperation());
  whileOp.erase();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createControlFlowToScfPass() {
  return std::make_unique<ControlFlowToScfPass>();
}

}  // namespace mhlo
}  // namespace mlir
