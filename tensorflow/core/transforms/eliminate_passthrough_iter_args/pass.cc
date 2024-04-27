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

#include "tensorflow/core/transforms/eliminate_passthrough_iter_args/pass.h"

#include <memory>
#include <utility>

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/transforms/utils/utils.h"

// Define the debug label used by `LLVM_DEBUG`.
#define DEBUG_TYPE "uncapture-region"

namespace mlir {
namespace tfg {

#define GEN_PASS_DEF_ELIMINATEPASSTHROUGHITERARGS
#include "tensorflow/core/transforms/passes.h.inc"

// Given a range of elements, this function returns a vector of elements
// excluding the ones whose index is contained in a bit vector.
template <typename RangeT>
static SmallVector<llvm::detail::ValueOfRange<RangeT>> FilterByIndex(
    RangeT &&range, const llvm::BitVector &indices) {
  SmallVector<llvm::detail::ValueOfRange<RangeT>> result;
  for (const auto &it : llvm::enumerate(range))
    if (!indices.test(it.index())) result.push_back(it.value());
  return result;
}

// Given a region, return the indices of arguments that are passthrough.
static SmallVector<unsigned> GetPassthroughIndices(Region &region) {
  Block::BlockArgListType iter_args = GetLoopRegionDataArgs(region);

  // Skip the index argument for `For` ops.
  if (isa<ForRegionOp>(region.getParentOp()))
    iter_args = iter_args.drop_front();

  SmallVector<unsigned> indices;
  auto yield = cast<YieldOp>(region.front().getTerminator());
  for (auto it : llvm::zip(llvm::enumerate(yield.getArgs()), iter_args)) {
    if (std::get<0>(it).value() == std::get<1>(it))
      indices.push_back(std::get<0>(it).index());
  }
  return indices;
}

// Given a terminator, erase the iteration arguments at the specified index.
static void EraseIterArgsFromTerminator(Operation *terminator,
                                        ArrayRef<unsigned> indices) {
  if (isa<YieldOp>(terminator)) {
    util::SizedOperandSegmentsEraseOperands(terminator, indices);
    return;
  }

  // Skip the condition operand.
  assert(isa<ConditionOp>(terminator));
  SmallVector<unsigned> cond_indices = llvm::to_vector(indices);
  for (unsigned &index : cond_indices) ++index;
  util::SizedOperandSegmentsEraseOperands(terminator, cond_indices);
}

namespace {
template <typename ConcreteT, typename OpT>
struct EliminatePassthroughIterArgs {
  // Eliminate the passthrough iteration arguments for the given op. Returns the
  // number of eliminated arguments.
  static size_t Run(OpT op, IRRewriter &rewriter) {
    SmallVector<unsigned> indices = GetPassthroughIndices(op.getBodyRegion());
    if (indices.empty()) return 0;

    LLVM_DEBUG(llvm::dbgs()
               << "Number of captures to erase: " << indices.size() << "\n");
    // We need to:
    // 1. remove the terminator operands
    // 2. remove the operands and corresponding results (and replace them)
    // 3. remove the block arguments (and update preserved attributes)
    llvm::BitVector remove(op.getInit().size());
    for (unsigned index : indices) remove.set(index);

    for (Region &region : op->getRegions())
      EraseIterArgsFromTerminator(region.front().getTerminator(), indices);

    OpT new_op = ConcreteT::RebuildOp(remove, op, rewriter);
    util::ForwardNonIntrinsicAttributes(op, new_op);

    // Replace uses of each passthrough argument with the implicit capture
    // value and remove the argument. Insert the implicitly captured value into
    // the result list to replace the removed results from the original op.
    SmallVector<Value> results = llvm::to_vector(ValueRange(new_op.getOuts()));
    for (const auto &it : llvm::enumerate(indices)) {
      unsigned idx = it.value() - it.index();
      Value data = op.getInit()[it.value()];
      results.insert(results.begin() + it.value(), data);
      ConcreteT::ReplaceArguments(idx, new_op, data,
                                  LookupControlDependency(data));
    }
    results.push_back(new_op.getCtl());
    rewriter.replaceOp(op, results);
    return indices.size();
  }
};

struct EliminateForPassthroughIterArgs
    : public EliminatePassthroughIterArgs<EliminateForPassthroughIterArgs,
                                          ForRegionOp> {
  static ForRegionOp RebuildOp(const llvm::BitVector &indices, ForRegionOp op,
                               IRRewriter &rewriter) {
    rewriter.setInsertionPoint(op);
    auto new_op = rewriter.create<ForRegionOp>(
        op.getLoc(), FilterByIndex(op.getOuts().getTypes(), indices),
        op.getCtl().getType(), op.getStart(), op.getLimit(), op.getDelta(),
        FilterByIndex(op.getInit(), indices), op.getCtls(),
        op.getBodyAttrsAttr(), op.getRegionAttrsAttr());
    new_op.getBodyRegion().takeBody(op.getBodyRegion());
    return new_op;
  }

  static void ReplaceArguments(unsigned index, ForRegionOp op, Value data,
                               Value ctl) {
    // Argument indexing starts from 1 (skip the loop index argument).
    GetLoopRegionDataArgs(op.getBodyRegion())[index + 1].replaceAllUsesWith(
        data);
    GetLoopRegionControlTokens(op.getBodyRegion())[index + 1]
        .replaceAllUsesWith(ctl);
    util::LoopRegionEraseArgument(op.getBodyRegion(), index + 1);
    util::LoopRegionResultErased(op.getBodyRegion(), index);
  }
};

template <typename WhileLikeRegionOp>
struct EliminateWhileLikePassthroughIterArgs
    : public EliminatePassthroughIterArgs<
          EliminateWhileLikePassthroughIterArgs<WhileLikeRegionOp>,
          WhileLikeRegionOp> {
  static WhileLikeRegionOp RebuildOp(const llvm::BitVector &indices,
                                     WhileLikeRegionOp op,
                                     IRRewriter &rewriter) {
    rewriter.setInsertionPoint(op);
    auto new_op = rewriter.create<WhileLikeRegionOp>(
        op.getLoc(), FilterByIndex(op.getOuts().getTypes(), indices),
        op.getCtl().getType(), FilterByIndex(op.getInit(), indices),
        op.getCtls(), op.getParallelIterationsAttr(), op.getCondAttrsAttr(),
        op.getBodyAttrsAttr(), op.getCondRegionAttrsAttr(),
        op.getBodyRegionAttrsAttr());
    new_op.getCondRegion().takeBody(op.getCondRegion());
    new_op.getBodyRegion().takeBody(op.getBodyRegion());
    return new_op;
  }

  static void ReplaceArguments(unsigned index, WhileLikeRegionOp op, Value data,
                               Value ctl) {
    // The while loop's condition function only has one result: the condition.
    // So there are no preserved attributes to delete when removing an iteration
    // argument.
    GetLoopRegionDataArgs(op.getCondRegion())[index].replaceAllUsesWith(data);
    GetLoopRegionControlTokens(op.getCondRegion())[index].replaceAllUsesWith(
        ctl);
    util::LoopRegionEraseArgument(op.getCondRegion(), index);

    GetLoopRegionDataArgs(op.getBodyRegion())[index].replaceAllUsesWith(data);
    GetLoopRegionControlTokens(op.getBodyRegion())[index].replaceAllUsesWith(
        ctl);
    util::LoopRegionEraseArgument(op.getBodyRegion(), index);
    util::LoopRegionResultErased(op.getBodyRegion(), index);
  }
};

struct EliminatePassthroughIterArgsPass
    : public impl::EliminatePassthroughIterArgsBase<
          EliminatePassthroughIterArgsPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](Operation *op) {
      if (auto for_op = dyn_cast<ForRegionOp>(op)) {
        EliminateForPassthroughIterArgs::Run(for_op, rewriter);
      } else if (auto while_op = dyn_cast<WhileRegionOp>(op)) {
        EliminateWhileLikePassthroughIterArgs<WhileRegionOp>::Run(while_op,
                                                                  rewriter);
      } else if (auto while_op = dyn_cast<StatelessWhileRegionOp>(op)) {
        EliminateWhileLikePassthroughIterArgs<StatelessWhileRegionOp>::Run(
            while_op, rewriter);
      } else if (auto while_op = dyn_cast<StatefulWhileRegionOp>(op)) {
        EliminateWhileLikePassthroughIterArgs<StatefulWhileRegionOp>::Run(
            while_op, rewriter);
      }
    });
  }
};
}  // namespace

std::unique_ptr<Pass> CreateEliminatePassthroughIterArgsPass() {
  return std::make_unique<EliminatePassthroughIterArgsPass>();
}

}  // namespace tfg
}  // namespace mlir
