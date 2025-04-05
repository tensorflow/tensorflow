//===- LoopPipelining.cpp - Code to perform loop software pipelining-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
//
//===----------------------------------------------------------------------===//

// Fork of upstream pipeliner. This will be merged upstream once things are
// stable. Modifications so far are:
// -Bug fix for def with a distance of 1 scheduled in stage 0.
// -Support dynamic loops and predicate operations in the prologue.
// -Support for non-index type for induction variable.
// -Support source with distance of 1 used multiple stages later.
// -Fix bug when a value yield is used outside the loop and the value def is not
// in the last stage. If we are not peeling the epilgue we need to remap the
// output correctly.
// -Allow for distance of 2 or more between producer and consumer for the cases
// where the producer is in the same stage as the consumer.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"

// FIXME: PipelineExpander should not depend on Triton-specific headers!
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#define DEBUG_TYPE "triton-loop-pipelining"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::triton;

namespace {

/// Helper to keep internal information during pipelining transformation.
struct LoopPipelinerInternal {
  /// Coarse liverange information for ops used across stages.
  struct LiverangeInfo {
    unsigned lastUseStage = 0;
    unsigned defStage = 0;
  };

protected:
  ForOp forOp;
  unsigned maxStage = 0;
  DenseMap<Operation *, unsigned> stages;
  std::vector<Operation *> opOrder;
  Value ub;
  Value lb;
  Value step;
  bool dynamicLoop;
  triton::PipeliningOption::AnnotationlFnType annotateFn = nullptr;
  bool peelEpilogue;
  triton::PipeliningOption::PredicateOpFnType predicateFn = nullptr;

  // When peeling the kernel we generate several version of each value for
  // different stage of the prologue. This map tracks the mapping between
  // original Values in the loop and the different versions
  // peeled from the loop.
  DenseMap<Value, llvm::SmallVector<Value>> valueMapping;

  /// Assign a value to `valueMapping`, this means `val` represents the version
  /// `idx` of `key` in the epilogue.
  void setValueMapping(Value key, Value el, int64_t idx);

  /// Return the defining op of the given value, if the Value is an argument of
  /// the loop return the associated defining op in the loop and its distance to
  /// the Value.
  std::pair<Operation *, int64_t> getDefiningOpAndDistance(Value value);

  /// Return true if the schedule is possible and return false otherwise. A
  /// schedule is correct if all definitions are scheduled before uses.
  bool verifySchedule();

public:
  /// Initialize the information for the given `op`, return true if it
  /// satisfies the pre-condition to apply pipelining.
  bool initializeLoopInfo(ForOp op, const triton::PipeliningOption &options);
  /// Emits the prologue, this creates `maxStage - 1` part which will contain
  /// operations from stages [0; i], where i is the part index.
  LogicalResult emitPrologue(RewriterBase &rewriter);
  /// Gather liverange information for Values that are used in a different stage
  /// than its definition.
  llvm::MapVector<Value, LiverangeInfo> analyzeCrossStageValues();
  scf::ForOp createKernelLoop(
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      RewriterBase &rewriter,
      llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap);
  /// Emits the pipelined kernel. This clones loop operations following user
  /// order and remaps operands defined in a different stage as their use.
  LogicalResult createKernel(
      scf::ForOp newForOp,
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
      RewriterBase &rewriter);
  /// Emits the epilogue, this creates `maxStage - 1` part which will contain
  /// operations from stages [i; maxStage], where i is the part index.
  LogicalResult emitEpilogue(RewriterBase &rewriter,
                             llvm::SmallVector<Value> &returnValues);
};

/// Find operands of all the nested operations within `op`.
static SetVector<Value> getNestedOperands(Operation *op) {
  SetVector<Value> operands;
  op->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      operands.insert(operand);
    }
  });
  return operands;
}

bool LoopPipelinerInternal::initializeLoopInfo(
    ForOp op, const triton::PipeliningOption &options) {
  LDBG("Start initializeLoopInfo");
  forOp = op;
  ub = forOp.getUpperBound();
  lb = forOp.getLowerBound();
  step = forOp.getStep();

  dynamicLoop = true;
  auto upperBoundCst = ub.getDefiningOp<arith::ConstantIndexOp>();
  auto lowerBoundCst = lb.getDefiningOp<arith::ConstantIndexOp>();
  auto stepCst = step.getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBoundCst || !lowerBoundCst || !stepCst) {
    if (!options.supportDynamicLoops) {
      LDBG("--dynamic loop not supported -> BAIL");
      return false;
    }
  } else {
    int64_t ubImm = upperBoundCst.value();
    int64_t lbImm = lowerBoundCst.value();
    int64_t stepImm = stepCst.value();
    int64_t numIteration = llvm::divideCeilSigned(ubImm - lbImm, stepImm);
    if (numIteration > maxStage) {
      dynamicLoop = false;
    } else if (!options.supportDynamicLoops) {
      LDBG("--fewer loop iterations than pipeline stages -> BAIL");
      return false;
    }
  }
  peelEpilogue = options.peelEpilogue;
  predicateFn = options.predicateFn;
  if ((!peelEpilogue || dynamicLoop) && predicateFn == nullptr) {
    LDBG("--no epilogue or predicate set -> BAIL");
    return false;
  }
  std::vector<std::pair<Operation *, unsigned>> schedule;
  options.getScheduleFn(forOp, schedule);
  if (schedule.empty()) {
    LDBG("--empty schedule -> BAIL");
    return false;
  }

  opOrder.reserve(schedule.size());
  for (auto &opSchedule : schedule) {
    maxStage = std::max(maxStage, opSchedule.second);
    stages[opSchedule.first] = opSchedule.second;
    opOrder.push_back(opSchedule.first);
  }

  // All operations need to have a stage.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!stages.contains(&op)) {
      op.emitOpError("not assigned a pipeline stage");
      LDBG("--op not assigned a pipeline stage: " << op << " -> BAIL");
      return false;
    }
  }

  if (!verifySchedule()) {
    LDBG("--invalid schedule: " << op << " -> BAIL");
    return false;
  }

  // Currently, we do not support assigning stages to ops in nested regions. The
  // block of all operations assigned a stage should be the single `scf.for`
  // body block.
  for (const auto &[op, stageNum] : stages) {
    (void)stageNum;
    if (op == forOp.getBody()->getTerminator()) {
      op->emitError("terminator should not be assigned a stage");
      LDBG("--terminator should not be assigned stage: " << *op << " -> BAIL");
      return false;
    }
    if (op->getBlock() != forOp.getBody()) {
      op->emitOpError("the owning Block of all operations assigned a stage "
                      "should be the loop body block");
      LDBG("--the owning Block of all operations assigned a stage "
           "should be the loop body block: "
           << *op << " -> BAIL");
      return false;
    }
  }

  // Support only loop-carried dependencies with a distance of one iteration or
  // those defined outside of the loop. This means that any dependency within a
  // loop should either be on the immediately preceding iteration, the current
  // iteration, or on variables whose values are set before entering the loop.
  for (auto &op : forOp.getBody()->without_terminator()) {
    for (auto operand : getNestedOperands(&op)) {
      auto [def, distance] = getDefiningOpAndDistance(operand);
      if (!def)
        continue;
      if (distance > 1 && (stages[def] != stages[&op])) {
        // Allow the case of loop carried dependency between the ops in the same
        // stage.
        LDBG("--only support loop carried dependency with a distance of 1 or "
             "defined outside of the loop -> BAIL");
        return false;
      }
    }
  }
  annotateFn = options.annotateFn;
  return true;
}

/// Compute unrolled cycles of each op (consumer) and verify that each op is
/// scheduled after its operands (producers) while adjusting for the distance
/// between producer and consumer.
bool LoopPipelinerInternal::verifySchedule() {
  int64_t numCylesPerIter = opOrder.size();
  // Pre-compute the unrolled cycle of each op.
  DenseMap<Operation *, int64_t> unrolledCyles;
  for (int64_t cycle = 0; cycle < numCylesPerIter; cycle++) {
    Operation *def = opOrder[cycle];
    auto it = stages.find(def);
    assert(it != stages.end());
    int64_t stage = it->second;
    unrolledCyles[def] = cycle + stage * numCylesPerIter;
  }
  for (Operation *consumer : opOrder) {
    int64_t consumerCycle = unrolledCyles[consumer];
    for (Value operand : getNestedOperands(consumer)) {
      auto [producer, distance] = getDefiningOpAndDistance(operand);
      if (!producer)
        continue;
      auto it = unrolledCyles.find(producer);
      // Skip producer coming from outside the loop.
      if (it == unrolledCyles.end())
        continue;
      int64_t producerCycle = it->second;
      if (consumerCycle < producerCycle - numCylesPerIter * distance) {
        InFlightDiagnostic diag =
            consumer->emitWarning("operation scheduled before its operands. "
                                  "Pipelining will be disabled.");
        diag.attachNote(producer->getLoc())
            .append("operand defined here: ")
            .appendOp(*producer, OpPrintingFlags().printGenericOpForm());
        return false;
      }
    }
  }
  return true;
}

/// Clone `op` and call `callback` on the cloned op's operands as well as any
/// operands of nested ops that:
/// 1) aren't defined within the new op or
/// 2) are block arguments.
static Operation *
cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                       function_ref<void(OpOperand *newOperand)> callback) {
  Operation *clone = rewriter.clone(*op);
  clone->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    // 'clone' itself will be visited first.
    for (OpOperand &operand : nested->getOpOperands()) {
      Operation *def = operand.get().getDefiningOp();
      if ((def && !clone->isAncestor(def)) || isa<BlockArgument>(operand.get()))
        callback(&operand);
    }
  });
  return clone;
}

LogicalResult LoopPipelinerInternal::emitPrologue(RewriterBase &rewriter) {
  // Initialize the iteration argument to the loop initiale values.
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getInitsMutable())) {
    setValueMapping(arg, operand.get(), 0);
  }

  // If the incoming value to an iter arg from the loop yield is defined outside
  // the loop, then that means the iter arg takes that value for all stages
  // after the first stage.
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), yield->getOpOperands())) {
    if (forOp.getBodyRegion().isAncestor(operand.get().getParentRegion()))
      continue;
    for (int64_t i = 1; i < maxStage; ++i)
      setValueMapping(arg, operand.get(), i);
  }

  Location loc = forOp.getLoc();
  SmallVector<Value> predicates(maxStage);
  for (int64_t i = 0; i < maxStage; i++) {
    // special handling for induction variable as the increment is implicit.
    // iv = lb + i * step
    Type t = lb.getType();
    Value iv = rewriter.create<arith::AddIOp>(
        loc, lb,
        rewriter.create<arith::MulIOp>(
            loc, step,
            rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getIntegerAttr(t, i))));
    setValueMapping(forOp.getInductionVar(), iv, i);

    if (dynamicLoop) {
      // pred = ub > lb + (i * step)
      predicates[i] = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, iv, ub);
    }

    for (Operation *op : opOrder) {
      if (stages[op] > i)
        continue;
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[i - stages[op]];
              newOperand->set(replacement);
            }
          });
      int predicateIdx = i - stages[op];
      if (predicates[predicateIdx]) {
        OpBuilder::InsertionGuard insertGuard(rewriter);
        newOp = predicateFn(rewriter, newOp, predicates[predicateIdx]);
        if (newOp == nullptr)
          return failure();
      }
      if (annotateFn)
        annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Prologue, i);
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        Value source = newOp->getResult(destId);
        // If the value is a loop carried dependency update the loop argument
        for (OpOperand &operand : yield->getOpOperands()) {
          if (operand.get() != op->getResult(destId))
            continue;
          if (predicates[predicateIdx] &&
              !forOp.getResult(operand.getOperandNumber()).use_empty()) {
            // If the value is used outside the loop, we need to make sure we
            // return the correct version of it.
            Value prevValue = valueMapping
                [forOp.getRegionIterArgs()[operand.getOperandNumber()]]
                [i - stages[op]];
            source = rewriter.create<arith::SelectOp>(
                loc, predicates[predicateIdx], source, prevValue);
          }
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],
                          source, i - stages[op] + 1);
        }
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        i - stages[op]);
      }
    }
  }
  return success();
}

llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
LoopPipelinerInternal::analyzeCrossStageValues() {
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo> crossStageValues;
  for (Operation *op : opOrder) {
    unsigned stage = stages[op];

    auto analyzeOperand = [&](OpOperand &operand) {
      auto [def, distance] = getDefiningOpAndDistance(operand.get());
      if (!def)
        return;
      auto defStage = stages.find(def);
      if (defStage == stages.end() || defStage->second == stage ||
          defStage->second == stage + distance)
        return;
      assert(stage > defStage->second);
      LiverangeInfo &info = crossStageValues[operand.get()];
      info.defStage = defStage->second;
      info.lastUseStage = std::max(info.lastUseStage, stage);
    };

    for (OpOperand &operand : op->getOpOperands())
      analyzeOperand(operand);
    visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand *operand) {
      analyzeOperand(*operand);
    });
  }
  return crossStageValues;
}

std::pair<Operation *, int64_t>
LoopPipelinerInternal::getDefiningOpAndDistance(Value value) {
  return triton::getDefiningOpAndDistance(forOp, value);
}

scf::ForOp LoopPipelinerInternal::createKernelLoop(
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    RewriterBase &rewriter,
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap) {
  // Creates the list of initial values associated to values used across
  // stages. The initial values come from the prologue created above.
  // Keep track of the kernel argument associated to each version of the
  // values passed to the kernel.
  llvm::SmallVector<Value> newLoopArg;
  // For existing loop argument initialize them with the right version from the
  // prologue.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    auto defStage = stages.find(def);
    if (defStage != stages.end()) {
      Value valueVersion =
          valueMapping[forOp.getRegionIterArgs()[retVal.index()]]
                      [maxStage - defStage->second];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
    } else
      newLoopArg.push_back(forOp.getInitArgs()[retVal.index()]);
  }
  for (auto escape : crossStageValues) {
    LiverangeInfo &info = escape.second;
    Value value = escape.first;
    for (unsigned stageIdx = 0; stageIdx < info.lastUseStage - info.defStage;
         stageIdx++) {
      Value valueVersion =
          valueMapping[value][maxStage - info.lastUseStage + stageIdx];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
      loopArgMap[std::make_pair(value, info.lastUseStage - info.defStage -
                                           stageIdx)] = newLoopArg.size() - 1;
    }
  }

  // Create the new kernel loop. When we peel the epilgue we need to peel
  // `numStages - 1` iterations. Then we adjust the upper bound to remove those
  // iterations.
  Value newUb = forOp.getUpperBound();
  if (peelEpilogue) {
    Type t = ub.getType();
    Location loc = forOp.getLoc();
    // newUb = ub - maxStage * step
    Value maxStageValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(t, maxStage));
    Value maxStageByStep =
        rewriter.create<arith::MulIOp>(loc, step, maxStageValue);
    newUb = rewriter.create<arith::SubIOp>(loc, ub, maxStageByStep);
  }
  auto newForOp =
      rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), newUb,
                                  forOp.getStep(), newLoopArg);
  newForOp->setAttrs(forOp->getAttrs());
  // When there are no iter args, the loop body terminator will be created.
  // Since we always create it below, remove the terminator if it was created.
  if (!newForOp.getBody()->empty())
    rewriter.eraseOp(newForOp.getBody()->getTerminator());
  return newForOp;
}

LogicalResult LoopPipelinerInternal::createKernel(
    scf::ForOp newForOp,
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
    RewriterBase &rewriter) {
  valueMapping.clear();

  // Create the kernel, we clone instruction based on the order given by
  // user and remap operands coming from a previous stages.
  rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  }
  SmallVector<Value> predicates(maxStage + 1, nullptr);
  if (!peelEpilogue) {
    // Create a predicate for each stage except the last stage.
    Location loc = newForOp.getLoc();
    Type t = ub.getType();
    for (unsigned i = 0; i < maxStage; i++) {
      // c = ub - (maxStage - i) * step
      Value c = rewriter.create<arith::SubIOp>(
          loc, ub,
          rewriter.create<arith::MulIOp>(
              loc, step,
              rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getIntegerAttr(t, int64_t(maxStage - i)))));

      Value pred = rewriter.create<arith::CmpIOp>(
          newForOp.getLoc(), arith::CmpIPredicate::slt,
          newForOp.getInductionVar(), c);
      predicates[i] = pred;
    }
  }
  for (Operation *op : opOrder) {
    int64_t useStage = stages[op];
    auto *newOp = rewriter.clone(*op, mapping);
    SmallVector<OpOperand *> operands;
    // Collect all the operands for the cloned op and its nested ops.
    op->walk([&operands](Operation *nestedOp) {
      for (OpOperand &operand : nestedOp->getOpOperands()) {
        operands.push_back(&operand);
      }
    });
    for (OpOperand *operand : operands) {
      Operation *nestedNewOp = mapping.lookup(operand->getOwner());
      // Special case for the induction variable uses. We replace it with a
      // version incremented based on the stage where it is used.
      if (operand->get() == forOp.getInductionVar()) {
        rewriter.setInsertionPoint(newOp);

        // offset = (maxStage - stages[op]) * step
        Type t = step.getType();
        Value offset = rewriter.create<arith::MulIOp>(
            forOp.getLoc(), step,
            rewriter.create<arith::ConstantOp>(
                forOp.getLoc(),
                rewriter.getIntegerAttr(t, maxStage - stages[op])));
        Value iv = rewriter.create<arith::AddIOp>(
            forOp.getLoc(), newForOp.getInductionVar(), offset);
        nestedNewOp->setOperand(operand->getOperandNumber(), iv);
        rewriter.setInsertionPointAfter(newOp);
        continue;
      }
      Value source = operand->get();
      auto arg = dyn_cast<BlockArgument>(source);
      if (arg && arg.getOwner() == forOp.getBody()) {
        Value ret = forOp.getBody()->getTerminator()->getOperand(
            arg.getArgNumber() - 1);
        Operation *dep = ret.getDefiningOp();
        if (!dep)
          continue;
        auto stageDep = stages.find(dep);
        if (stageDep == stages.end() || stageDep->second == useStage)
          continue;
        // If the value is a loop carried value coming from stage N + 1 remap,
        // it will become a direct use.
        if (stageDep->second == useStage + 1) {
          nestedNewOp->setOperand(operand->getOperandNumber(),
                                  mapping.lookupOrDefault(ret));
          continue;
        }
        source = ret;
      }
      // For operands defined in a previous stage we need to remap it to use
      // the correct region argument. We look for the right version of the
      // Value based on the stage where it is used.
      Operation *def = source.getDefiningOp();
      if (!def)
        continue;
      auto stageDef = stages.find(def);
      if (stageDef == stages.end() || stageDef->second == useStage)
        continue;
      auto remap = loopArgMap.find(
          std::make_pair(operand->get(), useStage - stageDef->second));
      assert(remap != loopArgMap.end());
      nestedNewOp->setOperand(operand->getOperandNumber(),
                              newForOp.getRegionIterArgs()[remap->second]);
    }

    if (predicates[useStage]) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      newOp = predicateFn(rewriter, newOp, predicates[useStage]);
      if (!newOp)
        return failure();
      // Remap the results to the new predicated one.
      for (auto values : llvm::zip(op->getResults(), newOp->getResults()))
        mapping.map(std::get<0>(values), std::get<1>(values));
    }
    if (annotateFn)
      annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Kernel, 0);
  }

  // Collect the Values that need to be returned by the forOp. For each
  // value we need to have `LastUseStage - DefStage` number of versions
  // returned.
  // We create a mapping between original values and the associated loop
  // returned values that will be needed by the epilogue.
  llvm::SmallVector<Value> yieldOperands;
  for (OpOperand &yieldOperand :
       forOp.getBody()->getTerminator()->getOpOperands()) {
    Value source = mapping.lookupOrDefault(yieldOperand.get());
    // When we don't peel the epilogue and the yield value is used outside the
    // loop we need to make sure we return the version from numStages -
    // defStage.
    if (!peelEpilogue &&
        !forOp.getResult(yieldOperand.getOperandNumber()).use_empty()) {
      Operation *def = getDefiningOpAndDistance(yieldOperand.get()).first;
      if (def) {
        auto defStage = stages.find(def);
        if (defStage != stages.end() && defStage->second < maxStage) {
          Value pred = predicates[defStage->second];
          source = rewriter.create<arith::SelectOp>(
              pred.getLoc(), pred, source,
              newForOp.getBody()
                  ->getArguments()[yieldOperand.getOperandNumber() + 1]);
        }
      }
    }
    yieldOperands.push_back(source);
  }

  for (auto &it : crossStageValues) {
    int64_t version = maxStage - it.second.lastUseStage + 1;
    unsigned numVersionReturned = it.second.lastUseStage - it.second.defStage;
    // add the original version to yield ops.
    // If there is a live range spanning across more than 2 stages we need to
    // add extra arg.
    for (unsigned i = 1; i < numVersionReturned; i++) {
      setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                      version++);
      yieldOperands.push_back(
          newForOp.getBody()->getArguments()[yieldOperands.size() + 1 +
                                             newForOp.getNumInductionVars()]);
    }
    setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                    version++);
    yieldOperands.push_back(mapping.lookupOrDefault(it.first));
  }
  // Map the yield operand to the forOp returned value.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    auto defStage = stages.find(def);
    if (defStage == stages.end()) {
      for (unsigned int stage = 1; stage <= maxStage; stage++)
        setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                        retVal.value(), stage);
    } else if (defStage->second > 0) {
      setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                      newForOp->getResult(retVal.index()),
                      maxStage - defStage->second + 1);
    }
  }
  rewriter.create<scf::YieldOp>(forOp.getLoc(), yieldOperands);
  return success();
}

LogicalResult
LoopPipelinerInternal::emitEpilogue(RewriterBase &rewriter,
                                    llvm::SmallVector<Value> &returnValues) {
  Location loc = forOp.getLoc();
  Type t = lb.getType();
  // Emit different versions of the induction variable. They will be
  // removed by dead code if not used.

  auto createConst = [&](int v) {
    return rewriter.create<arith::ConstantOp>(loc,
                                              rewriter.getIntegerAttr(t, v));
  };

  // total_iterations = cdiv(range_diff, step);
  // - range_diff = ub - lb
  // - total_iterations = (range_diff + step + (step < 0 ? 1 : -1)) / step
  Value zero = createConst(0);
  Value one = createConst(1);
  Value stepLessZero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, step, zero);
  Value stepDecr =
      rewriter.create<arith::SelectOp>(loc, stepLessZero, one, createConst(-1));

  Value rangeDiff = rewriter.create<arith::SubIOp>(loc, ub, lb);
  Value rangeIncrStep = rewriter.create<arith::AddIOp>(loc, rangeDiff, step);
  Value rangeDecr =
      rewriter.create<arith::AddIOp>(loc, rangeIncrStep, stepDecr);
  Value totalIterations = rewriter.create<arith::DivSIOp>(loc, rangeDecr, step);

  // If total_iters < max_stage, start the epilogue at zero to match the
  // ramp-up in the prologue.
  // start_iter = max(0, total_iters - max_stage)
  Value iterI = rewriter.create<arith::SubIOp>(loc, totalIterations,
                                               createConst(maxStage));
  iterI = rewriter.create<arith::MaxSIOp>(loc, zero, iterI);

  // Capture predicates for dynamic loops.
  SmallVector<Value> predicates(maxStage + 1);

  for (int64_t i = 1; i <= maxStage; i++) {
    // newLastIter = lb + step * iterI
    Value newlastIter = rewriter.create<arith::AddIOp>(
        loc, lb, rewriter.create<arith::MulIOp>(loc, step, iterI));

    setValueMapping(forOp.getInductionVar(), newlastIter, i);

    // increment to next iterI
    iterI = rewriter.create<arith::AddIOp>(loc, iterI, one);

    if (dynamicLoop) {
      // Disable stages when `i` is greater than total_iters.
      // pred = total_iters >= i
      predicates[i] = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, totalIterations, createConst(i));
    }
  }

  // Emit `maxStage - 1` epilogue part that includes operations from stages
  // [i; maxStage].
  for (int64_t i = 1; i <= maxStage; i++) {
    SmallVector<std::pair<Value, unsigned>> returnMap(returnValues.size());
    for (Operation *op : opOrder) {
      if (stages[op] < i)
        continue;
      unsigned currentVersion = maxStage - stages[op] + i;
      unsigned nextVersion = currentVersion + 1;
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[currentVersion];
              newOperand->set(replacement);
            }
          });
      if (dynamicLoop) {
        OpBuilder::InsertionGuard insertGuard(rewriter);
        newOp = predicateFn(rewriter, newOp, predicates[currentVersion]);
        if (!newOp)
          return failure();
      }
      if (annotateFn)
        annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Epilogue,
                   i - 1);
      for (auto [opRes, newRes] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        setValueMapping(opRes, newRes, currentVersion);
        // If the value is a loop carried dependency update the loop argument
        // mapping and keep track of the last version to replace the original
        // forOp uses.
        for (OpOperand &operand :
             forOp.getBody()->getTerminator()->getOpOperands()) {
          if (operand.get() != opRes)
            continue;
          // If the version is greater than maxStage it means it maps to the
          // original forOp returned value.
          unsigned ri = operand.getOperandNumber();
          returnValues[ri] = newRes;
          Value mapVal = forOp.getRegionIterArgs()[ri];
          returnMap[ri] = std::make_pair(mapVal, currentVersion);
          if (nextVersion <= maxStage)
            setValueMapping(mapVal, newRes, nextVersion);
        }
      }
    }
    if (dynamicLoop) {
      // Select return values from this stage (live outs) based on predication.
      // If the stage is valid select the peeled value, else use previous stage
      // value.
      for (auto pair : llvm::enumerate(returnValues)) {
        unsigned ri = pair.index();
        auto [mapVal, currentVersion] = returnMap[ri];
        if (mapVal) {
          unsigned nextVersion = currentVersion + 1;
          Value pred = predicates[currentVersion];
          Value prevValue = valueMapping[mapVal][currentVersion];
          auto selOp = rewriter.create<arith::SelectOp>(loc, pred, pair.value(),
                                                        prevValue);
          returnValues[ri] = selOp;
          if (nextVersion <= maxStage)
            setValueMapping(mapVal, selOp, nextVersion);
        }
      }
    }
  }
  return success();
}

void LoopPipelinerInternal::setValueMapping(Value key, Value el, int64_t idx) {
  auto it = valueMapping.find(key);
  // If the value is not in the map yet add a vector big enough to store all
  // versions.
  if (it == valueMapping.end())
    it =
        valueMapping
            .insert(std::make_pair(key, llvm::SmallVector<Value>(maxStage + 1)))
            .first;
  it->second[idx] = el;
}

} // namespace

FailureOr<ForOp>
mlir::triton::pipelineForLoop(RewriterBase &rewriter, ForOp forOp,
                              const triton::PipeliningOption &options,
                              bool *modifiedIR) {
  if (modifiedIR)
    *modifiedIR = false;
  LoopPipelinerInternal pipeliner;
  if (!pipeliner.initializeLoopInfo(forOp, options))
    return failure();

  if (modifiedIR)
    *modifiedIR = true;

  // 1. Emit prologue.
  if (failed(pipeliner.emitPrologue(rewriter)))
    return failure();

  // 2. Track values used across stages. When a value cross stages it will
  // need to be passed as loop iteration arguments.
  // We first collect the values that are used in a different stage than where
  // they are defined.
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
      crossStageValues = pipeliner.analyzeCrossStageValues();

  // Mapping between original loop values used cross stage and the block
  // arguments associated after pipelining. A Value may map to several
  // arguments if its liverange spans across more than 2 stages.
  llvm::DenseMap<std::pair<Value, unsigned>, unsigned> loopArgMap;
  // 3. Create the new kernel loop and return the block arguments mapping.
  ForOp newForOp =
      pipeliner.createKernelLoop(crossStageValues, rewriter, loopArgMap);
  // Create the kernel block, order ops based on user choice and remap
  // operands.
  if (failed(pipeliner.createKernel(newForOp, crossStageValues, loopArgMap,
                                    rewriter)))
    return failure();

  llvm::SmallVector<Value> returnValues =
      newForOp.getResults().take_front(forOp->getNumResults());
  if (options.peelEpilogue) {
    // 4. Emit the epilogue after the new forOp.
    rewriter.setInsertionPointAfter(newForOp);
    if (failed(pipeliner.emitEpilogue(rewriter, returnValues)))
      return failure();
  }
  // 5. Erase the original loop and replace the uses with the epilogue output.
  if (forOp->getNumResults() > 0)
    rewriter.replaceOp(forOp, returnValues);
  else
    rewriter.eraseOp(forOp);

  return newForOp;
}
