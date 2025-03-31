#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace tt = mlir::triton;

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;
constexpr int64_t kDefaultMaxPrograms = 2 << 15; // 65536

std::optional<int64_t> maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  if (lowerBound && upperBound && step)
    return constantTripCount(*lowerBound, *upperBound, *step);
  return {};
}

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

void inferResultRangesPID(Operation *op, uint64_t max,
                          SetIntRangeFn setResultRange) {
  assert(op->getNumResults() == 1 && "expected op to have one result");
  auto result = op->getResult(0);
  assert(llvm::isa<IntegerType>(result.getType()) &&
         "expected result type to be int");
  IntegerType resTy = llvm::cast<IntegerType>(result.getType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(resTy);
  setResultRange(result, ConstantIntRanges::range(
                             /*min*/ {/*numBits*/ bitWidth, /*val*/ 0,
                                      /*isSigned*/ resTy.isSigned()},
                             /*max*/
                             {/*numBits*/ bitWidth, /*val*/ max,
                              /*isSigned*/ resTy.isSigned()},
                             /*isSigned*/ resTy.isSigned()));
}

void inferResultRanges(tt::MakeRangeOp *op, SetIntRangeFn setResultRange) {
  auto result = op->getResult();
  RankedTensorType resTy = result.getType();
  assert(llvm::isa<IntegerType>(resTy.getElementType()) && "expected int type");
  IntegerType elTy = llvm::cast<IntegerType>(resTy.getElementType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(elTy);
  setResultRange(result,
                 ConstantIntRanges::range(
                     /*min*/ {/*numBits*/ bitWidth, /*val*/ op->getStart(),
                              /*isSigned*/ elTy.isSigned()},
                     /*max*/
                     {/*numBits*/ bitWidth, /*val*/ op->getEnd(),
                      /*isSigned*/ elTy.isSigned()},
                     /*isSigned*/ elTy.isSigned()));
}

void inferResultRanges(tt::GatherOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  assert(argRanges.size() == 2 && "expected two arg ranges");
  setResultRange(op->getResult(), argRanges[0]);
}

void inferResultRangesUnaryOpForwardArgRange(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0]);
}

void inferResultRangesBinaryOpUnionArgRanges(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  assert(op->getNumOperands() == 2 && "expected op to have two operands");
  assert(argRanges.size() == 2 && "expected two arg ranges");
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0].rangeUnion(argRanges[1]));
}

void inferResultRangesMaxNonNegSigned(Operation *op,
                                      SetIntRangeFn setResultRange) {
  for (auto result : op->getResults()) {
    auto bitWidth =
        mlir::ConstantIntRanges::getStorageBitwidth(result.getType());
    setResultRange(result, ConstantIntRanges::fromSigned(
                               APInt::getZero(bitWidth).sext(bitWidth),
                               APInt::getMaxValue(bitWidth).sext(bitWidth)));
  }
}

std::optional<ConstantIntRanges> maybeGetAssumedRange(Operation *assumption,
                                                      Value anchor) {
  arith::CmpIOp cmpOp = llvm::dyn_cast<arith::CmpIOp>(assumption);
  if (!cmpOp) {
    emitRemark(assumption->getLoc(), "unsupported assumption operation");
    return {};
  }

  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult:
    emitRemark(assumption->getLoc(),
               "unsigned arithmetic not currently supported");
    return {};
  default:
    break;
  }
  bool isSigned = true;

  bool anchorIsLhs = cmpOp.getLhs() == anchor;
  auto maybeConstantIntValue = getConstantIntValue(
      getAsOpFoldResult(anchorIsLhs ? cmpOp.getRhs() : cmpOp.getLhs()));
  if (auto constValue = maybeConstantIntValue) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
    assert(bitWidth > 0 && "expected non-zero bitwdith");
    // This is always true in
    APInt apVal = {bitWidth, static_cast<uint64_t>(*constValue), isSigned},
          min = APInt::getSignedMinValue(bitWidth),
          max = APInt::getSignedMaxValue(bitWidth);

    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return mlir::ConstantIntRanges::constant(apVal);
    case arith::CmpIPredicate::sge: {
      // K >= apVal implies K ∈ [apVal, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal, max, isSigned);
      // apVal >= K implies K ∈ [min, apVal]
      return mlir::ConstantIntRanges::range(min, apVal, isSigned);
    }
    case arith::CmpIPredicate::sgt: {
      // K > apVal implies K >= apVal + 1 implies K ∈ [apVal + 1, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
      // apVal > K implies apVal - 1 >= K implies K ∈ [min, apVal - 1]
      return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
    }
    case arith::CmpIPredicate::sle: {
      // K <= apVal implies K ∈ [min, apVal]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal, isSigned);
      // apVal <= K implies K ∈ [apVal, max]
      return mlir::ConstantIntRanges::range(apVal, max, isSigned);
    }
    case arith::CmpIPredicate::slt: {
      // K < apVal implies K <= apVal -1 implies K ∈ [min, apVal - 1]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
      // apVal < K implies apVal + 1 <= K implies K ∈ [apVal + 1, max]
      return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
    }
    default:
      emitRemark(cmpOp.getLoc(), "unsupported cmp predicate for assumption");
      return {};
    }
  }
  return {};
}

} // namespace

namespace mlir::triton::AMD {

bool isEmptyInitializedRange(ConstantIntRanges rv) {
  if (!rv.umin().getBitWidth() || !rv.umax().getBitWidth() ||
      !rv.smin().getBitWidth() || !rv.smax().getBitWidth())
    return true;
  return false;
}

std::optional<SmallVector<ConstantIntRanges>>
collectRanges(const DataFlowSolver &solver, ValueRange values) {
  SmallVector<ConstantIntRanges> ranges;
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return {};
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    if (isEmptyInitializedRange(inferredRange))
      return {};
    ranges.push_back(inferredRange);
  }
  return ranges;
}

bool cmpIIsStaticallyTrue(const DataFlowSolver &solver, arith::CmpIOp cmpOp) {
  if (auto inputRanges =
          collectRanges(solver, ValueRange{cmpOp.getOperands()})) {
    intrange::CmpPredicate pred =
        static_cast<intrange::CmpPredicate>(cmpOp.getPredicate());
    return intrange::evaluatePred(pred, (*inputRanges)[0], (*inputRanges)[1])
        .value_or(false);
  }
  return false;
}

std::optional<ConstantIntRanges>
TritonIntegerRangeAnalysis::maybeGetAssumedRange(Value anchor) const {
  auto matchingAssumptions = this->assumptions.lookup(anchor);
  if (matchingAssumptions.empty())
    return {};

  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
  assert(bitWidth > 0 && "expected non-zero bitwidth");
  ConstantIntRanges constIntRange = ConstantIntRanges::maxRange(bitWidth);
  for (auto assumption : matchingAssumptions) {
    if (auto constIntRange_ = ::maybeGetAssumedRange(assumption, anchor))
      constIntRange = constIntRange.intersection(*constIntRange_);
  }
  return constIntRange;
}

void TritonIntegerRangeAnalysis::setToEntryState(
    dataflow::IntegerValueRangeLattice *lattice) {
  auto anchor = lattice->getAnchor();
  IntegerValueRange range = IntegerValueRange::getMaxRange(anchor);
  if (auto maybeRange = maybeGetAssumedRange(anchor))
    range = *maybeRange;
  auto changed = lattice->join(range);
  LLVM_DEBUG({
    if (changed == ChangeResult::Change) {
      DBGS() << "Set range of ";
      anchor.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << " to " << range << "\n";
    }
  });
  propagateIfChanged(lattice, changed);
}

LogicalResult TritonIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  LDBG("Inferring ranges for " << *op);
  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &resultsLattices,
                       this](Value v, const IntegerValueRange &incomingRange) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    dataflow::IntegerValueRangeLattice *lattice =
        resultsLattices[result.getResultNumber()];
    ChangeResult changed = lattice->join(incomingRange);
    LLVM_DEBUG({
      if (changed == ChangeResult::Change) {
        DBGS() << "Inferred range for ";
        v.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << " to " << incomingRange << "\n";
      }
    });
    propagateIfChanged(lattice, changed);
  };

  // Initialize lattices with assumptions.
  for (const auto &resultLattice : resultsLattices) {
    if (!resultLattice->getValue().isUninitialized())
      continue;
    auto anchor = resultLattice->getAnchor();
    if (auto assumptions = this->assumptions.lookup(anchor);
        !assumptions.empty()) {
      setToEntryState(resultLattice);
      return success();
    }
  }

  // Ops with fixed/constant ranges.
  if (llvm::isa<GetProgramIdOp, MakeRangeOp, HistogramOp, GetNumProgramsOp>(
          op)) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<GetProgramIdOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms - 1, joinCallback);
        })
        .Case<GetNumProgramsOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<MakeRangeOp>([&](MakeRangeOp makeROp) {
          inferResultRanges(&makeROp, joinCallback);
        })
        .Case<HistogramOp>([&](HistogramOp histOp) {
          return inferResultRangesMaxNonNegSigned(histOp, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  SmallVector<IntegerValueRange> argIntValueRanges = llvm::map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  // Ops with actually changing/variable input/output ranges.
  if (llvm::isa<TransOp, SplitOp, BroadcastOp, ReshapeOp, gpu::ConvertLayoutOp,
                SplatOp, ExpandDimsOp, JoinOp, CatOp, GatherOp>(op)) {
    SmallVector<ConstantIntRanges> argConstIntRanges;
    for (const auto &r : argIntValueRanges) {
      if (r.isUninitialized()) {
        setAllToEntryStates(resultsLattices);
        return success();
      }
      argConstIntRanges.push_back(r.getValue());
    }
    llvm::TypeSwitch<Operation *>(op)
        .Case<TransOp, SplitOp, BroadcastOp, ExpandDimsOp, SplatOp, ReshapeOp,
              gpu::ConvertLayoutOp>([&](auto) {
          return inferResultRangesUnaryOpForwardArgRange(op, argConstIntRanges,
                                                         joinCallback);
        })
        .Case<JoinOp, CatOp>([&](auto joinOp) {
          return inferResultRangesBinaryOpUnionArgRanges(
              joinOp, argConstIntRanges, joinCallback);
        })
        .Case<GatherOp>([&](GatherOp gatherOp) {
          return inferResultRanges(&gatherOp, argConstIntRanges, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    inferrable.inferResultRangesFromOptional(argIntValueRanges, joinCallback);
    return success();
  }

  setAllToEntryStates(resultsLattices);
  return success();
}

void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  SmallVector<dataflow::IntegerValueRangeLattice *> lattices;
  for (auto abstractLat : abstractLattices) {
    lattices.push_back(
        static_cast<dataflow::IntegerValueRangeLattice *>(abstractLat));
  }
  // Initialize loop trip counts
  LoopLikeOpInterface loop =
      llvm::dyn_cast<LoopLikeOpInterface>(branch.getOperation());
  if (loop) {
    if (!loopTripCounts.contains(loop)) {
      SmallVector<LoopLikeOpInterface> loops{loop};
      getEnclosingLoops(*loop, loops);
      int loopTripCount =
          std::accumulate(loops.begin(), loops.end(), 1,
                          [](int accum, LoopLikeOpInterface loop) {
                            return accum * maybeGetTripCount(loop).value_or(
                                               kDefaultMaxTripCount + 1);
                          });
      loopTripCounts[loop] = loopTripCount;
    }
    for (auto argLat : lattices) {
      if (!loopVisits.contains({loop, argLat})) {
        loopVisits[{loop, argLat}] = 0;
      }
    }
  }

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");
  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }
    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty()) {
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        }
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #tripcount times, stop propagating.
      if (loop && loopVisits[loopArgLat] >= loopTripCounts[loop])
        continue;
      ChangeResult changed;
      if (loop && loopTripCounts[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).
        changed = argLat->join(IntegerValueRange::getMaxRange(oper));
      } else {
        // Else, propagate pred operands.
        changed = argLat->join(*getLatticeElementFor(point, oper));
      }
      propagateIfChanged(argLat, changed);
      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (loop && changed == ChangeResult::Change)
        ++loopVisits[loopArgLat];
    }
  }
}

DenseMap<Value, SetVector<Operation *>>
TritonIntegerRangeAnalysis::collectAssumptions(Operation *rootOp,
                                               bool filterConstants) {
  DenseMap<Value, SetVector<Operation *>> assumptions;
  rootOp->walk([&](LLVM::AssumeOp op) {
    auto assump = op.getCond().getDefiningOp();
    for (auto operand : assump->getOperands()) {
      if (filterConstants && getConstantIntValue(operand))
        continue;
      assumptions[operand].insert(assump);
    }
  });
  return assumptions;
}

} // namespace mlir::triton::AMD
