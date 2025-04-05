#ifndef TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H
#define TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::triton::AMD {
/// This struct (analysis) adapt's upstream's IntegerRangeAnalysis (inferring
/// lower/upperbounds on integer constants) to our needs.
/// Specifically there are 2 points of extension:
///
/// 1. Support for GetProgramIdOp, MakeRangeOp, SplatOp, ExpandDimsOp. *Note*,
/// upstream already supports range inference for shaped types such as tensors
/// (here we just implement effectively implement the interfaces for our ops).
///    * Upstream's semantics for "range of shape type" is union over ranges of
///    elements.
///    * We do not use tablegen to implement
///    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
///    in order to keep the entire implementation contained/encapsulated.
///
/// 2. Support for inference "through loops". Upstream's analysis conservatively
/// inferences [min_int, max_int] for loop carried values (and therefore loop
/// body values). Here we attempt to do better by analysis the loop bounds and
/// "abstractly interpreting" the loop when loop bounds are statically known.
/// See visitRegionSuccessors.
struct TritonIntegerRangeAnalysis : dataflow::IntegerRangeAnalysis {
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;
  TritonIntegerRangeAnalysis(
      DataFlowSolver &solver,
      const DenseMap<Value, SetVector<Operation *>> &assumptions)
      : dataflow::IntegerRangeAnalysis(solver), assumptions(assumptions) {}

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) override;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) override;

  /// This method (which overloads
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors)
  /// implements "abstract interpretation" of loops with statically known bounds
  /// in order to infer tight ranges for loop carried values (and therefore loop
  /// body values). By "abstract interpretation" we mean lattice states are
  /// propagated to all region successors N times, where N is the total trip
  /// count of the loop. Recall for scf.for, both the loop itself and the users
  /// of the loop successors. Thus, after N propagations both loop body values
  /// and users of loop results will have accurate ranges (assuming we have
  /// implemented support for range analysis on the ops).
  /// *Note*, this implementation is majority similar to
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors
  /// (so check there for more explanation/insight) and basically only does two
  /// things differently:
  ///
  /// 1. If the branch op is a loop (LoopLikeOpInterface) then we attempt to
  /// compute its total trip count (nested loop trip counts multiply) and
  /// initialize a visit count to 0. Note, due to how Dataflow analysis works we
  /// have to actually visit the loop N times for each iter_arg (each argument
  /// lattice) so we actually track visit count for (loop, arg) not just (loop).
  ///
  /// 2. Before propagating, we check if we have propagated for (loop, arg) >= N
  /// times. If so, we do not propagate (and thus the traversal converges/ends).
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops with a
  /// total trip count larger than `kDefaultMaxTripCount`, fallback to
  /// upstream's conservative inference (i.e., we infer [min_int, max_int]) for
  /// the loop operands and all users and all users of the results of the loop.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override;

  /// Collect all operands that participate in assumptions (see description of
  /// `assumptions` field below) under the rootOp. By default, operands that can
  /// be folded to constants are excluded.
  static DenseMap<Value, SetVector<Operation *>>
  collectAssumptions(Operation *rootOp, bool filterConstants = true);

  /// Construct the tightest/narrowest range possible using all the assumptions
  /// that `anchor` participates in. For example, the pattern
  ///   %assumesltlhs = arith.cmpi sge, %K, %c0 : i32
  ///   llvm.intr.assume %assumesltlhs : i1
  ///   %assumesltlhs = arith.cmpi slt, %K, %c128 : i32
  ///   llvm.intr.assume %assumesltlhs : i1
  /// for %K, will produce a final range
  ///   [0, 2147483647] ∩ [-2147483648, 128] = [0, 128]
  std::optional<ConstantIntRanges> maybeGetAssumedRange(Value anchor) const;

  /// Trip counts of all loops with static loop bounds contained under the root
  /// operation being analyzed. Note, nested loops have trip counts computed as
  /// a product of enclosing loops; i.e. for
  ///   scf.for i = 1 to 10
  ///     scf.for j = 1 to 10
  /// the trip count of the outer loop (on i) is 10 but the trip count of the
  /// inner loop (on j) is 100.
  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopTripCounts;

  /// Visit counts tabulating how many times each lattice has been propagated
  /// through each loop. This is used in visitRegionSuccessors to end
  /// propagation when loopVisits[loop, lattice] reaches loopTripCounts[loop].
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;

  /// `assumptions` maps from values to (possibly) any operations that satisfy
  /// the pattern
  ///   %assumesltlhs = arith.cmpi sge, %K, %c0 : i32
  ///   llvm.intr.assume %assumesltlhs : i1
  ///   %assumesltlhs = arith.cmpi slt, %K, %c128 : i32
  ///   llvm.intr.assume %assumesltlhs : i1
  /// If one uses collectAssumptions below then `assumptions` will look like
  /// %K -> {arith.cmpi slt..., arith.cmpi sge}.
  llvm::DenseMap<Value, SetVector<Operation *>> assumptions;
};

std::optional<SmallVector<ConstantIntRanges>>
collectRanges(const DataFlowSolver &solver, ValueRange values);

bool cmpIIsStaticallyTrue(const DataFlowSolver &solver, arith::CmpIOp cmpOp);

bool isEmptyInitializedRange(ConstantIntRanges rv);

} // namespace mlir::triton::AMD

#endif
