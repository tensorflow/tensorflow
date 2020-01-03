//===- Utils.cpp ---- Misc utilities for analysis -------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous analysis routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Utils.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "analysis-utils"

using namespace mlir;

using llvm::SmallDenseMap;

/// Populates 'loops' with IVs of the loops surrounding 'op' ordered from
/// the outermost 'affine.for' operation to the innermost one.
void mlir::getLoopIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops) {
  auto *currOp = op.getParentOp();
  AffineForOp currAffineForOp;
  // Traverse up the hierarchy collecting all 'affine.for' operation while
  // skipping over 'affine.if' operations.
  while (currOp && ((currAffineForOp = dyn_cast<AffineForOp>(currOp)) ||
                    isa<AffineIfOp>(currOp))) {
    if (currAffineForOp)
      loops->push_back(currAffineForOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(loops->begin(), loops->end());
}

// Populates 'cst' with FlatAffineConstraints which represent slice bounds.
LogicalResult
ComputationSliceState::getAsConstraints(FlatAffineConstraints *cst) {
  assert(!lbOperands.empty());
  // Adds src 'ivs' as dimension identifiers in 'cst'.
  unsigned numDims = ivs.size();
  // Adds operands (dst ivs and symbols) as symbols in 'cst'.
  unsigned numSymbols = lbOperands[0].size();

  SmallVector<Value, 4> values(ivs);
  // Append 'ivs' then 'operands' to 'values'.
  values.append(lbOperands[0].begin(), lbOperands[0].end());
  cst->reset(numDims, numSymbols, 0, values);

  // Add loop bound constraints for values which are loop IVs and equality
  // constraints for symbols which are constants.
  for (const auto &value : values) {
    assert(cst->containsId(*value) && "value expected to be present");
    if (isValidSymbol(value)) {
      // Check if the symbol is a constant.

      if (auto cOp = dyn_cast_or_null<ConstantIndexOp>(value->getDefiningOp()))
        cst->setIdToConstant(*value, cOp.getValue());
    } else if (auto loop = getForInductionVarOwner(value)) {
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }

  // Add slices bounds on 'ivs' using maps 'lbs'/'ubs' with 'lbOperands[0]'
  LogicalResult ret = cst->addSliceBounds(ivs, lbs, ubs, lbOperands[0]);
  assert(succeeded(ret) &&
         "should not fail as we never have semi-affine slice maps");
  (void)ret;
  return success();
}

// Clears state bounds and operand state.
void ComputationSliceState::clearBounds() {
  lbs.clear();
  ubs.clear();
  lbOperands.clear();
  ubOperands.clear();
}

unsigned MemRefRegion::getRank() const {
  return memref->getType().cast<MemRefType>().getRank();
}

Optional<int64_t> MemRefRegion::getConstantBoundingSizeAndShape(
    SmallVectorImpl<int64_t> *shape, std::vector<SmallVector<int64_t, 4>> *lbs,
    SmallVectorImpl<int64_t> *lbDivisors) const {
  auto memRefType = memref->getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  if (shape)
    shape->reserve(rank);

  assert(rank == cst.getNumDimIds() && "inconsistent memref region");

  // Find a constant upper bound on the extent of this memref region along each
  // dimension.
  int64_t numElements = 1;
  int64_t diffConstant;
  int64_t lbDivisor;
  for (unsigned d = 0; d < rank; d++) {
    SmallVector<int64_t, 4> lb;
    Optional<int64_t> diff = cst.getConstantBoundOnDimSize(d, &lb, &lbDivisor);
    if (diff.hasValue()) {
      diffConstant = diff.getValue();
      assert(lbDivisor > 0);
    } else {
      // If no constant bound is found, then it can always be bound by the
      // memref's dim size if the latter has a constant size along this dim.
      auto dimSize = memRefType.getDimSize(d);
      if (dimSize == -1)
        return None;
      diffConstant = dimSize;
      // Lower bound becomes 0.
      lb.resize(cst.getNumSymbolIds() + 1, 0);
      lbDivisor = 1;
    }
    numElements *= diffConstant;
    if (lbs) {
      lbs->push_back(lb);
      assert(lbDivisors && "both lbs and lbDivisor or none");
      lbDivisors->push_back(lbDivisor);
    }
    if (shape) {
      shape->push_back(diffConstant);
    }
  }
  return numElements;
}

LogicalResult MemRefRegion::unionBoundingBox(const MemRefRegion &other) {
  assert(memref == other.memref);
  return cst.unionBoundingBox(*other.getConstraints());
}

/// Computes the memory region accessed by this memref with the region
/// represented as constraints symbolic/parametric in 'loopDepth' loops
/// surrounding opInst and any additional Function symbols.
//  For example, the memref region for this load operation at loopDepth = 1 will
//  be as below:
//
//    affine.for %i = 0 to 32 {
//      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
// region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineConstraints symbolic in %i.
//
// TODO(bondhugula): extend this to any other memref dereferencing ops
// (dma_start, dma_wait).
LogicalResult MemRefRegion::compute(Operation *op, unsigned loopDepth,
                                    ComputationSliceState *sliceState,
                                    bool addMemRefDimBounds) {
  assert((isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) &&
         "affine load/store op expected");

  MemRefAccess access(op);
  memref = access.memref;
  write = access.isStore();

  unsigned rank = access.getRank();

  LLVM_DEBUG(llvm::dbgs() << "MemRefRegion::compute: " << *op
                          << "depth: " << loopDepth << "\n";);

  if (rank == 0) {
    SmallVector<AffineForOp, 4> ivs;
    getLoopIVs(*op, &ivs);
    SmallVector<Value, 8> regionSymbols;
    extractForInductionVars(ivs, &regionSymbols);
    // A rank 0 memref has a 0-d region.
    cst.reset(rank, loopDepth, 0, regionSymbols);
    return success();
  }

  // Build the constraints for this region.
  AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  AffineMap accessMap = accessValueMap.getAffineMap();

  unsigned numDims = accessMap.getNumDims();
  unsigned numSymbols = accessMap.getNumSymbols();
  unsigned numOperands = accessValueMap.getNumOperands();
  // Merge operands with slice operands.
  SmallVector<Value, 4> operands;
  operands.resize(numOperands);
  for (unsigned i = 0; i < numOperands; ++i)
    operands[i] = accessValueMap.getOperand(i);

  if (sliceState != nullptr) {
    operands.reserve(operands.size() + sliceState->lbOperands[0].size());
    // Append slice operands to 'operands' as symbols.
    for (auto extraOperand : sliceState->lbOperands[0]) {
      if (!llvm::is_contained(operands, extraOperand)) {
        operands.push_back(extraOperand);
        numSymbols++;
      }
    }
  }
  // We'll first associate the dims and symbols of the access map to the dims
  // and symbols resp. of cst. This will change below once cst is
  // fully constructed out.
  cst.reset(numDims, numSymbols, 0, operands);

  // Add equality constraints.
  // Add inequalities for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    auto operand = operands[i];
    if (auto loop = getForInductionVarOwner(operand)) {
      // Note that cst can now have more dimensions than accessMap if the
      // bounds expressions involve outer loops or other symbols.
      // TODO(bondhugula): rewrite this to use getInstIndexSet; this way
      // conditionals will be handled when the latter supports it.
      if (failed(cst.addAffineForOpDomain(loop)))
        return failure();
    } else {
      // Has to be a valid symbol.
      auto symbol = operand;
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto *op = symbol->getDefiningOp()) {
        if (auto constOp = dyn_cast<ConstantIndexOp>(op)) {
          cst.setIdToConstant(*symbol, constOp.getValue());
        }
      }
    }
  }

  // Add lower/upper bounds on loop IVs using bounds from 'sliceState'.
  if (sliceState != nullptr) {
    // Add dim and symbol slice operands.
    for (auto operand : sliceState->lbOperands[0]) {
      cst.addInductionVarOrTerminalSymbol(operand);
    }
    // Add upper/lower bounds from 'sliceState' to 'cst'.
    LogicalResult ret =
        cst.addSliceBounds(sliceState->ivs, sliceState->lbs, sliceState->ubs,
                           sliceState->lbOperands[0]);
    assert(succeeded(ret) &&
           "should not fail as we never have semi-affine slice maps");
    (void)ret;
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  if (failed(cst.composeMap(&accessValueMap))) {
    op->emitError("getMemRefRegion: compose affine map failed");
    LLVM_DEBUG(accessValueMap.getAffineMap().dump());
    return failure();
  }

  // Set all identifiers appearing after the first 'rank' identifiers as
  // symbolic identifiers - so that the ones corresponding to the memref
  // dimensions are the dimensional identifiers for the memref region.
  cst.setDimSymbolSeparation(cst.getNumDimAndSymbolIds() - rank);

  // Eliminate any loop IVs other than the outermost 'loopDepth' IVs, on which
  // this memref region is symbolic.
  SmallVector<AffineForOp, 4> enclosingIVs;
  getLoopIVs(*op, &enclosingIVs);
  assert(loopDepth <= enclosingIVs.size() && "invalid loop depth");
  enclosingIVs.resize(loopDepth);
  SmallVector<Value, 4> ids;
  cst.getIdValues(cst.getNumDimIds(), cst.getNumDimAndSymbolIds(), &ids);
  for (auto id : ids) {
    AffineForOp iv;
    if ((iv = getForInductionVarOwner(id)) &&
        llvm::is_contained(enclosingIVs, iv) == false) {
      cst.projectOut(id);
    }
  }

  // Project out any local variables (these would have been added for any
  // mod/divs).
  cst.projectOut(cst.getNumDimAndSymbolIds(), cst.getNumLocalIds());

  // Constant fold any symbolic identifiers.
  cst.constantFoldIdRange(/*pos=*/cst.getNumDimIds(),
                          /*num=*/cst.getNumSymbolIds());

  assert(cst.getNumDimIds() == rank && "unexpected MemRefRegion format");

  // Add upper/lower bounds for each memref dimension with static size
  // to guard against potential over-approximation from projection.
  // TODO(andydavis) Support dynamic memref dimensions.
  if (addMemRefDimBounds) {
    auto memRefType = memref->getType().cast<MemRefType>();
    for (unsigned r = 0; r < rank; r++) {
      cst.addConstantLowerBound(r, 0);
      int64_t dimSize = memRefType.getDimSize(r);
      if (ShapedType::isDynamic(dimSize))
        continue;
      cst.addConstantUpperBound(r, dimSize - 1);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Memory region:\n");
  LLVM_DEBUG(cst.dump());
  return success();
}

//  TODO(mlir-team): improve/complete this when we have target data.
static unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Returns the size of the region.
Optional<int64_t> MemRefRegion::getRegionSize() {
  auto memRefType = memref->getType().cast<MemRefType>();

  auto layoutMaps = memRefType.getAffineMaps();
  if (layoutMaps.size() > 1 ||
      (layoutMaps.size() == 1 && !layoutMaps[0].isIdentity())) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return false;
  }

  // Indices to use for the DmaStart op.
  // Indices for the original memref being DMAed from/to.
  SmallVector<Value, 4> memIndices;
  // Indices for the faster buffer being DMAed into/from.
  SmallVector<Value, 4> bufIndices;

  // Compute the extents of the buffer.
  Optional<int64_t> numElements = getConstantBoundingSizeAndShape();
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Dynamic shapes not yet supported\n");
    return None;
  }
  return getMemRefEltSizeInBytes(memRefType) * numElements.getValue();
}

/// Returns the size of memref data in bytes if it's statically shaped, None
/// otherwise.  If the element of the memref has vector type, takes into account
/// size of the vector as well.
//  TODO(mlir-team): improve/complete this when we have target data.
Optional<uint64_t> mlir::getMemRefSizeInBytes(MemRefType memRefType) {
  if (!memRefType.hasStaticShape())
    return None;
  auto elementType = memRefType.getElementType();
  if (!elementType.isIntOrFloat() && !elementType.isa<VectorType>())
    return None;

  uint64_t sizeInBytes = getMemRefEltSizeInBytes(memRefType);
  for (unsigned i = 0, e = memRefType.getRank(); i < e; i++) {
    sizeInBytes = sizeInBytes * memRefType.getDimSize(i);
  }
  return sizeInBytes;
}

template <typename LoadOrStoreOpPointer>
LogicalResult mlir::boundCheckLoadOrStoreOp(LoadOrStoreOpPointer loadOrStoreOp,
                                            bool emitError) {
  static_assert(std::is_same<LoadOrStoreOpPointer, AffineLoadOp>::value ||
                    std::is_same<LoadOrStoreOpPointer, AffineStoreOp>::value,
                "argument should be either a AffineLoadOp or a AffineStoreOp");

  Operation *opInst = loadOrStoreOp.getOperation();
  MemRefRegion region(opInst->getLoc());
  if (failed(region.compute(opInst, /*loopDepth=*/0, /*sliceState=*/nullptr,
                            /*addMemRefDimBounds=*/false)))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "Memory region");
  LLVM_DEBUG(region.getConstraints()->dump());

  bool outOfBounds = false;
  unsigned rank = loadOrStoreOp.getMemRefType().getRank();

  // For each dimension, check for out of bounds.
  for (unsigned r = 0; r < rank; r++) {
    FlatAffineConstraints ucst(*region.getConstraints());

    // Intersect memory region with constraint capturing out of bounds (both out
    // of upper and out of lower), and check if the constraint system is
    // feasible. If it is, there is at least one point out of bounds.
    SmallVector<int64_t, 4> ineq(rank + 1, 0);
    int64_t dimSize = loadOrStoreOp.getMemRefType().getDimSize(r);
    // TODO(bondhugula): handle dynamic dim sizes.
    if (dimSize == -1)
      continue;

    // Check for overflow: d_i >= memref dim size.
    ucst.addConstantLowerBound(r, dimSize);
    outOfBounds = !ucst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of upper bound access along dimension #" << (r + 1);
    }

    // Check for a negative index.
    FlatAffineConstraints lcst(*region.getConstraints());
    std::fill(ineq.begin(), ineq.end(), 0);
    // d_i <= -1;
    lcst.addConstantUpperBound(r, -1);
    outOfBounds = !lcst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of lower bound access along dimension #" << (r + 1);
    }
  }
  return failure(outOfBounds);
}

// Explicitly instantiate the template so that the compiler knows we need them!
template LogicalResult mlir::boundCheckLoadOrStoreOp(AffineLoadOp loadOp,
                                                     bool emitError);
template LogicalResult mlir::boundCheckLoadOrStoreOp(AffineStoreOp storeOp,
                                                     bool emitError);

// Returns in 'positions' the Block positions of 'op' in each ancestor
// Block from the Block containing operation, stopping at 'limitBlock'.
static void findInstPosition(Operation *op, Block *limitBlock,
                             SmallVectorImpl<unsigned> *positions) {
  Block *block = op->getBlock();
  while (block != limitBlock) {
    // FIXME: This algorithm is unnecessarily O(n) and should be improved to not
    // rely on linear scans.
    int instPosInBlock = std::distance(block->begin(), op->getIterator());
    positions->push_back(instPosInBlock);
    op = block->getParentOp();
    block = op->getBlock();
  }
  std::reverse(positions->begin(), positions->end());
}

// Returns the Operation in a possibly nested set of Blocks, where the
// position of the operation is represented by 'positions', which has a
// Block position for each level of nesting.
static Operation *getInstAtPosition(ArrayRef<unsigned> positions,
                                    unsigned level, Block *block) {
  unsigned i = 0;
  for (auto &op : *block) {
    if (i != positions[level]) {
      ++i;
      continue;
    }
    if (level == positions.size() - 1)
      return &op;
    if (auto childAffineForOp = dyn_cast<AffineForOp>(op))
      return getInstAtPosition(positions, level + 1,
                               childAffineForOp.getBody());

    for (auto &region : op.getRegions()) {
      for (auto &b : region)
        if (auto *ret = getInstAtPosition(positions, level + 1, &b))
          return ret;
    }
    return nullptr;
  }
  return nullptr;
}

// Adds loop IV bounds to 'cst' for loop IVs not found in 'ivs'.
LogicalResult addMissingLoopIVBounds(SmallPtrSet<Value, 8> &ivs,
                                     FlatAffineConstraints *cst) {
  for (unsigned i = 0, e = cst->getNumDimIds(); i < e; ++i) {
    auto value = cst->getIdValue(i);
    if (ivs.count(value) == 0) {
      assert(isForInductionVar(value));
      auto loop = getForInductionVarOwner(value);
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

// Returns the innermost common loop depth for the set of operations in 'ops'.
// TODO(andydavis) Move this to LoopUtils.
static unsigned
getInnermostCommonLoopDepth(ArrayRef<Operation *> ops,
                            SmallVectorImpl<AffineForOp> &surroundingLoops) {
  unsigned numOps = ops.size();
  assert(numOps > 0);

  std::vector<SmallVector<AffineForOp, 4>> loops(numOps);
  unsigned loopDepthLimit = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < numOps; ++i) {
    getLoopIVs(*ops[i], &loops[i]);
    loopDepthLimit =
        std::min(loopDepthLimit, static_cast<unsigned>(loops[i].size()));
  }

  unsigned loopDepth = 0;
  for (unsigned d = 0; d < loopDepthLimit; ++d) {
    unsigned i;
    for (i = 1; i < numOps; ++i) {
      if (loops[i - 1][d] != loops[i][d])
        return loopDepth;
    }
    surroundingLoops.push_back(loops[i - 1][d]);
    ++loopDepth;
  }
  return loopDepth;
}

/// Computes in 'sliceUnion' the union of all slice bounds computed at
/// 'loopDepth' between all dependent pairs of ops in 'opsA' and 'opsB'.
/// Returns 'Success' if union was computed, 'failure' otherwise.
LogicalResult mlir::computeSliceUnion(ArrayRef<Operation *> opsA,
                                      ArrayRef<Operation *> opsB,
                                      unsigned loopDepth,
                                      unsigned numCommonLoops,
                                      bool isBackwardSlice,
                                      ComputationSliceState *sliceUnion) {
  // Compute the union of slice bounds between all pairs in 'opsA' and
  // 'opsB' in 'sliceUnionCst'.
  FlatAffineConstraints sliceUnionCst;
  assert(sliceUnionCst.getNumDimAndSymbolIds() == 0);
  std::vector<std::pair<Operation *, Operation *>> dependentOpPairs;
  for (unsigned i = 0, numOpsA = opsA.size(); i < numOpsA; ++i) {
    MemRefAccess srcAccess(opsA[i]);
    for (unsigned j = 0, numOpsB = opsB.size(); j < numOpsB; ++j) {
      MemRefAccess dstAccess(opsB[j]);
      if (srcAccess.memref != dstAccess.memref)
        continue;
      // Check if 'loopDepth' exceeds nesting depth of src/dst ops.
      if ((!isBackwardSlice && loopDepth > getNestingDepth(*opsA[i])) ||
          (isBackwardSlice && loopDepth > getNestingDepth(*opsB[j]))) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid loop depth\n.");
        return failure();
      }

      bool readReadAccesses = isa<AffineLoadOp>(srcAccess.opInst) &&
                              isa<AffineLoadOp>(dstAccess.opInst);
      FlatAffineConstraints dependenceConstraints;
      // Check dependence between 'srcAccess' and 'dstAccess'.
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
          &dependenceConstraints, /*dependenceComponents=*/nullptr,
          /*allowRAR=*/readReadAccesses);
      if (result.value == DependenceResult::Failure) {
        LLVM_DEBUG(llvm::dbgs() << "Dependence check failed\n.");
        return failure();
      }
      if (result.value == DependenceResult::NoDependence)
        continue;
      dependentOpPairs.push_back({opsA[i], opsB[j]});

      // Compute slice bounds for 'srcAccess' and 'dstAccess'.
      ComputationSliceState tmpSliceState;
      mlir::getComputationSliceState(opsA[i], opsB[j], &dependenceConstraints,
                                     loopDepth, isBackwardSlice,
                                     &tmpSliceState);

      if (sliceUnionCst.getNumDimAndSymbolIds() == 0) {
        // Initialize 'sliceUnionCst' with the bounds computed in previous step.
        if (failed(tmpSliceState.getAsConstraints(&sliceUnionCst))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Unable to compute slice bound constraints\n.");
          return failure();
        }
        assert(sliceUnionCst.getNumDimAndSymbolIds() > 0);
        continue;
      }

      // Compute constraints for 'tmpSliceState' in 'tmpSliceCst'.
      FlatAffineConstraints tmpSliceCst;
      if (failed(tmpSliceState.getAsConstraints(&tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute slice bound constraints\n.");
        return failure();
      }

      // Align coordinate spaces of 'sliceUnionCst' and 'tmpSliceCst' if needed.
      if (!sliceUnionCst.areIdsAlignedWithOther(tmpSliceCst)) {

        // Pre-constraint id alignment: record loop IVs used in each constraint
        // system.
        SmallPtrSet<Value, 8> sliceUnionIVs;
        for (unsigned k = 0, l = sliceUnionCst.getNumDimIds(); k < l; ++k)
          sliceUnionIVs.insert(sliceUnionCst.getIdValue(k));
        SmallPtrSet<Value, 8> tmpSliceIVs;
        for (unsigned k = 0, l = tmpSliceCst.getNumDimIds(); k < l; ++k)
          tmpSliceIVs.insert(tmpSliceCst.getIdValue(k));

        sliceUnionCst.mergeAndAlignIdsWithOther(/*offset=*/0, &tmpSliceCst);

        // Post-constraint id alignment: add loop IV bounds missing after
        // id alignment to constraint systems. This can occur if one constraint
        // system uses an loop IV that is not used by the other. The call
        // to unionBoundingBox below expects constraints for each Loop IV, even
        // if they are the unsliced full loop bounds added here.
        if (failed(addMissingLoopIVBounds(sliceUnionIVs, &sliceUnionCst)))
          return failure();
        if (failed(addMissingLoopIVBounds(tmpSliceIVs, &tmpSliceCst)))
          return failure();
      }
      // Compute union bounding box of 'sliceUnionCst' and 'tmpSliceCst'.
      if (sliceUnionCst.getNumLocalIds() > 0 ||
          tmpSliceCst.getNumLocalIds() > 0 ||
          failed(sliceUnionCst.unionBoundingBox(tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute union bounding box of slice bounds."
                      "\n.");
        return failure();
      }
    }
  }

  // Empty union.
  if (sliceUnionCst.getNumDimAndSymbolIds() == 0)
    return failure();

  // Gather loops surrounding ops from loop nest where slice will be inserted.
  SmallVector<Operation *, 4> ops;
  for (auto &dep : dependentOpPairs) {
    ops.push_back(isBackwardSlice ? dep.second : dep.first);
  }
  SmallVector<AffineForOp, 4> surroundingLoops;
  unsigned innermostCommonLoopDepth =
      getInnermostCommonLoopDepth(ops, surroundingLoops);
  if (loopDepth > innermostCommonLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "Exceeds max loop depth\n.");
    return failure();
  }

  // Store 'numSliceLoopIVs' before converting dst loop IVs to dims.
  unsigned numSliceLoopIVs = sliceUnionCst.getNumDimIds();

  // Convert any dst loop IVs which are symbol identifiers to dim identifiers.
  sliceUnionCst.convertLoopIVSymbolsToDims();
  sliceUnion->clearBounds();
  sliceUnion->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceUnion->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get slice bounds from slice union constraints 'sliceUnionCst'.
  sliceUnionCst.getSliceBounds(/*offset=*/0, numSliceLoopIVs,
                               opsA[0]->getContext(), &sliceUnion->lbs,
                               &sliceUnion->ubs);

  // Add slice bound operands of union.
  SmallVector<Value, 4> sliceBoundOperands;
  sliceUnionCst.getIdValues(numSliceLoopIVs,
                            sliceUnionCst.getNumDimAndSymbolIds(),
                            &sliceBoundOperands);

  // Copy src loop IVs from 'sliceUnionCst' to 'sliceUnion'.
  sliceUnion->ivs.clear();
  sliceUnionCst.getIdValues(0, numSliceLoopIVs, &sliceUnion->ivs);

  // Set loop nest insertion point to block start at 'loopDepth'.
  sliceUnion->insertPoint =
      isBackwardSlice
          ? surroundingLoops[loopDepth - 1].getBody()->begin()
          : std::prev(surroundingLoops[loopDepth - 1].getBody()->end());

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceUnion->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceUnion->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  return success();
}

const char *const kSliceFusionBarrierAttrName = "slice_fusion_barrier";
// Computes slice bounds by projecting out any loop IVs from
// 'dependenceConstraints' at depth greater than 'loopDepth', and computes slice
// bounds in 'sliceState' which represent the one loop nest's IVs in terms of
// the other loop nest's IVs, symbols and constants (using 'isBackwardsSlice').
void mlir::getComputationSliceState(
    Operation *depSourceOp, Operation *depSinkOp,
    FlatAffineConstraints *dependenceConstraints, unsigned loopDepth,
    bool isBackwardSlice, ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*depSourceOp, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*depSinkOp, &dstLoopIVs);
  unsigned numDstLoopIVs = dstLoopIVs.size();

  assert((!isBackwardSlice && loopDepth <= numSrcLoopIVs) ||
         (isBackwardSlice && loopDepth <= numDstLoopIVs));

  // Project out dimensions other than those up to 'loopDepth'.
  unsigned pos = isBackwardSlice ? numSrcLoopIVs + loopDepth : loopDepth;
  unsigned num =
      isBackwardSlice ? numDstLoopIVs - loopDepth : numSrcLoopIVs - loopDepth;
  dependenceConstraints->projectOut(pos, num);

  // Add slice loop IV values to 'sliceState'.
  unsigned offset = isBackwardSlice ? 0 : loopDepth;
  unsigned numSliceLoopIVs = isBackwardSlice ? numSrcLoopIVs : numDstLoopIVs;
  dependenceConstraints->getIdValues(offset, offset + numSliceLoopIVs,
                                     &sliceState->ivs);

  // Set up lower/upper bound affine maps for the slice.
  sliceState->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceState->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get bounds for slice IVs in terms of other IVs, symbols, and constants.
  dependenceConstraints->getSliceBounds(offset, numSliceLoopIVs,
                                        depSourceOp->getContext(),
                                        &sliceState->lbs, &sliceState->ubs);

  // Set up bound operands for the slice's lower and upper bounds.
  SmallVector<Value, 4> sliceBoundOperands;
  unsigned numDimsAndSymbols = dependenceConstraints->getNumDimAndSymbolIds();
  for (unsigned i = 0; i < numDimsAndSymbols; ++i) {
    if (i < offset || i >= offset + numSliceLoopIVs) {
      sliceBoundOperands.push_back(dependenceConstraints->getIdValue(i));
    }
  }

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceState->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceState->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Set destination loop nest insertion point to block start at 'dstLoopDepth'.
  sliceState->insertPoint =
      isBackwardSlice ? dstLoopIVs[loopDepth - 1].getBody()->begin()
                      : std::prev(srcLoopIVs[loopDepth - 1].getBody()->end());

  llvm::SmallDenseSet<Value, 8> sequentialLoops;
  if (isa<AffineLoadOp>(depSourceOp) && isa<AffineLoadOp>(depSinkOp)) {
    // For read-read access pairs, clear any slice bounds on sequential loops.
    // Get sequential loops in loop nest rooted at 'srcLoopIVs[0]'.
    getSequentialLoops(isBackwardSlice ? srcLoopIVs[0] : dstLoopIVs[0],
                       &sequentialLoops);
  }
  // Clear all sliced loop bounds beginning at the first sequential loop, or
  // first loop with a slice fusion barrier attribute..
  // TODO(andydavis, bondhugula) Use MemRef read/write regions instead of
  // using 'kSliceFusionBarrierAttrName'.
  auto getSliceLoop = [&](unsigned i) {
    return isBackwardSlice ? srcLoopIVs[i] : dstLoopIVs[i];
  };
  for (unsigned i = 0; i < numSliceLoopIVs; ++i) {
    Value iv = getSliceLoop(i).getInductionVar();
    if (sequentialLoops.count(iv) == 0 &&
        getSliceLoop(i).getAttr(kSliceFusionBarrierAttrName) == nullptr)
      continue;
    for (unsigned j = i; j < numSliceLoopIVs; ++j) {
      sliceState->lbs[j] = AffineMap();
      sliceState->ubs[j] = AffineMap();
    }
    break;
  }
}

/// Creates a computation slice of the loop nest surrounding 'srcOpInst',
/// updates the slice loop bounds with any non-null bound maps specified in
/// 'sliceState', and inserts this slice into the loop nest surrounding
/// 'dstOpInst' at loop depth 'dstLoopDepth'.
// TODO(andydavis,bondhugula): extend the slicing utility to compute slices that
// aren't necessarily a one-to-one relation b/w the source and destination. The
// relation between the source and destination could be many-to-many in general.
// TODO(andydavis,bondhugula): the slice computation is incorrect in the cases
// where the dependence from the source to the destination does not cover the
// entire destination index set. Subtract out the dependent destination
// iterations from destination index set and check for emptiness --- this is one
// solution.
AffineForOp
mlir::insertBackwardComputationSlice(Operation *srcOpInst, Operation *dstOpInst,
                                     unsigned dstLoopDepth,
                                     ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*dstOpInst, &dstLoopIVs);
  unsigned dstLoopIVsSize = dstLoopIVs.size();
  if (dstLoopDepth > dstLoopIVsSize) {
    dstOpInst->emitError("invalid destination loop depth");
    return AffineForOp();
  }

  // Find the op block positions of 'srcOpInst' within 'srcLoopIVs'.
  SmallVector<unsigned, 4> positions;
  // TODO(andydavis): This code is incorrect since srcLoopIVs can be 0-d.
  findInstPosition(srcOpInst, srcLoopIVs[0].getOperation()->getBlock(),
                   &positions);

  // Clone src loop nest and insert it a the beginning of the operation block
  // of the loop at 'dstLoopDepth' in 'dstLoopIVs'.
  auto dstAffineForOp = dstLoopIVs[dstLoopDepth - 1];
  OpBuilder b(dstAffineForOp.getBody(), dstAffineForOp.getBody()->begin());
  auto sliceLoopNest =
      cast<AffineForOp>(b.clone(*srcLoopIVs[0].getOperation()));

  Operation *sliceInst =
      getInstAtPosition(positions, /*level=*/0, sliceLoopNest.getBody());
  // Get loop nest surrounding 'sliceInst'.
  SmallVector<AffineForOp, 4> sliceSurroundingLoops;
  getLoopIVs(*sliceInst, &sliceSurroundingLoops);

  // Sanity check.
  unsigned sliceSurroundingLoopsSize = sliceSurroundingLoops.size();
  (void)sliceSurroundingLoopsSize;
  assert(dstLoopDepth + numSrcLoopIVs >= sliceSurroundingLoopsSize);
  unsigned sliceLoopLimit = dstLoopDepth + numSrcLoopIVs;
  (void)sliceLoopLimit;
  assert(sliceLoopLimit >= sliceSurroundingLoopsSize);

  // Update loop bounds for loops in 'sliceLoopNest'.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    auto forOp = sliceSurroundingLoops[dstLoopDepth + i];
    if (AffineMap lbMap = sliceState->lbs[i])
      forOp.setLowerBound(sliceState->lbOperands[i], lbMap);
    if (AffineMap ubMap = sliceState->ubs[i])
      forOp.setUpperBound(sliceState->ubOperands[i], ubMap);
  }
  return sliceLoopNest;
}

// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
MemRefAccess::MemRefAccess(Operation *loadOrStoreOpInst) {
  if (auto loadOp = dyn_cast<AffineLoadOp>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();
    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineStoreOp>(loadOrStoreOpInst) && "load/store op expected");
    auto storeOp = dyn_cast<AffineStoreOp>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();
    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }
}

unsigned MemRefAccess::getRank() const {
  return memref->getType().cast<MemRefType>().getRank();
}

bool MemRefAccess::isStore() const { return isa<AffineStoreOp>(opInst); }

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
unsigned mlir::getNestingDepth(Operation &op) {
  Operation *currOp = &op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<AffineForOp>(currOp))
      depth++;
  }
  return depth;
}

/// Equal if both affine accesses are provably equivalent (at compile
/// time) when considering the memref, the affine maps and their respective
/// operands. The equality of access functions + operands is checked by
/// subtracting fully composed value maps, and then simplifying the difference
/// using the expression flattener.
/// TODO: this does not account for aliasing of memrefs.
bool MemRefAccess::operator==(const MemRefAccess &rhs) const {
  if (memref != rhs.memref)
    return false;

  AffineValueMap diff, thisMap, rhsMap;
  getAccessMap(&thisMap);
  rhs.getAccessMap(&rhsMap);
  AffineValueMap::difference(thisMap, rhsMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned mlir::getNumCommonSurroundingLoops(Operation &A, Operation &B) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getLoopIVs(A, &loopsA);
  getLoopIVs(B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i].getOperation() != loopsB[i].getOperation())
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

static Optional<int64_t> getMemoryFootprintBytes(Block &block,
                                                 Block::iterator start,
                                                 Block::iterator end,
                                                 int memorySpace) {
  SmallDenseMap<Value, std::unique_ptr<MemRefRegion>, 4> regions;

  // Walk this 'affine.for' operation to gather all memory regions.
  auto result = block.walk(start, end, [&](Operation *opInst) -> WalkResult {
    if (!isa<AffineLoadOp>(opInst) && !isa<AffineStoreOp>(opInst)) {
      // Neither load nor a store op.
      return WalkResult::advance();
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(
            region->compute(opInst,
                            /*loopDepth=*/getNestingDepth(*block.begin())))) {
      return opInst->emitError("error obtaining memory region\n");
    }

    auto it = regions.find(region->memref);
    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      return opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return None;

  int64_t totalSizeInBytes = 0;
  for (const auto &region : regions) {
    Optional<int64_t> size = region.second->getRegionSize();
    if (!size.hasValue())
      return None;
    totalSizeInBytes += size.getValue();
  }
  return totalSizeInBytes;
}

Optional<int64_t> mlir::getMemoryFootprintBytes(AffineForOp forOp,
                                                int memorySpace) {
  auto *forInst = forOp.getOperation();
  return ::getMemoryFootprintBytes(
      *forInst->getBlock(), Block::iterator(forInst),
      std::next(Block::iterator(forInst)), memorySpace);
}

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void mlir::getSequentialLoops(AffineForOp forOp,
                              llvm::SmallDenseSet<Value, 8> *sequentialLoops) {
  forOp.getOperation()->walk([&](Operation *op) {
    if (auto innerFor = dyn_cast<AffineForOp>(op))
      if (!isLoopParallel(innerFor))
        sequentialLoops->insert(innerFor.getInductionVar());
  });
}

/// Returns true if 'forOp' is parallel.
bool mlir::isLoopParallel(AffineForOp forOp) {
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  auto walkResult = forOp.walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst))
      loadAndStoreOpInsts.push_back(opInst);
    else if (!isa<AffineForOp>(opInst) && !isa<AffineTerminatorOp>(opInst) &&
             !isa<AffineIfOp>(opInst) && !opInst->hasNoSideEffect())
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  // Stop early if the loop has unknown ops with side effects.
  if (walkResult.wasInterrupted())
    return false;

  // Dep check depth would be number of enclosing loops + 1.
  unsigned depth = getNestingDepth(*forOp.getOperation()) + 1;

  // Check dependences between all pairs of ops in 'loadAndStoreOpInsts'.
  for (auto *srcOpInst : loadAndStoreOpInsts) {
    MemRefAccess srcAccess(srcOpInst);
    for (auto *dstOpInst : loadAndStoreOpInsts) {
      MemRefAccess dstAccess(dstOpInst);
      FlatAffineConstraints dependenceConstraints;
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, depth, &dependenceConstraints,
          /*dependenceComponents=*/nullptr);
      if (result.value != DependenceResult::NoDependence)
        return false;
    }
  }
  return true;
}
