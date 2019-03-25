//===- Utils.cpp ---- Misc utilities for analysis -------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements miscellaneous analysis routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Utils.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/Builders.h"
#include "mlir/StandardOps/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "analysis-utils"

using namespace mlir;

using llvm::SmallDenseMap;

/// Populates 'loops' with IVs of the loops surrounding 'inst' ordered from
/// the outermost 'affine.for' instruction to the innermost one.
void mlir::getLoopIVs(Instruction &inst, SmallVectorImpl<AffineForOp> *loops) {
  auto *currInst = inst.getParentInst();
  AffineForOp currAffineForOp;
  // Traverse up the hierarchy collecing all 'affine.for' instruction while
  // skipping over 'affine.if' instructions.
  while (currInst && ((currAffineForOp = currInst->dyn_cast<AffineForOp>()) ||
                      currInst->isa<AffineIfOp>())) {
    if (currAffineForOp)
      loops->push_back(currAffineForOp);
    currInst = currInst->getParentInst();
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

  SmallVector<Value *, 4> values(ivs);
  // Append 'ivs' then 'operands' to 'values'.
  values.append(lbOperands[0].begin(), lbOperands[0].end());
  cst->reset(numDims, numSymbols, 0, values);

  // Add loop bound constraints for values which are loop IVs and equality
  // constraints for symbols which are constants.
  for (const auto &value : values) {
    assert(cst->containsId(*value) && "value expected to be present");
    if (isValidSymbol(value)) {
      // Check if the symbol is a constant.
      if (auto *inst = value->getDefiningInst()) {
        if (auto constOp = inst->dyn_cast<ConstantIndexOp>()) {
          cst->setIdToConstant(*value, constOp.getValue());
        }
      }
    } else {
      if (auto loop = getForInductionVarOwner(value)) {
        if (failed(cst->addAffineForOpDomain(loop)))
          return failure();
      }
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
/// represented as constraints symbolic/parameteric in 'loopDepth' loops
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
LogicalResult MemRefRegion::compute(Instruction *inst, unsigned loopDepth,
                                    ComputationSliceState *sliceState) {
  assert((inst->isa<LoadOp>() || inst->isa<StoreOp>()) &&
         "load/store op expected");

  MemRefAccess access(inst);
  memref = access.memref;
  write = access.isStore();

  unsigned rank = access.getRank();

  LLVM_DEBUG(llvm::dbgs() << "MemRefRegion::compute: " << *inst
                          << "depth: " << loopDepth << "\n";);

  if (rank == 0) {
    SmallVector<AffineForOp, 4> ivs;
    getLoopIVs(*inst, &ivs);
    SmallVector<Value *, 8> regionSymbols;
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
  SmallVector<Value *, 4> operands;
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
  // Add inequalties for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    auto *operand = operands[i];
    if (auto loop = getForInductionVarOwner(operand)) {
      // Note that cst can now have more dimensions than accessMap if the
      // bounds expressions involve outer loops or other symbols.
      // TODO(bondhugula): rewrite this to use getInstIndexSet; this way
      // conditionals will be handled when the latter supports it.
      if (failed(cst.addAffineForOpDomain(loop)))
        return failure();
    } else {
      // Has to be a valid symbol.
      auto *symbol = operand;
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto *inst = symbol->getDefiningInst()) {
        if (auto constOp = inst->dyn_cast<ConstantIndexOp>()) {
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
    inst->emitError("getMemRefRegion: compose affine map failed");
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
  getLoopIVs(*inst, &enclosingIVs);
  assert(loopDepth <= enclosingIVs.size() && "invalid loop depth");
  enclosingIVs.resize(loopDepth);
  SmallVector<Value *, 4> ids;
  cst.getIdValues(cst.getNumDimIds(), cst.getNumDimAndSymbolIds(), &ids);
  for (auto *id : ids) {
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
  SmallVector<Value *, 4> memIndices;
  // Indices for the faster buffer being DMAed into/from.
  SmallVector<Value *, 4> bufIndices;

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
  if (memRefType.getNumDynamicDims() > 0)
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
  static_assert(std::is_same<LoadOrStoreOpPointer, LoadOp>::value ||
                    std::is_same<LoadOrStoreOpPointer, StoreOp>::value,
                "argument should be either a LoadOp or a StoreOp");

  Instruction *opInst = loadOrStoreOp->getInstruction();

  MemRefRegion region(opInst->getLoc());
  if (failed(region.compute(opInst, /*loopDepth=*/0)))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "Memory region");
  LLVM_DEBUG(region.getConstraints()->dump());

  bool outOfBounds = false;
  unsigned rank = loadOrStoreOp->getMemRefType().getRank();

  // For each dimension, check for out of bounds.
  for (unsigned r = 0; r < rank; r++) {
    FlatAffineConstraints ucst(*region.getConstraints());

    // Intersect memory region with constraint capturing out of bounds (both out
    // of upper and out of lower), and check if the constraint system is
    // feasible. If it is, there is at least one point out of bounds.
    SmallVector<int64_t, 4> ineq(rank + 1, 0);
    int64_t dimSize = loadOrStoreOp->getMemRefType().getDimSize(r);
    // TODO(bondhugula): handle dynamic dim sizes.
    if (dimSize == -1)
      continue;

    // Check for overflow: d_i >= memref dim size.
    ucst.addConstantLowerBound(r, dimSize);
    outOfBounds = !ucst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp->emitOpError(
          "memref out of upper bound access along dimension #" + Twine(r + 1));
    }

    // Check for a negative index.
    FlatAffineConstraints lcst(*region.getConstraints());
    std::fill(ineq.begin(), ineq.end(), 0);
    // d_i <= -1;
    lcst.addConstantUpperBound(r, -1);
    outOfBounds = !lcst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp->emitOpError(
          "memref out of lower bound access along dimension #" + Twine(r + 1));
    }
  }
  return failure(outOfBounds);
}

// Explicitly instantiate the template so that the compiler knows we need them!
template LogicalResult mlir::boundCheckLoadOrStoreOp(LoadOp loadOp,
                                                     bool emitError);
template LogicalResult mlir::boundCheckLoadOrStoreOp(StoreOp storeOp,
                                                     bool emitError);

// Returns in 'positions' the Block positions of 'inst' in each ancestor
// Block from the Block containing instruction, stopping at 'limitBlock'.
static void findInstPosition(Instruction *inst, Block *limitBlock,
                             SmallVectorImpl<unsigned> *positions) {
  Block *block = inst->getBlock();
  while (block != limitBlock) {
    // FIXME: This algorithm is unnecessarily O(n) and should be improved to not
    // rely on linear scans.
    int instPosInBlock = std::distance(block->begin(), inst->getIterator());
    positions->push_back(instPosInBlock);
    inst = block->getContainingInst();
    block = inst->getBlock();
  }
  std::reverse(positions->begin(), positions->end());
}

// Returns the Instruction in a possibly nested set of Blocks, where the
// position of the instruction is represented by 'positions', which has a
// Block position for each level of nesting.
static Instruction *getInstAtPosition(ArrayRef<unsigned> positions,
                                      unsigned level, Block *block) {
  unsigned i = 0;
  for (auto &inst : *block) {
    if (i != positions[level]) {
      ++i;
      continue;
    }
    if (level == positions.size() - 1)
      return &inst;
    if (auto childAffineForOp = inst.dyn_cast<AffineForOp>())
      return getInstAtPosition(positions, level + 1,
                               childAffineForOp->getBody());

    for (auto &region : inst.getRegions()) {
      for (auto &b : region)
        if (auto *ret = getInstAtPosition(positions, level + 1, &b))
          return ret;
    }
    return nullptr;
  }
  return nullptr;
}

const char *const kSliceFusionBarrierAttrName = "slice_fusion_barrier";
// Computes memref dependence between 'srcAccess' and 'dstAccess', projects
// out any dst loop IVs at depth greater than 'dstLoopDepth', and computes slice
// bounds in 'sliceState' which represent the src IVs in terms of the dst IVs,
// symbols and constants.
LogicalResult mlir::getBackwardComputationSliceState(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned dstLoopDepth, ComputationSliceState *sliceState) {
  bool readReadAccesses =
      srcAccess.opInst->isa<LoadOp>() && dstAccess.opInst->isa<LoadOp>();
  FlatAffineConstraints dependenceConstraints;
  if (!checkMemrefAccessDependence(
          srcAccess, dstAccess, /*loopDepth=*/1, &dependenceConstraints,
          /*dependenceComponents=*/nullptr, /*allowRAR=*/readReadAccesses)) {
    return failure();
  }
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*srcAccess.opInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*dstAccess.opInst, &dstLoopIVs);
  unsigned numDstLoopIVs = dstLoopIVs.size();
  if (dstLoopDepth > numDstLoopIVs) {
    dstAccess.opInst->emitError("invalid destination loop depth");
    return failure();
  }

  // Project out dimensions other than those up to 'dstLoopDepth'.
  dependenceConstraints.projectOut(numSrcLoopIVs + dstLoopDepth,
                                   numDstLoopIVs - dstLoopDepth);

  // Add src loop IV values to 'sliceState'.
  dependenceConstraints.getIdValues(0, numSrcLoopIVs, &sliceState->ivs);

  // Set up lower/upper bound affine maps for the slice.
  sliceState->lbs.resize(numSrcLoopIVs, AffineMap());
  sliceState->ubs.resize(numSrcLoopIVs, AffineMap());

  // Get bounds for src IVs in terms of dst IVs, symbols, and constants.
  dependenceConstraints.getSliceBounds(numSrcLoopIVs,
                                       srcAccess.opInst->getContext(),
                                       &sliceState->lbs, &sliceState->ubs);

  // Set up bound operands for the slice's lower and upper bounds.
  SmallVector<Value *, 4> sliceBoundOperands;
  dependenceConstraints.getIdValues(
      numSrcLoopIVs, dependenceConstraints.getNumDimAndSymbolIds(),
      &sliceBoundOperands);
  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceState->lbOperands.resize(numSrcLoopIVs, sliceBoundOperands);
  sliceState->ubOperands.resize(numSrcLoopIVs, sliceBoundOperands);

  llvm::SmallDenseSet<Value *, 8> sequentialLoops;
  if (readReadAccesses) {
    // For read-read access pairs, clear any slice bounds on sequential loops.
    // Get sequential loops in loop nest rooted at 'srcLoopIVs[0]'.
    getSequentialLoops(srcLoopIVs[0], &sequentialLoops);
  }
  // Clear all sliced loop bounds beginning at the first sequential loop, or
  // first loop with a slice fusion barrier attribute..
  // TODO(andydavis, bondhugula) Use MemRef read/write regions instead of
  // using 'kSliceFusionBarrierAttrName'.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    Value *iv = srcLoopIVs[i]->getInductionVar();
    if (sequentialLoops.count(iv) == 0 &&
        srcLoopIVs[i]->getAttr(kSliceFusionBarrierAttrName) == nullptr)
      continue;
    for (unsigned j = i; j < numSrcLoopIVs; ++j) {
      sliceState->lbs[j] = AffineMap();
      sliceState->ubs[j] = AffineMap();
    }
    break;
  }

  return success();
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
AffineForOp mlir::insertBackwardComputationSlice(
    Instruction *srcOpInst, Instruction *dstOpInst, unsigned dstLoopDepth,
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

  // Find the inst block positions of 'srcOpInst' within 'srcLoopIVs'.
  SmallVector<unsigned, 4> positions;
  // TODO(andydavis): This code is incorrect since srcLoopIVs can be 0-d.
  findInstPosition(srcOpInst, srcLoopIVs[0]->getInstruction()->getBlock(),
                   &positions);

  // Clone src loop nest and insert it a the beginning of the instruction block
  // of the loop at 'dstLoopDepth' in 'dstLoopIVs'.
  auto dstAffineForOp = dstLoopIVs[dstLoopDepth - 1];
  FuncBuilder b(dstAffineForOp->getBody(), dstAffineForOp->getBody()->begin());
  auto sliceLoopNest =
      b.clone(*srcLoopIVs[0]->getInstruction())->cast<AffineForOp>();

  Instruction *sliceInst =
      getInstAtPosition(positions, /*level=*/0, sliceLoopNest->getBody());
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
      forOp->setLowerBound(sliceState->lbOperands[i], lbMap);
    if (AffineMap ubMap = sliceState->ubs[i])
      forOp->setUpperBound(sliceState->ubOperands[i], ubMap);
  }
  return sliceLoopNest;
}

// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
MemRefAccess::MemRefAccess(Instruction *loadOrStoreOpInst) {
  if (auto loadOp = loadOrStoreOpInst->dyn_cast<LoadOp>()) {
    memref = loadOp->getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp->getMemRefType();
    indices.reserve(loadMemrefType.getRank());
    for (auto *index : loadOp->getIndices()) {
      indices.push_back(index);
    }
  } else {
    assert(loadOrStoreOpInst->isa<StoreOp>() && "load/store op expected");
    auto storeOp = loadOrStoreOpInst->dyn_cast<StoreOp>();
    opInst = loadOrStoreOpInst;
    memref = storeOp->getMemRef();
    auto storeMemrefType = storeOp->getMemRefType();
    indices.reserve(storeMemrefType.getRank());
    for (auto *index : storeOp->getIndices()) {
      indices.push_back(index);
    }
  }
}

unsigned MemRefAccess::getRank() const {
  return memref->getType().cast<MemRefType>().getRank();
}

bool MemRefAccess::isStore() const { return opInst->isa<StoreOp>(); }

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
unsigned mlir::getNestingDepth(Instruction &inst) {
  Instruction *currInst = &inst;
  unsigned depth = 0;
  while ((currInst = currInst->getParentInst())) {
    if (currInst->isa<AffineForOp>())
      depth++;
  }
  return depth;
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned mlir::getNumCommonSurroundingLoops(Instruction &A, Instruction &B) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getLoopIVs(A, &loopsA);
  getLoopIVs(B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i]->getInstruction() != loopsB[i]->getInstruction())
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

static Optional<int64_t> getMemoryFootprintBytes(Block &block,
                                                 Block::iterator start,
                                                 Block::iterator end,
                                                 int memorySpace) {
  SmallDenseMap<Value *, std::unique_ptr<MemRefRegion>, 4> regions;

  // Walk this 'affine.for' instruction to gather all memory regions.
  bool error = false;
  block.walk(start, end, [&](Instruction *opInst) {
    if (!opInst->isa<LoadOp>() && !opInst->isa<StoreOp>()) {
      // Neither load nor a store op.
      return;
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = llvm::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(
            region->compute(opInst,
                            /*loopDepth=*/getNestingDepth(*block.begin())))) {
      opInst->emitError("Error obtaining memory region\n");
      error = true;
      return;
    }
    auto it = regions.find(region->memref);
    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
      error = true;
      return;
    }
  });

  if (error)
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
  auto *forInst = forOp->getInstruction();
  return ::getMemoryFootprintBytes(
      *forInst->getBlock(), Block::iterator(forInst),
      std::next(Block::iterator(forInst)), memorySpace);
}

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void mlir::getSequentialLoops(
    AffineForOp forOp, llvm::SmallDenseSet<Value *, 8> *sequentialLoops) {
  forOp->getInstruction()->walk([&](Instruction *inst) {
    if (auto innerFor = inst->dyn_cast<AffineForOp>())
      if (!isLoopParallel(innerFor))
        sequentialLoops->insert(innerFor->getInductionVar());
  });
}

/// Returns true if 'forOp' is parallel.
bool mlir::isLoopParallel(AffineForOp forOp) {
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Instruction *, 8> loadAndStoreOpInsts;
  forOp->getInstruction()->walk([&](Instruction *opInst) {
    if (opInst->isa<LoadOp>() || opInst->isa<StoreOp>())
      loadAndStoreOpInsts.push_back(opInst);
  });

  // Dep check depth would be number of enclosing loops + 1.
  unsigned depth = getNestingDepth(*forOp->getInstruction()) + 1;

  // Check dependences between all pairs of ops in 'loadAndStoreOpInsts'.
  for (auto *srcOpInst : loadAndStoreOpInsts) {
    MemRefAccess srcAccess(srcOpInst);
    for (auto *dstOpInst : loadAndStoreOpInsts) {
      MemRefAccess dstAccess(dstOpInst);
      FlatAffineConstraints dependenceConstraints;
      if (checkMemrefAccessDependence(srcAccess, dstAccess, depth,
                                      &dependenceConstraints,
                                      /*dependenceComponents=*/nullptr))
        return false;
    }
  }
  return true;
}
