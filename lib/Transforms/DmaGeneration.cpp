//===- DmaGeneration.cpp - DMA generation pass ------------------------ -*-===//
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
// This file implements a pass to automatically promote accessed memref regions
// to buffers in a faster memory space that is explicitly managed, with the
// necessary data movement operations expressed as DMAs.
//
//===----------------------------------------------------------------------===//

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "dma-generate"

using namespace mlir;
using llvm::SmallMapVector;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<unsigned long long> clFastMemoryCapacity(
    "dma-fast-mem-capacity",
    llvm::cl::init(std::numeric_limits<uint64_t>::max() / 1024),
    llvm::cl::desc(
        "Set fast memory space capacity in KiB (default: unlimited)"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clFastMemorySpace(
    "dma-fast-mem-space", llvm::cl::init(1),
    llvm::cl::desc(
        "Fast memory space identifier for DMA generation (default: 1)"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clSkipNonUnitStrideLoop(
    "dma-skip-non-unit-stride-loops", llvm::cl::Hidden, llvm::cl::init(false),
    llvm::cl::desc("Testing purposes: avoid non-unit stride loop choice depths "
                   "for DMA placement"),
    llvm::cl::cat(clOptionsCategory));

namespace {

/// Replaces all loads and stores on memref's living in 'slowMemorySpace' by
/// introducing DMA operations (strided DMA if necessary) to transfer data into
/// `fastMemorySpace` and rewriting the original load's/store's to instead
/// load/store from the allocated fast memory buffers. Additional options
/// specify the identifier corresponding to the fast memory space and the amount
/// of fast memory space available. The pass traverses through the nesting
/// structure, recursing to inner levels if necessary to determine at what depth
/// DMA transfers need to be placed so that the allocated buffers fit within the
/// memory capacity provided.
// TODO(bondhugula): We currently can't generate DMAs correctly when stores are
// strided. Check for strided stores.
struct DmaGeneration : public FunctionPass<DmaGeneration> {
  explicit DmaGeneration(unsigned slowMemorySpace = 0,
                         unsigned fastMemorySpace = clFastMemorySpace,
                         int minDmaTransferSize = 1024,
                         uint64_t fastMemCapacityBytes = clFastMemoryCapacity *
                                                         1024)
      : slowMemorySpace(slowMemorySpace), fastMemorySpace(fastMemorySpace),
        minDmaTransferSize(minDmaTransferSize),
        fastMemCapacityBytes(fastMemCapacityBytes) {}

  void runOnFunction() override;
  bool runOnBlock(Block *block);
  uint64_t runOnBlock(Block::iterator begin, Block::iterator end);

  bool generateDma(const MemRefRegion &region, Block *block,
                   Block::iterator begin, Block::iterator end,
                   uint64_t *sizeInBytes, Block::iterator *nBegin,
                   Block::iterator *nEnd);

  // List of memory regions to DMA for. We need a map vector to have a
  // guaranteed iteration order to write test cases. CHECK-DAG doesn't help here
  // since the alloc's for example are identical except for the SSA id.
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> readRegions;
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> writeRegions;

  // Map from original memref's to the DMA buffers that their accesses are
  // replaced with.
  DenseMap<Value *, Value *> fastBufferMap;

  // Slow memory space associated with DMAs.
  const unsigned slowMemorySpace;
  // Fast memory space associated with DMAs.
  unsigned fastMemorySpace;
  // Minimum DMA transfer size supported by the target in bytes.
  const int minDmaTransferSize;
  // Capacity of the faster memory space.
  uint64_t fastMemCapacityBytes;

  // Constant zero index to avoid too many duplicates.
  Value *zeroIndex = nullptr;
};

} // end anonymous namespace

/// Generates DMAs for memref's living in 'slowMemorySpace' into newly created
/// buffers in 'fastMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
FunctionPassBase *mlir::createDmaGenerationPass(unsigned slowMemorySpace,
                                                unsigned fastMemorySpace,
                                                int minDmaTransferSize,
                                                uint64_t fastMemCapacityBytes) {
  return new DmaGeneration(slowMemorySpace, fastMemorySpace, minDmaTransferSize,
                           fastMemCapacityBytes);
}

// Info comprising stride and number of elements transferred every stride.
struct StrideInfo {
  int64_t stride;
  int64_t numEltPerStride;
};

/// Returns striding information for a copy/transfer of this region with
/// potentially multiple striding levels from outermost to innermost. For an
/// n-dimensional region, there can be at most n-1 levels of striding
/// successively nested.
//  TODO(bondhugula): make this work with non-identity layout maps.
static void getMultiLevelStrides(const MemRefRegion &region,
                                 ArrayRef<int64_t> bufferShape,
                                 SmallVectorImpl<StrideInfo> *strideInfos) {
  if (bufferShape.size() <= 1)
    return;

  int64_t numEltPerStride = 1;
  int64_t stride = 1;
  for (int d = bufferShape.size() - 1; d >= 1; d--) {
    int64_t dimSize = region.memref->getType().cast<MemRefType>().getDimSize(d);
    stride *= dimSize;
    numEltPerStride *= bufferShape[d];
    // A stride is needed only if the region has a shorter extent than the
    // memref along the dimension *and* has an extent greater than one along the
    // next major dimension.
    if (bufferShape[d] < dimSize && bufferShape[d - 1] > 1) {
      strideInfos->push_back({stride, numEltPerStride});
    }
  }
}

/// Construct the memref region to just include the entire memref. Returns false
/// dynamic shaped memref's for now. `numParamLoopIVs` is the number of
/// enclosing loop IVs of opInst (starting from the outermost) that the region
/// is parametric on.
static bool getFullMemRefAsRegion(Instruction *opInst, unsigned numParamLoopIVs,
                                  MemRefRegion *region) {
  unsigned rank;
  if (auto loadOp = opInst->dyn_cast<LoadOp>()) {
    rank = loadOp->getMemRefType().getRank();
    region->memref = loadOp->getMemRef();
    region->setWrite(false);
  } else if (auto storeOp = opInst->dyn_cast<StoreOp>()) {
    rank = storeOp->getMemRefType().getRank();
    region->memref = storeOp->getMemRef();
    region->setWrite(true);
  } else {
    assert(false && "expected load or store op");
    return false;
  }
  auto memRefType = region->memref->getType().cast<MemRefType>();
  if (memRefType.getNumDynamicDims() > 0)
    return false;

  auto *regionCst = region->getConstraints();

  // Just get the first numSymbols IVs, which the memref region is parametric
  // on.
  SmallVector<OpPointer<AffineForOp>, 4> ivs;
  getLoopIVs(*opInst, &ivs);
  ivs.resize(numParamLoopIVs);
  SmallVector<Value *, 4> symbols;
  extractForInductionVars(ivs, &symbols);
  regionCst->reset(rank, numParamLoopIVs, 0);
  regionCst->setIdValues(rank, rank + numParamLoopIVs, symbols);

  // Memref dim sizes provide the bounds.
  for (unsigned d = 0; d < rank; d++) {
    auto dimSize = memRefType.getDimSize(d);
    assert(dimSize > 0 && "filtered dynamic shapes above");
    regionCst->addConstantLowerBound(d, 0);
    regionCst->addConstantUpperBound(d, dimSize - 1);
  }
  return true;
}

static void emitNoteForBlock(const Block &block, const Twine &message) {
  auto *inst = block.getContainingInst();
  if (!inst) {
    block.getFunction()->emitNote(message);
  } else {
    inst->emitNote(message);
  }
}

/// Creates a buffer in the faster memory space for the specified region;
/// generates a DMA from the lower memory space to this one, and replaces all
/// loads to load from that buffer. Returns false if DMAs could not be generated
/// due to yet unimplemented cases. `begin` and `end` specify the insertion
/// points where the incoming DMAs and outgoing DMAs, respectively, should
/// be inserted (the insertion happens right before the insertion point). Since
/// `begin` can itself be invalidated due to the memref rewriting done from this
/// method, the output argument `nBegin` is set to its replacement (set
/// to `begin` if no invalidation happens). Since outgoing DMAs are inserted at
/// `end`, the output argument `nEnd` is set to the one following the original
/// end (since the latter could have been invalidated/replaced). `sizeInBytes`
/// is set to the size of the DMA buffer allocated.
bool DmaGeneration::generateDma(const MemRefRegion &region, Block *block,
                                Block::iterator begin, Block::iterator end,
                                uint64_t *sizeInBytes, Block::iterator *nBegin,
                                Block::iterator *nEnd) {
  *nBegin = begin;
  *nEnd = end;

  if (begin == end)
    return true;

  // DMAs for read regions are going to be inserted just before the for loop.
  FuncBuilder prologue(block, begin);
  // DMAs for write regions are going to be inserted just after the for loop.
  FuncBuilder epilogue(block, end);
  FuncBuilder *b = region.isWrite() ? &epilogue : &prologue;

  // Builder to create constants at the top level.
  auto *func = block->getFunction();
  FuncBuilder top(func);

  auto loc = region.loc;
  auto *memref = region.memref;
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

  unsigned rank = memRefType.getRank();
  SmallVector<int64_t, 4> fastBufferShape;

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  Optional<int64_t> numElements = region.getConstantBoundingSizeAndShape(
      &fastBufferShape, &lbs, &lbDivisors);
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
    return false;
  }

  if (numElements.getValue() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to DMA\n");
    *sizeInBytes = 0;
    return true;
  }

  const FlatAffineConstraints *cst = region.getConstraints();
  // 'regionSymbols' hold values that this memory region is symbolic/paramteric
  // on; these typically include loop IVs surrounding the level at which the DMA
  // generation is being done or other valid symbols in MLIR.
  SmallVector<Value *, 8> regionSymbols;
  cst->getIdValues(rank, cst->getNumIds(), &regionSymbols);

  // Construct the index expressions for the fast memory buffer. The index
  // expression for a particular dimension of the fast buffer is obtained by
  // subtracting out the lower bound on the original memref's data region
  // along the corresponding dimension.

  // Index start offsets for faster memory buffer relative to the original.
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; d++) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);

    // Set DMA start location for this dimension in the lower memory space
    // memref.
    if (auto caf = offset.dyn_cast<AffineConstantExpr>()) {
      auto indexVal = caf.getValue();
      if (indexVal == 0) {
        memIndices.push_back(zeroIndex);
      } else {
        memIndices.push_back(
            top.create<ConstantIndexOp>(loc, indexVal)->getResult());
      }
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      auto map = top.getAffineMap(
          cst->getNumDimIds() + cst->getNumSymbolIds() - rank, 0, offset, {});
      memIndices.push_back(b->create<AffineApplyOp>(loc, map, regionSymbols));
    }
    // The fast buffer is DMAed into at location zero; addressing is relative.
    bufIndices.push_back(zeroIndex);

    // Record the offsets since they are needed to remap the memory accesses of
    // the original memref further below.
    offsets.push_back(offset);
  }

  // The faster memory space buffer.
  Value *fastMemRef;

  // Check if a buffer was already created.
  bool existingBuf = fastBufferMap.count(memref) > 0;
  if (!existingBuf) {
    auto fastMemRefType = top.getMemRefType(
        fastBufferShape, memRefType.getElementType(), {}, fastMemorySpace);

    // Create the fast memory space buffer just before the 'for'
    // instruction.
    fastMemRef = prologue.create<AllocOp>(loc, fastMemRefType)->getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    *sizeInBytes = getMemRefSizeInBytes(fastMemRefType).getValue();
    LLVM_DEBUG(std::string ss; llvm::raw_string_ostream oss(ss);
               oss << "Creating DMA buffer of type " << fastMemRefType;
               oss << " and size " << llvm::divideCeil(*sizeInBytes, 1024)
                   << " KiB\n";
               emitNoteForBlock(*block, oss.str()));
  } else {
    // Reuse the one already created.
    fastMemRef = fastBufferMap[memref];
    *sizeInBytes = 0;
  }
  // Create a tag (single element 1-d memref) for the DMA.
  auto tagMemRefType = top.getMemRefType({1}, top.getIntegerType(32));
  auto tagMemRef = prologue.create<AllocOp>(loc, tagMemRefType);

  auto numElementsSSA =
      top.create<ConstantIndexOp>(loc, numElements.getValue());

  SmallVector<StrideInfo, 4> strideInfos;
  getMultiLevelStrides(region, fastBufferShape, &strideInfos);

  // TODO(bondhugula): use all stride levels once DmaStartOp is extended for
  // multi-level strides.
  if (strideInfos.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "Only up to one level of stride supported\n");
    return false;
  }

  Value *stride = nullptr;
  Value *numEltPerStride = nullptr;
  if (!strideInfos.empty()) {
    stride = top.create<ConstantIndexOp>(loc, strideInfos[0].stride);
    numEltPerStride =
        top.create<ConstantIndexOp>(loc, strideInfos[0].numEltPerStride);
  }

  // Record the last instruction just before the point where we insert the
  // outgoing DMAs. We later do the memref replacement later only in [begin,
  // postDomFilter] so that the original memref's in the DMA ops themselves
  // don't get replaced.
  auto postDomFilter = std::prev(end);

  if (!region.isWrite()) {
    // DMA non-blocking read from original buffer to fast buffer.
    b->create<DmaStartOp>(loc, memref, memIndices, fastMemRef, bufIndices,
                          numElementsSSA, tagMemRef, zeroIndex, stride,
                          numEltPerStride);
  } else {
    // DMA non-blocking write from fast buffer to the original memref.
    auto op = b->create<DmaStartOp>(loc, fastMemRef, bufIndices, memref,
                                    memIndices, numElementsSSA, tagMemRef,
                                    zeroIndex, stride, numEltPerStride);
    // Since new ops are being appended (for outgoing DMAs), adjust the end to
    // mark end of range of the original.
    *nEnd = Block::iterator(op->getInstruction());
  }

  // Matching DMA wait to block on completion; tag always has a 0 index.
  b->create<DmaWaitOp>(loc, tagMemRef, zeroIndex, numElementsSSA);

  // Generate dealloc for the tag.
  auto tagDeallocOp = epilogue.create<DeallocOp>(loc, tagMemRef);
  if (*nEnd == end)
    // Since new ops are being appended (for outgoing DMAs), adjust the end to
    // mark end of range of the original.
    *nEnd = Block::iterator(tagDeallocOp->getInstruction());

  // Generate dealloc for the DMA buffer.
  if (!existingBuf)
    epilogue.create<DeallocOp>(loc, fastMemRef);

  // Replace all uses of the old memref with the faster one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  // Ex: to replace load %A[%i, %j] with load %Abuf[%i - %iT, %j - %jT],
  // index remap will be (%i, %j) -> (%i - %iT, %j - %jT),
  // i.e., affine.apply (d0, d1, d2, d3) -> (d2-d0, d3-d1) (%iT, %jT, %i, %j),
  // and (%iT, %jT) will be the 'extraOperands' for 'rep all memref uses with'.
  // d2, d3 correspond to the original indices (%i, %j).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    // The starting operands of indexRemap will be regionSymbols (the symbols on
    // which the memref region is parametric); then those corresponding to
    // the memref's original indices follow.
    auto dimExpr = b->getAffineDimExpr(regionSymbols.size() + i);
    remapExprs.push_back(dimExpr - offsets[i]);
  }
  auto indexRemap =
      b->getAffineMap(regionSymbols.size() + rank, 0, remapExprs, {});

  // Record the begin since it may be invalidated by memref replacement.
  Block::iterator prev;
  bool wasAtStartOfBlock = (begin == block->begin());
  if (!wasAtStartOfBlock)
    prev = std::prev(begin);

  // *Only* those uses within the range [begin, end) of 'block' are replaced.
  replaceAllMemRefUsesWith(memref, fastMemRef,
                           /*extraIndices=*/{}, indexRemap,
                           /*extraOperands=*/regionSymbols,
                           /*domInstFilter=*/&*begin,
                           /*postDomInstFilter=*/&*postDomFilter);

  *nBegin = wasAtStartOfBlock ? block->begin() : std::next(prev);

  return true;
}

/// Generate DMAs for this block. The block is partitioned into separate
/// `regions`; each region is either a sequence of one or more instructions
/// starting and ending with a load or store op, or just a loop (which could
/// have other loops nested within). Returns false on an error, true otherwise.
bool DmaGeneration::runOnBlock(Block *block) {
  if (block->empty())
    return true;

  // Every loop in the block starts and ends a region. A contiguous sequence of
  // operation instructions starting and ending with a load/store op is also
  // identified as a region. Straightline code (contiguous chunks of operation
  // instructions) are always assumed to not exhaust memory. As a result, this
  // approach is conservative in some cases at the moment, we do a check later
  // and report an error with location info.
  // TODO(bondhugula): An 'if' instruction is being treated similar to an
  // operation instruction. 'if''s could have 'for's in them;
  // treat them separately.

  // Get to the first load, store, or for op.
  auto curBegin =
      std::find_if(block->begin(), block->end(), [&](const Instruction &inst) {
        return inst.isa<LoadOp>() || inst.isa<StoreOp>() ||
               inst.isa<AffineForOp>();
      });

  for (auto it = curBegin; it != block->end(); ++it) {
    if (auto forOp = it->dyn_cast<AffineForOp>()) {
      // Returns true if the footprint is known to exceed capacity.
      auto exceedsCapacity = [&](OpPointer<AffineForOp> forOp) {
        Optional<int64_t> footprint =
            getMemoryFootprintBytes(forOp,
                                    /*memorySpace=*/0);
        return (footprint.hasValue() &&
                static_cast<uint64_t>(footprint.getValue()) >
                    fastMemCapacityBytes);
      };

      // If the memory footprint of the 'for' loop is higher than fast
      // memory capacity (when provided), we recurse to DMA at an inner level
      // until we find a depth at which footprint fits in fast mem capacity. If
      // the footprint can't be calculated, we assume for now it fits. Recurse
      // inside if footprint for 'forOp' exceeds capacity, or when
      // clSkipNonUnitStrideLoop is set and the step size is not one.
      bool recurseInner = clSkipNonUnitStrideLoop ? forOp->getStep() != 1
                                                  : exceedsCapacity(forOp);
      if (recurseInner) {
        // We'll recurse and do the DMAs at an inner level for 'forInst'.
        runOnBlock(/*begin=*/curBegin, /*end=*/it);
        // Recurse onto the body of this loop.
        runOnBlock(forOp->getBody());
        // The next region starts right after the 'for' instruction.
        curBegin = std::next(it);
      } else {
        // We have enough capacity, i.e., DMAs will be computed for the portion
        // of the block until 'it', and for 'it', which is 'forOp'. Note that
        // for the latter, the DMAs are placed just before this loop (for
        // incoming DMAs) and right after (for outgoing ones).
        runOnBlock(/*begin=*/curBegin, /*end=*/it);

        // Inner loop DMAs have their own scope - we don't thus update consumed
        // capacity. The footprint check above guarantees this inner loop's
        // footprint fits.
        runOnBlock(/*begin=*/it, /*end=*/std::next(it));
        curBegin = std::next(it);
      }
    } else if (!it->isa<LoadOp>() && !it->isa<StoreOp>()) {
      runOnBlock(/*begin=*/curBegin, /*end=*/it);
      curBegin = std::next(it);
    }
  }

  // Generate the DMA for the final region.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->isKnownTerminator() && "can't be a terminator");
    runOnBlock(/*begin=*/curBegin, /*end=*/block->end());
  }

  return true;
}

/// Given a memref region, determine the lowest depth at which transfers can be
/// placed for it, and return the corresponding block, start and end positions
/// in the block for placing incoming (read) and outgoing (write) DMAs
/// respectively. The lowest depth depends on whether the region being accessed
/// is invariant with respect to one or more immediately surrounding loops.
static void findHighestBlockForPlacement(
    const MemRefRegion &region, const Block &block,
    const Block::iterator &begin, const Block::iterator &end,
    Block **dmaPlacementBlock, Block::iterator *dmaPlacementReadStart,
    Block::iterator *dmaPlacementWriteStart) {
  const auto *cst = region.getConstraints();
  SmallVector<Value *, 4> symbols;
  cst->getIdValues(cst->getNumDimIds(), cst->getNumDimAndSymbolIds(), &symbols);

  SmallVector<OpPointer<AffineForOp>, 4> enclosingFors;
  getLoopIVs(*block.begin(), &enclosingFors);
  // Walk up loop parents till we find an IV on which this region is
  // symbolic/variant.
  auto it = enclosingFors.rbegin();
  for (auto e = enclosingFors.rend(); it != e; ++it) {
    // TODO(bondhugula): also need to be checking this for regions symbols that
    // aren't loop IVs, whether we are within their resp. defs' dominance scope.
    if (llvm::is_contained(symbols, (*it)->getInductionVar()))
      break;
  }

  if (it != enclosingFors.rbegin()) {
    auto lastInvariantIV = *std::prev(it);
    *dmaPlacementReadStart = Block::iterator(lastInvariantIV->getInstruction());
    *dmaPlacementWriteStart = std::next(*dmaPlacementReadStart);
    *dmaPlacementBlock = lastInvariantIV->getInstruction()->getBlock();
  } else {
    *dmaPlacementReadStart = *const_cast<Block::iterator *>(&begin);
    *dmaPlacementWriteStart = *const_cast<Block::iterator *>(&end);
    *dmaPlacementBlock = const_cast<Block *>(&block);
  }
}

/// Generates DMAs for a contiguous sequence of instructions in `block` in the
/// iterator range [begin, end). Returns the total size of the DMA buffers used.
//  Since we generate alloc's and dealloc's for all DMA buffers (before and
//  after the range of instructions resp), all of the fast memory capacity is
//  assumed to be available.
uint64_t DmaGeneration::runOnBlock(Block::iterator begin, Block::iterator end) {
  if (begin == end)
    return 0;

  assert(begin->getBlock() == std::prev(end)->getBlock() &&
         "Inconsistent args");

  Block *block = begin->getBlock();

  // DMAs will be generated for this depth, i.e., symbolic in all loops
  // surrounding the region of this block.
  unsigned dmaDepth = getNestingDepth(*begin);

  LLVM_DEBUG(llvm::dbgs() << "Generating DMAs at depth " << dmaDepth << "\n");

  readRegions.clear();
  writeRegions.clear();
  fastBufferMap.clear();

  // To check for errors when walking the block.
  bool error = false;

  // Walk this range of instructions  to gather all memory regions.
  block->walk(begin, end, [&](Instruction *opInst) {
    // Gather regions to allocate to buffers in faster memory space.
    if (auto loadOp = opInst->dyn_cast<LoadOp>()) {
      if (loadOp->getMemRefType().getMemorySpace() != slowMemorySpace)
        return;
    } else if (auto storeOp = opInst->dyn_cast<StoreOp>()) {
      if (storeOp->getMemRefType().getMemorySpace() != slowMemorySpace)
        return;
    } else {
      // Neither load nor a store op.
      return;
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (!region->compute(opInst, dmaDepth)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Error obtaining memory region: semi-affine maps?\n");
      LLVM_DEBUG(llvm::dbgs() << "over-approximating to the entire memref\n");
      if (!getFullMemRefAsRegion(opInst, dmaDepth, region.get())) {
        LLVM_DEBUG(
            opInst->emitError("Non-constant memref sizes not yet supported"));
        error = true;
        return;
      }
    }

    // Each memref has a single buffer associated with it irrespective of how
    // many load's and store's happen on it.
    // TODO(bondhugula): in the future, when regions don't intersect and satisfy
    // other properties (based on load/store regions), we could consider
    // multiple buffers per memref.

    // Add to the appropriate region if it's not already in it, or take a
    // bounding box union with the existing one if it's already in there.
    // Note that a memref may have both read and write regions - so update the
    // region in the other list if one exists (write in case of read and vice
    // versa) since there is a single bounding box for a memref across all reads
    // and writes that happen on it.

    // Attempts to update; returns true if 'region' exists in targetRegions.
    auto updateRegion =
        [&](const SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4>
                &targetRegions) {
          auto it = targetRegions.find(region->memref);
          if (it == targetRegions.end())
            return false;

          // Perform a union with the existing region.
          if (!it->second->unionBoundingBox(*region)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Memory region bounding box failed; "
                          "over-approximating to the entire memref\n");
            // If the union fails, we will overapproximate.
            if (!getFullMemRefAsRegion(opInst, dmaDepth, region.get())) {
              LLVM_DEBUG(opInst->emitError(
                  "Non-constant memref sizes not yet supported"));
              error = true;
              return true;
            }
            it->second->getConstraints()->clearAndCopyFrom(
                *region->getConstraints());
          }
          return true;
        };

    bool existsInRead = updateRegion(readRegions);
    if (error)
      return;
    bool existsInWrite = updateRegion(writeRegions);
    if (error)
      return;

    // Finally add it to the region list.
    if (region->isWrite() && !existsInWrite) {
      writeRegions[region->memref] = std::move(region);
    } else if (!region->isWrite() && !existsInRead) {
      readRegions[region->memref] = std::move(region);
    }
  });

  if (error) {
    begin->emitError(
        "DMA generation failed for one or more memref's in this block\n");
    return 0;
  }

  uint64_t totalDmaBuffersSizeInBytes = 0;
  bool ret = true;
  auto processRegions =
      [&](const SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4>
              &regions) {
        for (const auto &regionEntry : regions) {
          // For each region, hoist DMA transfer past all invariant 'for's.
          Block::iterator dmaPlacementReadStart, dmaPlacementWriteStart;
          Block *dmaPlacementBlock;
          findHighestBlockForPlacement(
              *regionEntry.second, *block, begin, end, &dmaPlacementBlock,
              &dmaPlacementReadStart, &dmaPlacementWriteStart);

          uint64_t sizeInBytes;
          Block::iterator nBegin, nEnd;
          bool iRet = generateDma(*regionEntry.second, dmaPlacementBlock,
                                  dmaPlacementReadStart, dmaPlacementWriteStart,
                                  &sizeInBytes, &nBegin, &nEnd);
          if (iRet) {
            // dmaPlacmentStart/End (or begin/end) may be invalidated; use
            // nBegin, nEnd to reset.
            if (dmaPlacementBlock == block) {
              begin = nBegin;
              end = nEnd;
            }
            totalDmaBuffersSizeInBytes += sizeInBytes;
          }
          ret = ret & iRet;
        }
      };
  processRegions(readRegions);
  processRegions(writeRegions);

  if (!ret) {
    begin->emitError(
        "DMA generation failed for one or more memref's in this block\n");
    return totalDmaBuffersSizeInBytes;
  }

  // For a range of operation instructions, a note will be emitted at the
  // caller.
  OpPointer<AffineForOp> forOp;
  uint64_t sizeInKib = llvm::divideCeil(totalDmaBuffersSizeInBytes, 1024);
  if (llvm::DebugFlag && (forOp = begin->dyn_cast<AffineForOp>())) {
    forOp->emitNote(
        Twine(sizeInKib) +
        " KiB of DMA buffers in fast memory space for this block\n");
  }

  if (totalDmaBuffersSizeInBytes > fastMemCapacityBytes) {
    StringRef str = "Total size of all DMA buffers' for this block "
                    "exceeds fast memory capacity\n";
    if (auto *inst = block->getContainingInst())
      inst->emitError(str);
    else
      block->getFunction()->emitError(str);
  }

  return totalDmaBuffersSizeInBytes;
}

void DmaGeneration::runOnFunction() {
  Function *f = &getFunction();
  FuncBuilder topBuilder(f);
  zeroIndex = topBuilder.create<ConstantIndexOp>(f->getLoc(), 0);

  for (auto &block : *f)
    runOnBlock(&block);
}

static PassRegistration<DmaGeneration>
    pass("dma-generate", "Generate DMAs for memory operations");
