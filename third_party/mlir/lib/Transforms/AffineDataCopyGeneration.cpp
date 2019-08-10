//===- AffineDataCopyGeneration.cpp - Explicit memref copying pass ------*-===//
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
// necessary data movement operations performed through either regular
// point-wise load/store's or DMAs. Such explicit copying (also referred to as
// array packing/unpacking in the literature), when done on arrays that exhibit
// reuse, results in near elimination of conflict misses, TLB misses, reduced
// use of hardware prefetch streams, and reduced false sharing. It is also
// necessary for hardware that explicitly managed levels in the memory
// hierarchy, and where DMAs may have to be used. This optimization is often
// performed on already tiled code.
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

#define DEBUG_TYPE "affine-data-copy-generate"

using namespace mlir;
using llvm::SmallMapVector;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<unsigned long long> clFastMemoryCapacity(
    "affine-data-copy-generate-fast-mem-capacity",
    llvm::cl::desc(
        "Set fast memory space capacity in KiB (default: unlimited)"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool>
    clDma("affine-data-copy-generate-dma",
          llvm::cl::desc("Generate DMA instead of point-wise copy"),
          llvm::cl::cat(clOptionsCategory),
          llvm::cl::init(true));

static llvm::cl::opt<unsigned> clFastMemorySpace(
    "affine-data-copy-generate-fast-mem-space", llvm::cl::init(0),
    llvm::cl::desc(
        "Fast memory space identifier for copy generation (default: 1)"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clSkipNonUnitStrideLoop(
    "affine-data-copy-generate-skip-non-unit-stride-loops", llvm::cl::Hidden,
    llvm::cl::init(false),
    llvm::cl::desc("Testing purposes: avoid non-unit stride loop choice depths "
                   "for copy placement"),
    llvm::cl::cat(clOptionsCategory));

namespace {

/// Replaces all loads and stores on memref's living in 'slowMemorySpace' by
/// introducing copy operations to transfer data into `fastMemorySpace` and
/// rewriting the original load's/store's to instead load/store from the
/// allocated fast memory buffers. Additional options specify the identifier
/// corresponding to the fast memory space and the amount of fast memory space
/// available. The pass traverses through the nesting structure, recursing to
/// inner levels if necessary to determine at what depth copies need to be
/// placed so that the allocated buffers fit within the memory capacity
/// provided.
// TODO(bondhugula): We currently can't generate copies correctly when stores
// are strided. Check for strided stores.
struct AffineDataCopyGeneration
    : public FunctionPass<AffineDataCopyGeneration> {
  explicit AffineDataCopyGeneration(
      unsigned slowMemorySpace = 0,
      unsigned fastMemorySpace = clFastMemorySpace, unsigned tagMemorySpace = 0,
      int minDmaTransferSize = 1024,
      uint64_t fastMemCapacityBytes =
          (clFastMemoryCapacity.getNumOccurrences() > 0
               ? clFastMemoryCapacity * 1024 // cl-provided size is in KiB
               : std::numeric_limits<uint64_t>::max()),
      bool generateDma = clDma,
      bool skipNonUnitStrideLoops = clSkipNonUnitStrideLoop)
      : slowMemorySpace(slowMemorySpace), fastMemorySpace(fastMemorySpace),
        tagMemorySpace(tagMemorySpace), minDmaTransferSize(minDmaTransferSize),
        fastMemCapacityBytes(fastMemCapacityBytes), generateDma(generateDma),
        skipNonUnitStrideLoops(skipNonUnitStrideLoops) {}

  explicit AffineDataCopyGeneration(const AffineDataCopyGeneration &other)
      : slowMemorySpace(other.slowMemorySpace),
        fastMemorySpace(other.fastMemorySpace),
        tagMemorySpace(other.tagMemorySpace),
        minDmaTransferSize(other.minDmaTransferSize),
        fastMemCapacityBytes(other.fastMemCapacityBytes),
        generateDma(other.generateDma),
        skipNonUnitStrideLoops(other.skipNonUnitStrideLoops) {}

  void runOnFunction() override;
  LogicalResult runOnBlock(Block *block);
  uint64_t runOnBlock(Block::iterator begin, Block::iterator end);

  LogicalResult generateCopy(const MemRefRegion &region, Block *block,
                             Block::iterator begin, Block::iterator end,
                             uint64_t *sizeInBytes, Block::iterator *nBegin,
                             Block::iterator *nEnd);

  // List of memory regions to copy for. We need a map vector to have a
  // guaranteed iteration order to write test cases. CHECK-DAG doesn't help here
  // since the alloc's for example are identical except for the SSA id.
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> readRegions;
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> writeRegions;

  // Nests that are copy in's or copy out's; the root AffineForOp of that
  // nest is stored herein.
  DenseSet<Operation *> copyNests;

  // Map from original memref's to the fast buffers that their accesses are
  // replaced with.
  DenseMap<Value *, Value *> fastBufferMap;

  // Slow memory space associated with copies.
  const unsigned slowMemorySpace;
  // Fast memory space associated with copies.
  unsigned fastMemorySpace;
  // Memory space associated with DMA tags.
  unsigned tagMemorySpace;
  // Minimum DMA transfer size supported by the target in bytes.
  const int minDmaTransferSize;
  // Capacity of the faster memory space.
  uint64_t fastMemCapacityBytes;

  // If set, generate DMA operations instead of read/write.
  bool generateDma;

  // If set, ignore loops with steps other than 1.
  bool skipNonUnitStrideLoops;

  // Constant zero index to avoid too many duplicates.
  Value *zeroIndex = nullptr;
};

} // end anonymous namespace

/// Generates copies for memref's living in 'slowMemorySpace' into newly created
/// buffers in 'fastMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
FunctionPassBase *mlir::createAffineDataCopyGenerationPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace, unsigned tagMemorySpace,
    int minDmaTransferSize, uint64_t fastMemCapacityBytes) {
  return new AffineDataCopyGeneration(slowMemorySpace, fastMemorySpace,
                                      tagMemorySpace, minDmaTransferSize,
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
static bool getFullMemRefAsRegion(Operation *opInst, unsigned numParamLoopIVs,
                                  MemRefRegion *region) {
  unsigned rank;
  if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
    rank = loadOp.getMemRefType().getRank();
    region->memref = loadOp.getMemRef();
    region->setWrite(false);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
    rank = storeOp.getMemRefType().getRank();
    region->memref = storeOp.getMemRef();
    region->setWrite(true);
  } else {
    assert(false && "expected load or store op");
    return false;
  }
  auto memRefType = region->memref->getType().cast<MemRefType>();
  if (!memRefType.hasStaticShape())
    return false;

  auto *regionCst = region->getConstraints();

  // Just get the first numSymbols IVs, which the memref region is parametric
  // on.
  SmallVector<AffineForOp, 4> ivs;
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

static InFlightDiagnostic LLVM_ATTRIBUTE_UNUSED
emitRemarkForBlock(Block &block) {
  return block.getParentOp()->emitRemark();
}

/// Generates a point-wise copy from/to `memref' to/from `fastMemRef' and
/// returns the outermost AffineForOp of the copy loop nest. `memIndicesStart'
/// holds the lower coordinates of the region in the original memref to copy
/// in/out. If `copyOut' is true, generates a copy-out; otherwise a copy-in.
static AffineForOp generatePointWiseCopy(Location loc, Value *memref,
                                         Value *fastMemRef,
                                         ArrayRef<Value *> memIndicesStart,
                                         ArrayRef<int64_t> fastBufferShape,
                                         bool isCopyOut, OpBuilder b) {
  assert(!memIndicesStart.empty() && "only 1-d or more memrefs");

  // The copy-in nest is generated as follows as an example for a 2-d region:
  // for x = ...
  //   for y = ...
  //     fast_buf[x][y] = buf[mem_x + x][mem_y + y]

  SmallVector<Value *, 4> fastBufIndices, memIndices;
  AffineForOp copyNestRoot;
  for (unsigned d = 0, e = fastBufferShape.size(); d < e; ++d) {
    auto forOp = b.create<AffineForOp>(loc, 0, fastBufferShape[d]);
    if (d == 0)
      copyNestRoot = forOp;
    b = forOp.getBodyBuilder();
    fastBufIndices.push_back(forOp.getInductionVar());
    // Construct the subscript for the slow memref being copied.
    SmallVector<Value *, 2> operands = {memIndicesStart[d], forOp.getInductionVar()};
    auto memIndex = b.create<AffineApplyOp>(
        loc,
        b.getAffineMap(2, 0, b.getAffineDimExpr(0) + b.getAffineDimExpr(1)),
        operands);
    memIndices.push_back(memIndex);
  }

  if (!isCopyOut) {
    // Copy in.
    auto load = b.create<AffineLoadOp>(loc, memref, memIndices);
    b.create<AffineStoreOp>(loc, load, fastMemRef, fastBufIndices);
    return copyNestRoot;
  }

  // Copy out.
  auto load = b.create<AffineLoadOp>(loc, fastMemRef, fastBufIndices);
  b.create<AffineStoreOp>(loc, load, memref, memIndices);
  return copyNestRoot;
}

/// Creates a buffer in the faster memory space for the specified region;
/// generates a copy from the lower memory space to this one, and replaces all
/// loads to load from that buffer. Returns failure if copies could not be
/// generated due to yet unimplemented cases. `begin` and `end` specify the
/// insertion points where the incoming copies and outgoing copies,
/// respectively, should be inserted (the insertion happens right before the
/// insertion point). Since `begin` can itself be invalidated due to the memref
/// rewriting done from this method, the output argument `nBegin` is set to its
/// replacement (set to `begin` if no invalidation happens). Since outgoing
/// copies are inserted at `end`, the output argument `nEnd` is set to the one
/// following the original end (since the latter could have been
/// invalidated/replaced). `sizeInBytes` is set to the size of the fast buffer
/// allocated.
LogicalResult AffineDataCopyGeneration::generateCopy(
    const MemRefRegion &region, Block *block, Block::iterator begin,
    Block::iterator end, uint64_t *sizeInBytes, Block::iterator *nBegin,
    Block::iterator *nEnd) {
  *nBegin = begin;
  *nEnd = end;

  if (begin == end)
    return success();

  // Copies for read regions are going to be inserted at 'begin'.
  OpBuilder prologue(block, begin);
  // Copies for write regions are going to be inserted at 'end'.
  OpBuilder epilogue(block, end);
  OpBuilder &b = region.isWrite() ? epilogue : prologue;

  // Builder to create constants at the top level.
  auto func = block->getParent()->getParentOfType<FuncOp>();
  OpBuilder top(func.getBody());

  auto loc = region.loc;
  auto *memref = region.memref;
  auto memRefType = memref->getType().cast<MemRefType>();

  auto layoutMaps = memRefType.getAffineMaps();
  if (layoutMaps.size() > 1 ||
      (layoutMaps.size() == 1 && !layoutMaps[0].isIdentity())) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return failure();
  }

  // Indices to use for the copying.
  // Indices for the original memref being copied from/to.
  SmallVector<Value *, 4> memIndices;
  // Indices for the faster buffer being copied into/from.
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
    return failure();
  }

  if (numElements.getValue() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to copy\n");
    *sizeInBytes = 0;
    return success();
  }

  const FlatAffineConstraints *cst = region.getConstraints();
  // 'regionSymbols' hold values that this memory region is symbolic/paramteric
  // on; these typically include loop IVs surrounding the level at which the
  // copy generation is being done or other valid symbols in MLIR.
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

    // Set copy start location for this dimension in the lower memory space
    // memref.
    if (auto caf = offset.dyn_cast<AffineConstantExpr>()) {
      auto indexVal = caf.getValue();
      if (indexVal == 0) {
        memIndices.push_back(zeroIndex);
      } else {
        memIndices.push_back(
            top.create<ConstantIndexOp>(loc, indexVal).getResult());
      }
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      auto map = top.getAffineMap(
          cst->getNumDimIds() + cst->getNumSymbolIds() - rank, 0, offset);
      memIndices.push_back(b.create<AffineApplyOp>(loc, map, regionSymbols));
    }
    // The fast buffer is copied into at location zero; addressing is relative.
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

    // Create the fast memory space buffer just before the 'affine.for'
    // operation.
    fastMemRef = prologue.create<AllocOp>(loc, fastMemRefType).getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    *sizeInBytes = getMemRefSizeInBytes(fastMemRefType).getValue();
    LLVM_DEBUG(emitRemarkForBlock(*block)
               << "Creating fast buffer of type " << fastMemRefType
               << " and size " << llvm::divideCeil(*sizeInBytes, 1024)
               << " KiB\n");
  } else {
    // Reuse the one already created.
    fastMemRef = fastBufferMap[memref];
    *sizeInBytes = 0;
  }

  auto numElementsSSA =
      top.create<ConstantIndexOp>(loc, numElements.getValue());

  SmallVector<StrideInfo, 4> strideInfos;
  getMultiLevelStrides(region, fastBufferShape, &strideInfos);

  // TODO(bondhugula): use all stride levels once DmaStartOp is extended for
  // multi-level strides.
  if (strideInfos.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "Only up to one level of stride supported\n");
    return failure();
  }

  Value *stride = nullptr;
  Value *numEltPerStride = nullptr;
  if (!strideInfos.empty()) {
    stride = top.create<ConstantIndexOp>(loc, strideInfos[0].stride);
    numEltPerStride =
        top.create<ConstantIndexOp>(loc, strideInfos[0].numEltPerStride);
  }

  // Record the last operation just before the point where we insert the
  // copy out's. We later do the memref replacement later only in [begin,
  // postDomFilter] so that the original memref's in the data movement code
  // themselves don't get replaced.
  auto postDomFilter = std::prev(end);

  // Create fully composed affine maps for each memref.
  auto memAffineMap = b.getMultiDimIdentityMap(memIndices.size());
  fullyComposeAffineMapAndOperands(&memAffineMap, &memIndices);
  auto bufAffineMap = b.getMultiDimIdentityMap(bufIndices.size());
  fullyComposeAffineMapAndOperands(&bufAffineMap, &bufIndices);

  if (!generateDma) {
    auto copyNest = generatePointWiseCopy(loc, memref, fastMemRef, memIndices,
                                          fastBufferShape,
                                          /*isCopyOut=*/region.isWrite(), b);

    // Record this so that we can skip it from yet another copy.
    copyNests.insert(copyNest);

    if (region.isWrite())
      // Since new ops are being appended (for copy out's), adjust the end to
      // mark end of block range being processed.
      *nEnd = Block::iterator(copyNest.getOperation());
  } else {
    // Create a tag (single element 1-d memref) for the DMA.
    auto tagMemRefType =
        top.getMemRefType({1}, top.getIntegerType(32), {}, tagMemorySpace);
    auto tagMemRef = prologue.create<AllocOp>(loc, tagMemRefType);

    SmallVector<Value *, 4> tagIndices({zeroIndex});
    auto tagAffineMap = b.getMultiDimIdentityMap(tagIndices.size());
    fullyComposeAffineMapAndOperands(&tagAffineMap, &tagIndices);
    if (!region.isWrite()) {
      // DMA non-blocking read from original buffer to fast buffer.
      b.create<AffineDmaStartOp>(loc, memref, memAffineMap, memIndices,
                                 fastMemRef, bufAffineMap, bufIndices,
                                 tagMemRef, tagAffineMap, tagIndices,
                                 numElementsSSA, stride, numEltPerStride);
    } else {
      // DMA non-blocking write from fast buffer to the original memref.
      auto op = b.create<AffineDmaStartOp>(
          loc, fastMemRef, bufAffineMap, bufIndices, memref, memAffineMap,
          memIndices, tagMemRef, tagAffineMap, tagIndices, numElementsSSA,
          stride, numEltPerStride);
      // Since new ops are being appended (for outgoing DMAs), adjust the end to
      // mark end of block range being processed.
      *nEnd = Block::iterator(op.getOperation());
    }

    // Matching DMA wait to block on completion; tag always has a 0 index.
    b.create<AffineDmaWaitOp>(loc, tagMemRef, tagAffineMap, zeroIndex,
                              numElementsSSA);

    // Generate dealloc for the tag.
    auto tagDeallocOp = epilogue.create<DeallocOp>(loc, tagMemRef);
    if (*nEnd == end)
      // Since new ops are being appended (for outgoing DMAs), adjust the end to
      // mark end of range of the original.
      *nEnd = Block::iterator(tagDeallocOp.getOperation());
  }

  // Generate dealloc for the buffer.
  if (!existingBuf) {
    auto bufDeallocOp = epilogue.create<DeallocOp>(loc, fastMemRef);
    // When generating pointwise copies, `nEnd' has to be set to deallocOp on
    // the fast buffer (since it marks the new end insertion point).
    if (!generateDma && *nEnd == end)
      *nEnd = Block::iterator(bufDeallocOp.getOperation());
  }

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
    auto dimExpr = b.getAffineDimExpr(regionSymbols.size() + i);
    remapExprs.push_back(dimExpr - offsets[i]);
  }
  auto indexRemap = b.getAffineMap(regionSymbols.size() + rank, 0, remapExprs);

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

  return success();
}

/// Generate copies for this block. The block is partitioned into separate
/// ranges: each range is either a sequence of one or more operations starting
/// and ending with an affine load or store op, or just an affine.forop (which
/// could have other affine for op's nested within).
LogicalResult AffineDataCopyGeneration::runOnBlock(Block *block) {
  if (block->empty())
    return success();

  copyNests.clear();

  // Every affine.forop in the block starts and ends a block range for copying.
  // A contiguous sequence of operations starting and ending with a load/store
  // op is also identified as a copy block range. Straightline code (a
  // contiguous chunk of operations excluding AffineForOp's) are always assumed
  // to not exhaust memory. As a result, this approach is conservative in some
  // cases at the moment; we do a check later and report an error with location
  // info.
  // TODO(bondhugula): An 'affine.if' operation is being treated similar to an
  // operation. 'affine.if''s could have 'affine.for's in them;
  // treat them separately.

  // Get to the first load, store, or for op (that is not a copy nest itself).
  auto curBegin =
      std::find_if(block->begin(), block->end(), [&](Operation &op) {
        return (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op) ||
                isa<AffineForOp>(op)) &&
               copyNests.count(&op) == 0;
      });

  for (auto it = curBegin; it != block->end(); ++it) {
    AffineForOp forOp;
    if ((forOp = dyn_cast<AffineForOp>(&*it)) && copyNests.count(forOp) == 0) {
      // Returns true if the footprint is known to exceed capacity.
      auto exceedsCapacity = [&](AffineForOp forOp) {
        Optional<int64_t> footprint =
            getMemoryFootprintBytes(forOp,
                                    /*memorySpace=*/0);
        return (footprint.hasValue() &&
                static_cast<uint64_t>(footprint.getValue()) >
                    fastMemCapacityBytes);
      };

      // If the memory footprint of the 'affine.for' loop is higher than fast
      // memory capacity (when provided), we recurse to copy at an inner level
      // until we find a depth at which footprint fits in fast mem capacity. If
      // the footprint can't be calculated, we assume for now it fits. Recurse
      // inside if footprint for 'forOp' exceeds capacity, or when
      // skipNonUnitStrideLoops is set and the step size is not one.
      bool recurseInner = skipNonUnitStrideLoops ? forOp.getStep() != 1
                                                 : exceedsCapacity(forOp);
      if (recurseInner) {
        // We'll recurse and do the copies at an inner level for 'forInst'.
        runOnBlock(/*begin=*/curBegin, /*end=*/it);
        // Recurse onto the body of this loop.
        runOnBlock(forOp.getBody());
        // The next block range starts right after the 'affine.for' operation.
        curBegin = std::next(it);
      } else {
        // We have enough capacity, i.e., copies will be computed for the
        // portion of the block until 'it', and for 'it', which is 'forOp'. Note
        // that for the latter, the copies are placed just before this loop (for
        // incoming copies) and right after (for outgoing ones).
        runOnBlock(/*begin=*/curBegin, /*end=*/it);

        // Inner loop copies have their own scope - we don't thus update
        // consumed capacity. The footprint check above guarantees this inner
        // loop's footprint fits.
        runOnBlock(/*begin=*/it, /*end=*/std::next(it));
        curBegin = std::next(it);
      }
    } else if (!isa<AffineLoadOp>(&*it) && !isa<AffineStoreOp>(&*it)) {
      runOnBlock(/*begin=*/curBegin, /*end=*/it);
      curBegin = std::next(it);
    }
  }

  // Generate the copy for the final block range.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->isKnownTerminator() && "can't be a terminator");
    runOnBlock(/*begin=*/curBegin, /*end=*/block->end());
  }

  return success();
}

/// Given a memref region, determine the lowest depth at which transfers can be
/// placed for it, and return the corresponding block, start and end positions
/// in the block for placing incoming (read) and outgoing (write) copies
/// respectively. The lowest depth depends on whether the region being accessed
/// is invariant with respect to one or more immediately surrounding loops.
static void
findHighestBlockForPlacement(const MemRefRegion &region, Block &block,
                             Block::iterator &begin, Block::iterator &end,
                             Block **copyPlacementBlock,
                             Block::iterator *copyPlacementReadStart,
                             Block::iterator *copyPlacementWriteStart) {
  const auto *cst = region.getConstraints();
  SmallVector<Value *, 4> symbols;
  cst->getIdValues(cst->getNumDimIds(), cst->getNumDimAndSymbolIds(), &symbols);

  SmallVector<AffineForOp, 4> enclosingFors;
  getLoopIVs(*block.begin(), &enclosingFors);
  // Walk up loop parents till we find an IV on which this region is
  // symbolic/variant.
  auto it = enclosingFors.rbegin();
  for (auto e = enclosingFors.rend(); it != e; ++it) {
    // TODO(bondhugula): also need to be checking this for regions symbols that
    // aren't loop IVs, whether we are within their resp. defs' dominance scope.
    if (llvm::is_contained(symbols, it->getInductionVar()))
      break;
  }

  if (it != enclosingFors.rbegin()) {
    auto lastInvariantIV = *std::prev(it);
    *copyPlacementReadStart = Block::iterator(lastInvariantIV.getOperation());
    *copyPlacementWriteStart = std::next(*copyPlacementReadStart);
    *copyPlacementBlock = lastInvariantIV.getOperation()->getBlock();
  } else {
    *copyPlacementReadStart = begin;
    *copyPlacementWriteStart = end;
    *copyPlacementBlock = &block;
  }
}

/// Generates copies for a contiguous sequence of operations in `block` in the
/// iterator range [begin, end). Returns the total size of the fast buffers
/// used.
//  Since we generate alloc's and dealloc's for all fast buffers (before and
//  after the range of operations resp.), all of the fast memory capacity is
//  assumed to be available for processing this block range.
uint64_t AffineDataCopyGeneration::runOnBlock(Block::iterator begin,
                                              Block::iterator end) {
  if (begin == end)
    return 0;

  assert(begin->getBlock() == std::prev(end)->getBlock() &&
         "Inconsistent args");

  Block *block = begin->getBlock();

  // Copies will be generated for this depth, i.e., symbolic in all loops
  // surrounding the this block range.
  unsigned copyDepth = getNestingDepth(*begin);

  LLVM_DEBUG(llvm::dbgs() << "Generating copies at depth " << copyDepth
                          << "\n");

  readRegions.clear();
  writeRegions.clear();
  fastBufferMap.clear();

  // To check for errors when walking the block.
  bool error = false;

  // Walk this range of operations  to gather all memory regions.
  block->walk(begin, end, [&](Operation *opInst) {
    // Gather regions to allocate to buffers in faster memory space.
    if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
      if (loadOp.getMemRefType().getMemorySpace() != slowMemorySpace)
        return;
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
      if (storeOp.getMemRefType().getMemorySpace() != slowMemorySpace)
        return;
    } else {
      // Neither load nor a store op.
      return;
    }

    // Compute the MemRefRegion accessed.
    auto region = llvm::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(region->compute(opInst, copyDepth))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Error obtaining memory region: semi-affine maps?\n");
      LLVM_DEBUG(llvm::dbgs() << "over-approximating to the entire memref\n");
      if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
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
          if (failed(it->second->unionBoundingBox(*region))) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Memory region bounding box failed; "
                          "over-approximating to the entire memref\n");
            // If the union fails, we will overapproximate.
            if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
              LLVM_DEBUG(opInst->emitError(
                  "Non-constant memref sizes not yet supported"));
              error = true;
              return true;
            }
            it->second->getConstraints()->clearAndCopyFrom(
                *region->getConstraints());
          } else {
            // Union was computed and stored in 'it->second': copy to 'region'.
            region->getConstraints()->clearAndCopyFrom(
                *it->second->getConstraints());
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
        "copy generation failed for one or more memref's in this block\n");
    return 0;
  }

  uint64_t totalCopyBuffersSizeInBytes = 0;
  bool ret = true;
  auto processRegions =
      [&](const SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4>
              &regions) {
        for (const auto &regionEntry : regions) {
          // For each region, hoist copy in/out past all invariant
          // 'affine.for's.
          Block::iterator copyPlacementReadStart, copyPlacementWriteStart;
          Block *copyPlacementBlock;
          findHighestBlockForPlacement(
              *regionEntry.second, *block, begin, end, &copyPlacementBlock,
              &copyPlacementReadStart, &copyPlacementWriteStart);

          uint64_t sizeInBytes;
          Block::iterator nBegin, nEnd;
          LogicalResult iRet = generateCopy(
              *regionEntry.second, copyPlacementBlock, copyPlacementReadStart,
              copyPlacementWriteStart, &sizeInBytes, &nBegin, &nEnd);
          if (succeeded(iRet)) {
            // copyPlacmentStart/End (or begin/end) may be invalidated; use
            // nBegin, nEnd to reset.
            if (copyPlacementBlock == block) {
              begin = nBegin;
              end = nEnd;
            }
            totalCopyBuffersSizeInBytes += sizeInBytes;
          }
          ret = ret & succeeded(iRet);
        }
      };
  processRegions(readRegions);
  processRegions(writeRegions);

  if (!ret) {
    begin->emitError(
        "copy generation failed for one or more memref's in this block\n");
    return totalCopyBuffersSizeInBytes;
  }

  // For a range of operations, a note will be emitted at the caller.
  AffineForOp forOp;
  uint64_t sizeInKib = llvm::divideCeil(totalCopyBuffersSizeInBytes, 1024);
  if (llvm::DebugFlag && (forOp = dyn_cast<AffineForOp>(&*begin))) {
    forOp.emitRemark()
        << sizeInKib
        << " KiB of copy buffers in fast memory space for this block\n";
  }

  if (totalCopyBuffersSizeInBytes > fastMemCapacityBytes) {
    StringRef str = "Total size of all copy buffers' for this block "
                    "exceeds fast memory capacity\n";
    block->getParentOp()->emitError(str);
  }

  return totalCopyBuffersSizeInBytes;
}

void AffineDataCopyGeneration::runOnFunction() {
  FuncOp f = getFunction();
  OpBuilder topBuilder(f.getBody());
  zeroIndex = topBuilder.create<ConstantIndexOp>(f.getLoc(), 0);

  for (auto &block : f)
    runOnBlock(&block);
}

static PassRegistration<AffineDataCopyGeneration>
    pass("affine-data-copy-generate",
         "Generate explicit copying for memory operations");
