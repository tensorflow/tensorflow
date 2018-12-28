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

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "dma-generate"

using namespace mlir;

static llvm::cl::opt<unsigned> clFastMemorySpace(
    "dma-fast-memory-space", llvm::cl::Hidden,
    llvm::cl::desc("Set fast memory space id for DMA generation"));

namespace {

/// Generates DMAs for memref's living in 'slowMemorySpace' into newly created
/// buffers in 'fastMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
struct DmaGeneration : public FunctionPass, StmtWalker<DmaGeneration> {
  explicit DmaGeneration(unsigned slowMemorySpace = 0,
                         unsigned fastMemorySpaceArg = 1,
                         int minDmaTransferSize = 1024)
      : FunctionPass(&DmaGeneration::passID), slowMemorySpace(slowMemorySpace),
        minDmaTransferSize(minDmaTransferSize) {
    if (clFastMemorySpace.getNumOccurrences() > 0) {
      fastMemorySpace = clFastMemorySpace;
    } else {
      fastMemorySpace = fastMemorySpaceArg;
    }
  }

  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(Function *f) override { return success(); }
  PassResult runOnMLFunction(Function *f) override;
  void runOnForStmt(ForStmt *forStmt);

  void visitOperationInst(OperationInst *opStmt);
  bool generateDma(const MemRefRegion &region, ForStmt *forStmt,
                   uint64_t *sizeInBytes);

  // List of memory regions to DMA for.
  std::vector<std::unique_ptr<MemRefRegion>> regions;

  // Map from original memref's to the DMA buffers that their accesses are
  // replaced with.
  DenseMap<Value *, Value *> fastBufferMap;

  // Slow memory space associated with DMAs.
  const unsigned slowMemorySpace;
  // Fast memory space associated with DMAs.
  unsigned fastMemorySpace;
  // Minimum DMA transfer size supported by the target in bytes.
  const int minDmaTransferSize;

  // The loop level at which DMAs should be generated. '0' is an outermost loop.
  unsigned dmaDepth;

  static char passID;
};

} // end anonymous namespace

char DmaGeneration::passID = 0;

/// Generates DMAs for memref's living in 'slowMemorySpace' into newly created
/// buffers in 'fastMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
FunctionPass *mlir::createDmaGenerationPass(unsigned slowMemorySpace,
                                            unsigned fastMemorySpace,
                                            int minDmaTransferSize) {
  return new DmaGeneration(slowMemorySpace, fastMemorySpace,
                           minDmaTransferSize);
}

// Gather regions to promote to buffers in faster memory space.
// TODO(bondhugula): handle store op's; only load's handled for now.
void DmaGeneration::visitOperationInst(OperationInst *opStmt) {
  if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
    if (loadOp->getMemRefType().getMemorySpace() != slowMemorySpace)
      return;
  } else if (auto storeOp = opStmt->dyn_cast<StoreOp>()) {
    if (storeOp->getMemRefType().getMemorySpace() != slowMemorySpace)
      return;
  } else {
    // Neither load nor a store op.
    return;
  }

  // TODO(bondhugula): eventually, we need to be performing a union across all
  // regions for a given memref instead of creating one region per memory op.
  // This way we would be allocating O(num of memref's) sets instead of
  // O(num of load/store op's).
  auto region = std::make_unique<MemRefRegion>();
  if (!getMemRefRegion(opStmt, dmaDepth, region.get())) {
    LLVM_DEBUG(llvm::dbgs() << "Error obtaining memory region\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Memory region:\n");
  LLVM_DEBUG(region->getConstraints()->dump());

  regions.push_back(std::move(region));
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
                                 ArrayRef<int> bufferShape,
                                 SmallVectorImpl<StrideInfo> *strideInfos) {
  if (bufferShape.size() <= 1)
    return;

  int64_t numEltPerStride = 1;
  int64_t stride = 1;
  for (int d = bufferShape.size() - 1; d >= 1; d--) {
    int dimSize = region.memref->getType().cast<MemRefType>().getDimSize(d);
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

// Creates a buffer in the faster memory space for the specified region;
// generates a DMA from the lower memory space to this one, and replaces all
// loads to load from that buffer. Returns true if DMAs are generated.
bool DmaGeneration::generateDma(const MemRefRegion &region, ForStmt *forStmt,
                                uint64_t *sizeInBytes) {
  // DMAs for read regions are going to be inserted just before the for loop.
  FuncBuilder prologue(forStmt);
  // DMAs for write regions are going to be inserted just after the for loop.
  FuncBuilder epilogue(forStmt->getBlock(),
                       std::next(Block::iterator(forStmt)));
  FuncBuilder *b = region.isWrite() ? &epilogue : &prologue;

  // Builder to create constants at the top level.
  FuncBuilder top(forStmt->getFunction());

  auto loc = forStmt->getLoc();
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

  Value *zeroIndex = top.create<ConstantIndexOp>(loc, 0);

  unsigned rank = memRefType.getRank();
  SmallVector<int, 4> fastBufferShape;

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  lbs.reserve(rank);
  Optional<int64_t> numElements =
      region.getBoundingConstantSizeAndShape(&fastBufferShape, &lbs);
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
    *sizeInBytes = 0;
    return false;
  }

  if (numElements.getValue() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to DMA\n");
    *sizeInBytes = 0;
    return false;
  }

  // 'outerIVs' holds the values that this memory region is symbolic/paramteric
  // on; this would correspond to loop IVs surrounding the level at which the
  // DMA generation is being done.
  const FlatAffineConstraints *cst = region.getConstraints();
  auto ids = cst->getIds();
  SmallVector<Value *, 8> outerIVs;
  for (unsigned i = rank, e = ids.size(); i < e; i++) {
    auto id = cst->getIds()[i];
    assert(id.hasValue() && "Value id expected");
    outerIVs.push_back(id.getValue());
  }

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
    offset = offset + lbs[d][cst->getNumCols() - 1 - rank];

    // Set DMA start location for this dimension in the lower memory space
    // memref.
    if (auto caf = offset.dyn_cast<AffineConstantExpr>()) {
      memIndices.push_back(
          top.create<ConstantIndexOp>(loc, caf.getValue())->getResult());
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      auto map = top.getAffineMap(
          cst->getNumDimIds() + cst->getNumSymbolIds() - rank, 0, offset, {});
      memIndices.push_back(
          b->create<AffineApplyOp>(loc, map, outerIVs)->getResult(0));
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
  // TODO(bondhugula): union across all memory op's per buffer. For now assuming
  // that multiple memory op's on the same memref have the *same* memory
  // footprint.
  if (fastBufferMap.find(memref) == fastBufferMap.end()) {
    auto fastMemRefType = top.getMemRefType(
        fastBufferShape, memRefType.getElementType(), {}, fastMemorySpace);

    LLVM_DEBUG(llvm::dbgs() << "Creating a new buffer of type: ");
    LLVM_DEBUG(fastMemRefType.dump(); llvm::dbgs() << "\n");

    // Create the fast memory space buffer just before the 'for' statement.
    fastMemRef = prologue.create<AllocOp>(loc, fastMemRefType)->getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    *sizeInBytes = getMemRefSizeInBytes(fastMemRefType).getValue();
    LLVM_DEBUG(llvm::dbgs() << "Creating a new buffer of type ";
               fastMemRefType.dump();
               llvm::dbgs()
               << " and size " << Twine(llvm::divideCeil(*sizeInBytes, 1024))
               << " KiB\n";);

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

  // TODO(bondhugula): check for transfer sizes not being a multiple of
  // minDmaTransferSize and handle them appropriately.

  SmallVector<StrideInfo, 4> strideInfos;
  getMultiLevelStrides(region, fastBufferShape, &strideInfos);

  // TODO(bondhugula): use all stride level once DmaStartOp is extended for
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

  if (!region.isWrite()) {
    // DMA non-blocking read from original buffer to fast buffer.
    b->create<DmaStartOp>(loc, memref, memIndices, fastMemRef, bufIndices,
                          numElementsSSA, tagMemRef, zeroIndex, stride,
                          numEltPerStride);
  } else {
    // DMA non-blocking write from fast buffer to the original memref.
    b->create<DmaStartOp>(loc, fastMemRef, bufIndices, memref, memIndices,
                          numElementsSSA, tagMemRef, zeroIndex, stride,
                          numEltPerStride);
  }

  // Matching DMA wait to block on completion; tag always has a 0 index.
  b->create<DmaWaitOp>(loc, tagMemRef, zeroIndex, numElementsSSA);

  // Replace all uses of the old memref with the faster one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  // Ex: to replace load %A[%i, %j] with load %Abuf[%i - %iT, %j - %jT],
  // index remap will be (%i, %j) -> (%i - %iT, %j - %jT),
  // i.e., affine_apply (d0, d1, d2, d3) -> (d2-d0, d3-d1) (%iT, %jT, %i, %j),
  // and (%iT, %jT) will be the 'extraOperands' for 'rep all memref uses with'.
  // d2, d3 correspond to the original indices (%i, %j).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    // The starting operands of indexRemap will be outerIVs (the loops
    // surrounding the depth at which this DMA is being done); then those
    // corresponding to the memref's original indices follow.
    auto dimExpr = b->getAffineDimExpr(outerIVs.size() + i);
    remapExprs.push_back(dimExpr - offsets[i]);
  }
  auto indexRemap = b->getAffineMap(outerIVs.size() + rank, 0, remapExprs, {});
  // *Only* those uses within the body of 'forStmt' are replaced.
  replaceAllMemRefUsesWith(memref, fastMemRef,
                           /*extraIndices=*/{}, indexRemap,
                           /*extraOperands=*/outerIVs,
                           /*domStmtFilter=*/&*forStmt->getBody()->begin());
  return true;
}

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
// TODO(bondhugula): move this to utilities later.
static unsigned getNestingDepth(const Statement &stmt) {
  const Statement *currStmt = &stmt;
  unsigned depth = 0;
  while ((currStmt = currStmt->getParentStmt())) {
    if (isa<ForStmt>(currStmt))
      depth++;
  }
  return depth;
}

// TODO(bondhugula): make this run on a Block instead of a 'for' stmt.
void DmaGeneration::runOnForStmt(ForStmt *forStmt) {
  // For now (for testing purposes), we'll run this on the outermost among 'for'
  // stmt's with unit stride, i.e., right at the top of the tile if tiling has
  // been done. In the future, the DMA generation has to be done at a level
  // where the generated data fits in a higher level of the memory hierarchy; so
  // the pass has to be instantiated with additional information that we aren't
  // provided with at the moment.
  if (forStmt->getStep() != 1) {
    if (auto *innerFor = dyn_cast<ForStmt>(&*forStmt->getBody()->begin())) {
      runOnForStmt(innerFor);
    }
    return;
  }

  // DMAs will be generated for this depth, i.e., for all data accessed by this
  // loop.
  dmaDepth = getNestingDepth(*forStmt);

  regions.clear();
  fastBufferMap.clear();

  // Walk this 'for' statement to gather all memory regions.
  walk(forStmt);

  uint64_t totalSizeInBytes = 0;

  bool ret = false;
  for (const auto &region : regions) {
    uint64_t sizeInBytes;
    bool iRet = generateDma(*region, forStmt, &sizeInBytes);
    if (iRet)
      totalSizeInBytes += sizeInBytes;
    ret = ret | iRet;
  }
  if (!ret) {
    LLVM_DEBUG(llvm::dbgs()
                   << "DMA generation failed for one or more memref's\n";);
  }
  LLVM_DEBUG(llvm::dbgs() << Twine(llvm::divideCeil(totalSizeInBytes, 1024))
                          << " KiB of DMA buffers in fast memory space\n";);
}

PassResult DmaGeneration::runOnMLFunction(Function *f) {
  for (auto &stmt : *f->getBody()) {
    if (auto *forStmt = dyn_cast<ForStmt>(&stmt)) {
      runOnForStmt(forStmt);
    }
  }
  // This function never leaves the IR in an invalid state.
  return success();
}

static PassRegistration<DmaGeneration>
    pass("dma-generate", "Generate DMAs for memory operations");
