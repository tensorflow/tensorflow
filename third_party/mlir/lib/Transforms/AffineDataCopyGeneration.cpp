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

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "affine-data-copy-generate"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<unsigned long long> clFastMemoryCapacity(
    "affine-data-copy-generate-fast-mem-capacity",
    llvm::cl::desc(
        "Set fast memory space capacity in KiB (default: unlimited)"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool>
    clDma("affine-data-copy-generate-dma",
          llvm::cl::desc("Generate DMA instead of point-wise copy"),
          llvm::cl::cat(clOptionsCategory), llvm::cl::init(true));

static llvm::cl::opt<unsigned> clFastMemorySpace(
    "affine-data-copy-generate-fast-mem-space", llvm::cl::init(1),
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
  LogicalResult runOnBlock(Block *block, DenseSet<Operation *> &copyNests);

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
std::unique_ptr<OpPassBase<FuncOp>> mlir::createAffineDataCopyGenerationPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace, unsigned tagMemorySpace,
    int minDmaTransferSize, uint64_t fastMemCapacityBytes) {
  return std::make_unique<AffineDataCopyGeneration>(
      slowMemorySpace, fastMemorySpace, tagMemorySpace, minDmaTransferSize,
      fastMemCapacityBytes);
}

/// Generate copies for this block. The block is partitioned into separate
/// ranges: each range is either a sequence of one or more operations starting
/// and ending with an affine load or store op, or just an affine.forop (which
/// could have other affine for op's nested within).
LogicalResult
AffineDataCopyGeneration::runOnBlock(Block *block,
                                     DenseSet<Operation *> &copyNests) {
  if (block->empty())
    return success();

  AffineCopyOptions copyOptions = {generateDma, slowMemorySpace,
                                   fastMemorySpace, tagMemorySpace,
                                   fastMemCapacityBytes};

  // Every affine.forop in the block starts and ends a block range for copying;
  // in addition, a contiguous sequence of operations starting with a
  // load/store op but not including any copy nests themselves is also
  // identified as a copy block range. Straightline code (a contiguous chunk of
  // operations excluding AffineForOp's) are always assumed to not exhaust
  // memory. As a result, this approach is conservative in some cases at the
  // moment; we do a check later and report an error with location info.
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

  // Create [begin, end) ranges.
  auto it = curBegin;
  while (it != block->end()) {
    AffineForOp forOp;
    // If you hit a non-copy for loop, we will split there.
    if ((forOp = dyn_cast<AffineForOp>(&*it)) && copyNests.count(forOp) == 0) {
      // Perform the copying up unti this 'for' op first.
      affineDataCopyGenerate(/*begin=*/curBegin, /*end=*/it, copyOptions,
                             copyNests);

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
        // Recurse onto the body of this loop.
        runOnBlock(forOp.getBody(), copyNests);
      } else {
        // We have enough capacity, i.e., copies will be computed for the
        // portion of the block until 'it', and for 'it', which is 'forOp'. Note
        // that for the latter, the copies are placed just before this loop (for
        // incoming copies) and right after (for outgoing ones).

        // Inner loop copies have their own scope - we don't thus update
        // consumed capacity. The footprint check above guarantees this inner
        // loop's footprint fits.
        affineDataCopyGenerate(/*begin=*/it, /*end=*/std::next(it), copyOptions,
                               copyNests);
      }
      // Get to the next load or store op after 'forOp'.
      curBegin = std::find_if(std::next(it), block->end(), [&](Operation &op) {
        return (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op) ||
                isa<AffineForOp>(op)) &&
               copyNests.count(&op) == 0;
      });
      it = curBegin;
    } else {
      assert(copyNests.count(&*it) == 0 &&
             "all copy nests generated should have been skipped above");
      // We simply include this op in the current range and continue for more.
      ++it;
    }
  }

  // Generate the copy for the final block range.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->isKnownTerminator() && "can't be a terminator");
    // Exclude the affine terminator - hence, the std::prev.
    affineDataCopyGenerate(/*begin=*/curBegin, /*end=*/std::prev(block->end()),
                           copyOptions, copyNests);
  }

  return success();
}

void AffineDataCopyGeneration::runOnFunction() {
  FuncOp f = getFunction();
  OpBuilder topBuilder(f.getBody());
  zeroIndex = topBuilder.create<ConstantIndexOp>(f.getLoc(), 0);

  // Nests that are copy-in's or copy-out's; the root AffineForOps of those
  // nests are stored herein.
  DenseSet<Operation *> copyNests;

  // Clear recorded copy nests.
  copyNests.clear();

  for (auto &block : f)
    runOnBlock(&block, copyNests);

  // Promote any single iteration loops in the copy nests.
  for (auto nest : copyNests) {
    nest->walk([](AffineForOp forOp) { promoteIfSingleIteration(forOp); });
  }
}

static PassRegistration<AffineDataCopyGeneration>
    pass("affine-data-copy-generate",
         "Generate explicit copying for memory operations");
