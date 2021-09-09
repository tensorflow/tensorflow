/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

namespace {

static size_t computeUserangeSize(const UseInterval &interval);

static size_t computeByteSize(const Value &v);

static size_t computeAlignedSegments(const Value &v);

/// Returns the length of an userange interval.
static size_t computeUserangeSize(const UseInterval &interval) {
  return interval.end - interval.start + 1;
}

/// Compute the byte size of a given Value.
static size_t computeByteSize(const Value &v) {
  return v.getType().cast<ShapedType>().getSizeInBits() / 8;
}

/// Compute the 64 byte alinged segments of a given Value.
static size_t computeAlignedSegments(const Value &v) {
  size_t padding = 64;
  size_t bytes = computeByteSize(v);
  return std::ceil(bytes / (double)padding);
}

struct PackedInfo {
public:
  PackedInfo(Value source, size_t bufferId, size_t offset)
      : source(source), bufferId(bufferId), offset(offset) {}

  size_t bufferId;
  Value source;
  size_t offset;
};

/// Contains the important informations about a buffer allocation.
struct AllocationInfo {
public:
  using SizedGap = std::pair<UseInterval, size_t>;

public:
  AllocationInfo(Value alloc, size_t allocUserangeId, size_t firstUse,
                 size_t lastUse, size_t numSegments, size_t windowId,
                 const UseInterval::Vector *userangeIntervals)
      : alloc(alloc), allocUserangeId(allocUserangeId), firstUse(firstUse),
        lastUse(lastUse), numSegments(numSegments), windowId(windowId),
        userangeIntervals(userangeIntervals) {}

  /// The allocation value.
  Value alloc;

  /// The id of allocation based on the Userange Analysis.
  size_t allocUserangeId;

  /// The first use of the buffer.
  size_t firstUse;

  /// The last use of the buffer based on the Userange Analysis.
  size_t lastUse;

  /// The number of 64 byte aligned segments of contigous memory.
  size_t numSegments;

  /// The window id of the allocation position.
  size_t windowId;

  /// The userange intervals of the buffer.
  const UseInterval::Vector *userangeIntervals;

  /// Compute the gas of the alloc userange.
  std::vector<SizedGap> computeGaps(size_t maxUserangeId = 0) {
    std::vector<SizedGap> gaps;
    UseInterval::Vector useranges = *(userangeIntervals);
    auto useRangeIter = useranges.begin();

    // Add a dummy gap in front of the frist use of the buffer.
    if (useRangeIter->start > 0) {
      gaps.push_back(
          std::make_pair(UseInterval(0, useRangeIter->start - 1), numSegments));
    }

    // Compute the gap between two userange intervals.
    for (auto nextUseRangeIter = std::next(useranges.begin());
         nextUseRangeIter != useranges.end(); ++nextUseRangeIter) {
      gaps.push_back(std::make_pair(
          UseInterval(useRangeIter->end + 1, nextUseRangeIter->start - 1),
          this->numSegments));
      useRangeIter = nextUseRangeIter;
    }

    // Add a dummy gap behind the last use of the buffer.
    if (useRangeIter->end < maxUserangeId) {
      gaps.push_back(std::make_pair(
          UseInterval(useRangeIter->end + 1, maxUserangeId), numSegments));
    }

    return gaps;
  }

  /// Compute the userange size.
  size_t getUserangeSize() const { return lastUse - firstUse + 1; }
};

// Comperator to sort allocation informations by window id, userange and by
// number of memory segments.
class AllocInfoWinIdCompare {
public:
  bool operator()(const AllocationInfo &a, const AllocationInfo &b) {
    if (a.windowId == b.windowId) {
      if (a.allocUserangeId == b.allocUserangeId)
        return a.numSegments > b.numSegments;
      return a.allocUserangeId > b.allocUserangeId;
    }
    return a.windowId < b.windowId;
  }
};

// Comperator to sort the allocation informations by number of segments.
class AllocInfoMemSizeCompare {
public:
  bool operator()(const AllocationInfo &a, const AllocationInfo &b) {
    return a.numSegments > b.numSegments;
  }
};

/// This approach computes an allocation information list and sortes it by
/// a given comperator. From top to bottom the algortihm tries to fill userange
/// gaps with appropriate buffers behind it, to optimze the memory.
template <typename CompareT> class SortedPackingStrategy {
public:
  using AllocInfoList = std::vector<AllocationInfo>;
  using SizedGap = std::pair<UseInterval, size_t>;

public:
  /// Constructs the Sorted Packing Strategy.
  SortedPackingStrategy(size_t windowSize, CompareT compare)
      : windowSize(windowSize), compare(compare) {}

  /// Optimize the buffer allocations.
  void optimze(const mlir::BufferPlacementAllocs &allocs,
               const UserangeAnalysis &userangeAnalysis,
               std::vector<PackedInfo> &packingInfos,
               DenseMap<size_t, size_t> &bufferGen) {
    AllocInfoList allocInfos;
    allocInfos.reserve(std::distance(allocs.begin(), allocs.end()));

    // Create allocInformations and store them in allocInfos.
    size_t maxUserangeId =
        computeAllocationInfos(allocInfos, userangeAnalysis, allocs);

    // Sort the allocation infos.
    std::sort(allocInfos.begin(), allocInfos.end(), compare);

    size_t optMemory = 0;
    size_t currentOffset = 0;
    size_t globalBufferOffset = 0;

    for (auto currentIter = allocInfos.begin(); currentIter != allocInfos.end();
         ++currentIter) {
      size_t offsetInBytes = currentOffset * 64;
      UseInterval::Vector useranges = *(currentIter->userangeIntervals);

      // Compute userange gaps.
      std::vector<std::pair<UseInterval, size_t>> gaps =
          currentIter->computeGaps(maxUserangeId);

      // Add the current buffer to the packing infos.
      packingInfos.emplace_back(currentIter->alloc, bufferId, offsetInBytes);
      if (gaps.empty())
        continue;

      for (auto checkedAllocInfoIter = std::next(currentIter);
           checkedAllocInfoIter != allocInfos.end();) {

        // Check if a gap exists to pack the memory into.
        // If not continue.
        if (!findGapAndUpdate(gaps, packingInfos, *checkedAllocInfoIter,
                              *currentIter, offsetInBytes)) {
          ++checkedAllocInfoIter;
          continue;
        }
        optMemory += checkedAllocInfoIter->numSegments;
        checkedAllocInfoIter = allocInfos.erase(checkedAllocInfoIter);
      }
      // Increase the total global memory offset.
      currentOffset += currentIter->numSegments;
    }

    size_t totalBufferSize = currentOffset * 64;
    bufferGen.insert(std::make_pair(bufferId, totalBufferSize));
  }

private:
  const size_t windowSize;
  const size_t bufferId = 0;
  const CompareT compare;

  /// We try to find an appropriate userange gap to pack the buffer into it.
  /// If we find one we update only the gaps and the buffer offset map.
  bool findGapAndUpdate(std::vector<std::pair<UseInterval, size_t>> &gaps,
                        std::vector<PackedInfo> &allocBufferOffsets,
                        AllocationInfo &allocToPack,
                        AllocationInfo &allocToPackInto,
                        size_t unitedBufferOffset) {
    // Check if the buffer to pack into has enough memory.
    if (allocToPackInto.numSegments < allocToPack.numSegments)
      return false;
    for (auto gapIter = gaps.begin(); gapIter != gaps.end();) {
      size_t gapUserangeSize = computeUserangeSize(gapIter->first);

      // If a gap interval has no free contiguous memory anymore, erease it from
      // list.
      if (gapIter->second <= 0) {
        gapIter = gaps.erase(gapIter);
        continue;
      }

      // Checks if enough the userange and contigous memory segments is
      // free.
      if (gapIter->second < allocToPack.numSegments ||
          allocToPack.firstUse < gapIter->first.start ||
          allocToPack.lastUse > gapIter->first.end) {
        ++gapIter;
        continue;
      }

      // Stores the packed buffer with the offset.
      allocBufferOffsets.emplace_back(
          allocToPack.alloc, bufferId,
          unitedBufferOffset +
              (allocToPackInto.numSegments - gapIter->second) * 64);
      size_t freeContiguousMemory = gapIter->second;
      size_t oldStart = gapIter->first.start;
      size_t oldEnd = gapIter->first.end;

      // Update gap
      gapIter->second = freeContiguousMemory - allocToPack.numSegments;

      // Check if the gap must be splitted.
      if (gapUserangeSize > allocToPack.getUserangeSize()) {
        gapIter->first.end = allocToPack.lastUse;
        gapIter->first.start = allocToPack.firstUse;

        if (allocToPack.lastUse < oldEnd)
          gaps.push_back(
              std::make_pair(UseInterval(allocToPack.lastUse + 1, oldEnd),
                             freeContiguousMemory));
        if (allocToPack.firstUse > oldStart)
          gaps.push_back(
              std::make_pair(UseInterval(oldStart, allocToPack.firstUse - 1),
                             freeContiguousMemory));
      }
      return true;
    }
    return false;
  }

  /// Aggreagtes the allocation informations of the allocs.
  size_t computeAllocationInfos(AllocInfoList &allocInfos,
                                const UserangeAnalysis &userangeAnalysis,
                                const mlir::BufferPlacementAllocs &allocs) {
    // Create allocInformations and store them in allocInfos.
    size_t maxUserangeId = 0;

    for (auto &allocEntry : allocs) {
      Value v = std::get<0>(allocEntry);
      auto userangeIntervals = userangeAnalysis.getUserangeInterval(v);

      if (!userangeIntervals)
        continue;

      // Computes the userange id of the allocation.
      size_t allocUserangeId = userangeAnalysis.computeId(v, v.getDefiningOp());

      // Computes the last use of the allocated buffer.
      size_t lastUse = std::prev((*userangeIntervals.getValue()).end())->end;

      // Computes the first use of the allocated buffer.
      size_t firstUse = (*userangeIntervals.getValue()).begin()->start;

      // Computes the number of aligend segments of the buffer.
      size_t numSegments = computeAlignedSegments(v);
      maxUserangeId = std::max(maxUserangeId, lastUse);
      allocInfos.emplace_back(v, allocUserangeId, firstUse, lastUse,
                              numSegments, 0, userangeIntervals.getValue());
    }

    // If the window size is zero we are ready.
    if (windowSize == 0)
      return maxUserangeId;
    // Sorts the allocation informations to compute the window id.
    std::sort(allocInfos.begin(), allocInfos.end(),
              [](const AllocationInfo &a, const AllocationInfo &b) {
                return a.allocUserangeId < b.allocUserangeId;
              });

    // resize window id
    size_t windowId = 0;
    size_t lastAllocUserangeId = 0;
    for (auto &allocationInfo : allocInfos) {
      if (allocationInfo.allocUserangeId > lastAllocUserangeId + windowSize)
        ++windowId;

      lastAllocUserangeId = allocationInfo.allocUserangeId;
      allocationInfo.windowId = windowId;
    }
    return maxUserangeId;
  }
};

/// Template to reuses already allocated buffer to save allocation operations.
class BufferPacking : BufferPlacementTransformationBase {

public:
  template <typename StrategyT>
  BufferPacking(Operation *op, StrategyT strategy)
      : BufferPlacementTransformationBase(op),
        userangeAnalysis(op, allocs, aliases), dominators(op) {
    DenseMap<size_t, size_t> bufferGen;
    std::vector<PackedInfo> packingInfos;
    strategy.optimze(allocs, userangeAnalysis, packingInfos, bufferGen);

    DenseMap<size_t, Value> generatedBuffers;

    // Find common dominators.
    Block *block = findAllocationsDominator(packingInfos);
    // Find alloc position operation.
    mlir::OpBuilder packBuilder(&(block->front()));
    auto location = block->front().getLoc();

    // Create the packed AllocOps.
    for (auto &entry : bufferGen) {
      auto memrefType =
          MemRefType::get({entry.second}, packBuilder.getIntegerType(8));
      Value packedAlloc =
          packBuilder.create<memref::AllocOp>(location, memrefType);
      generatedBuffers.insert(std::make_pair(entry.first, packedAlloc));
    }

    for (auto &packInfo : packingInfos) {
      Value currentAlloc = packInfo.source;
      Value targetBuffer = generatedBuffers[packInfo.bufferId];
      size_t offset = packInfo.offset;
      Operation *viewDefOp = currentAlloc.getDefiningOp();
      Location loc = viewDefOp->getLoc();
      mlir::OpBuilder viewBuilder(viewDefOp);

      // Create a ConstantOp with the aligned offset.
      Value constantOp = viewBuilder.create<mlir::ConstantOp>(
          loc, viewBuilder.getIndexType(),
          viewBuilder.getIntegerAttr(viewBuilder.getIndexType(), offset));

      // Store the operands for the ViewOp.
      SmallVector<Value, 4> newOperands{targetBuffer};
      newOperands.push_back(constantOp);
      ShapedType shape = currentAlloc.getType().cast<ShapedType>();

      // Create a ViewOp with the shape of the old alloc and use the created
      // packed alloc and the constant for the operands.
      Value viewOp = viewBuilder.create<memref::ViewOp>(
          loc, shape.cast<MemRefType>(), newOperands);

      // Replace all old allocs references with the created ViewOp and
      // afterwards remove the old allocs.
      currentAlloc.replaceAllUsesWith(viewOp);
      viewDefOp->erase();
    }
  }

private:
  UserangeAnalysis userangeAnalysis;
  /// The current dominance info.
  DominanceInfo dominators;

  /// Find the block that dominates all buffer allocations.
  Block *findAllocationsDominator(const std::vector<PackedInfo> &packingInfos) {
    SmallPtrSet<Value, 16> allocValues;
    for (auto &packInfo : packingInfos) {
      allocValues.insert(packInfo.source);
    }

    // Find common dominators.
    return findCommonDominator(packingInfos.begin()->source, allocValues,
                               dominators);
  }
};

struct BufferPackingPass : public BufferPackingBase<BufferPackingPass> {
  explicit BufferPackingPass(unsigned windowSize) {
    this->window_size_ = windowSize;
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();

    if (window_size_ == 0) {
      SortedPackingStrategy<AllocInfoMemSizeCompare> strategy(
          window_size_, AllocInfoMemSizeCompare());
      BufferPacking packing(getFunction(), strategy);
    } else {
      SortedPackingStrategy<AllocInfoWinIdCompare> strategy(
          window_size_, AllocInfoWinIdCompare());
      BufferPacking packing(getFunction(), strategy);
    }
  }
};

<<<<<<< HEAD
=======
/// Reuses already allocated buffer to save allocation operations.
class MemoryCount : BufferPlacementTransformationBase {
public:
  using AllocList = std::vector<Value>;

public:
  MemoryCount(Operation *op) : BufferPlacementTransformationBase(op) {
    // Map that stores the allocs to their respective shape.
    DenseMap<Type, AllocList> shapeMap;

    // Padding in Byte
    const size_t padding = 64;
    size_t totalSize = 0;

    // Iterate over all allocs and fill the shapeMap accordingly.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      auto shape = alloc.getType().cast<ShapedType>();
      size_t shapeBytes = shape.getSizeInBits() / 8;
      size_t alignFactor = llvm::divideCeil(shapeBytes, padding);
      size_t size = alignFactor * padding;

      // llvm::errs() << "Alloc: " << alloc << ", Size: " << size << " Byte.\n";
      totalSize += size;
    }

    llvm::errs() << "Total size: " << totalSize << "\n";
  }
};

struct MemoryCountPass : MemoryCountBase<MemoryCountPass> {
  void runOnFunction() override { MemoryCount counter(getFunction()); }
};

>>>>>>> 7c4c319549f... Buffer sorting startegy get template of comperator
} // namespace

std::unique_ptr<FunctionPass> createBufferPackingPass(unsigned window_size) {
  return std::make_unique<BufferPackingPass>(window_size);
}

} // namespace mlir
