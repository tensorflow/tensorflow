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

/// Represents a memory interval.
struct MemoryInterval {
public:
  MemoryInterval(size_t begin, size_t end) : begin(begin), end(end) {}

  /// Unite two intervals if they overlap.
  bool intervalUnion(const MemoryInterval &other) {
    if (other.begin > end + 1 || other.end + 1 < begin)
      return false;
    begin = std::min(other.begin, begin);
    end = std::max(other.end, end);
    return true;
  }

  size_t getSize() { return end + 1 - begin; }

  size_t begin;
  size_t end;

  // TODO: Remove Debug Dump.
  void dump() { llvm::errs() << "(" << begin << ", " << end << ")\n"; }
};

/// The PackingInfo contains all important information of a value for the buffer
/// packing.
struct PackingInfo {
  using MemoryIntervalVector = std::vector<MemoryInterval>;

public:
  PackingInfo(Value value, UserangeAnalysis analysis)
      : alloc(std::move(value)), userange(std::move(analysis)) {
    // Compute the byte size and the aligned byte size of the alloc.
    lastUsedByte = computeByteSize(alloc);
    numAlignedSegments = computeAlignedSegments(alloc);
    UserangeAnalysis::IntervalVector intervals = userange.getUserangeOf(alloc);

    MemoryInterval memoryInterval((size_t)0, numAlignedSegments - 1);
    MemoryIntervalVector alignedBufferSegment{memoryInterval};

    // Distribute the aligned buffer segments for the whole useInterval. Also
    // save the userange IDs for later back mapping.
    for (auto &interval : intervals) {
      for (size_t i = interval.first, e = interval.second; i <= e; ++i) {
        memoryDistribution.insert(std::make_pair(i, alignedBufferSegment));
        bufferUserangeIDs.emplace_back(i);
      }
    }

    occupiedMemoryIntervals.insert(std::make_pair(alloc, memoryInterval));
  }

  /// Compute the 64 byte alinged segments of a given Value.
  size_t computeAlignedSegments(Value v) {
    size_t padding = 64;
    size_t bytes = computeByteSize(v);
    return std::ceil(bytes / (double)padding);
  }

  /// Compute the byte size of a given Value.
  size_t computeByteSize(Value v) {
    return v.getType().cast<ShapedType>().getSizeInBits() / 8;
  }

  /// Compute the potential reusers from the given PackingInfo vector. If the
  /// useIntervals interfere and the aligned buffer size of the other
  /// PackingInfo is bigger, we continue. Otherwise we order the potential
  /// reusers by their aligned buffer size.
  void computePotentialReuser(std::vector<PackingInfo> &other) {
    for (auto info : other) {
      Value v = info.alloc;
      // Skip if the other PackingInfo is this PackingInfo.
      if (alloc == v)
        continue;

      // If the useIntervals of this PackingInfo and the other PackingInfo
      // interfere, we cannot reuse the other PackingInfo.
      if (userange.rangesInterfere(alloc, info.alloc))
        continue;

      if (info.numAlignedSegments > numAlignedSegments)
        continue;

      // Find the correct place to insert the potential reuser based on the
      // aligned buffer size.
      auto iter = potentialReuser.begin();
      while (iter != potentialReuser.end() &&
             iter->numAlignedSegments >= info.numAlignedSegments)
        ++iter;

      potentialReuser.insert(iter, info);
    }
  }

  /// Try to merge all potential reuser in the current buffer and compute a new
  /// buffer memory distribution.
  void packPotentialReuser(DenseSet<Value> &alreadyUnited) {
    // Iterate over all potential reusers and check if possible candidates fit
    // into buffer.
    for (PackingInfo &candidate : potentialReuser) {
      tryPackCandidate(candidate, alreadyUnited);
    }
  }

  Value alloc;
  size_t numAlignedSegments;
  std::vector<PackingInfo> potentialReuser;
  DenseMap<size_t, MemoryIntervalVector> memoryDistribution;
  std::vector<size_t> bufferUserangeIDs;
  DenseMap<Value, MemoryInterval> occupiedMemoryIntervals;
  size_t lastUsedByte;

private:
  UserangeAnalysis userange;

  /// Check if the first segments are not occupied and there is enough
  /// contiguous memory. If not the memory distribution is updated.
  llvm::Optional<size_t>
  updateFirstSegmentUsage(MemoryIntervalVector &userangeOccupiedMemoryDist,
                          MemoryInterval const &intervalToCheck,
                          PackingInfo const &candidate) {
    if (intervalToCheck.begin < candidate.numAlignedSegments) {
      // Check if overlapping with first interval. If not we must add a new
      // first interval.
      if (!userangeOccupiedMemoryDist.begin()->intervalUnion(intervalToCheck))
        userangeOccupiedMemoryDist.insert(userangeOccupiedMemoryDist.begin(),
                                          intervalToCheck);

      return 0;
    }
    return llvm::None;
  }

  /// Find the relevant intervals that change the userange occupation of the
  /// memory.
  MemoryIntervalVector::iterator findRelevantIntervalsToAdd(
      MemoryIntervalVector const &userangeOccupiedDistribution,
      MemoryIntervalVector &intervalsToAdd,
      llvm::Optional<size_t> const &intervalOffset) {
    auto addIntervalIter = intervalsToAdd.begin();
    if (intervalOffset.hasValue()) {
      auto currentMemoryLocIter = std::next(
          userangeOccupiedDistribution.begin(), intervalOffset.getValue());
      addIntervalIter = std::find_if(
          addIntervalIter, intervalsToAdd.end(), [&](MemoryInterval interval) {
            return interval.end > currentMemoryLocIter->end;
          });
    }
    return addIntervalIter;
  }

  /// Check if Contigous emory is availabel and update the interval offset to
  /// the appropriate position if possible.
  bool
  updateContigousMemoryOffset(llvm::Optional<size_t> &intervalOffset,
                              MemoryIntervalVector &userangeOccupiedMemoryDist,
                              PackingInfo const &candidate) {
    // Because of the dummy interval the iterator points never to end.
    auto iter = userangeOccupiedMemoryDist.begin();
    size_t firstOccupiedMemorySegemt = iter->begin;
    bool foundPreviousGap =
        candidate.numAlignedSegments <= firstOccupiedMemorySegemt;
    intervalOffset = foundPreviousGap ? llvm::None : llvm::Optional<size_t>(0);

    // counter to find the memory interval offset.
    size_t count = 0;
    for (;;) {
      auto next = std::next(iter);
      if (next == userangeOccupiedMemoryDist.end())
        break;
      // Check interval union.
      if (next->intervalUnion(*iter)) {
        iter = userangeOccupiedMemoryDist.erase(iter);
        continue;
      }

      // Found a maximal occupied contigous memory segement and check if the
      // following gap has enough memory to pack the candidate and there exists
      // no previous suitable gap.
      if (!foundPreviousGap) {
        size_t currentGapSize = next->begin - iter->end - 1;
        if (currentGapSize >= candidate.numAlignedSegments) {
          foundPreviousGap = true;
          intervalOffset = count;
        }
      }
      ++count;
      iter = next;
    }
    return foundPreviousGap;
  }

  /// Inserts an occupied memory interval to the memory distribution and update
  /// the userange.
  void packCandidate(PackingInfo &candidate, size_t segment,
                     DenseSet<Value> &alreadyUnited) {

    alreadyUnited.insert(candidate.alloc);
    updateMemoryDistribution(candidate, segment);
    updateOccupiedMemoryIntervals(candidate, segment);
    std::sort(bufferUserangeIDs.begin(), bufferUserangeIDs.end());
  }

  /// Update the memory Distribution of all userange IDs of the candidate and
  /// unit the occupied memory intervals if possible.
  void updateMemoryDistribution(PackingInfo const &candidate, size_t segment) {
    // The new occupied memory interval to insert.
    MemoryInterval addedMemoryInterval =
        MemoryInterval(segment, segment + candidate.numAlignedSegments - 1);

    // Iterate over the candidate userange to update the memory distribution.
    for (size_t userangeID : candidate.bufferUserangeIDs) {
      auto iterUserange = memoryDistribution.find(userangeID);

      // Case found with no memory usage at this userangeID
      if (iterUserange == memoryDistribution.end()) {
        memoryDistribution.insert(std::make_pair(
            userangeID, MemoryIntervalVector{addedMemoryInterval}));
        bufferUserangeIDs.emplace_back(userangeID);
        continue;
      }

      MemoryIntervalVector &memIntervals = iterUserange->second;
      auto iter = std::find_if(memIntervals.begin(), memIntervals.end(),
                               [&](MemoryInterval interval) {
                                 return interval.begin >=
                                        segment + candidate.numAlignedSegments;
                               });

      // Check if we must enlarge the first memory interval or insert a new
      // interval before.
      if (iter == memIntervals.begin()) {
        if (iter->begin == segment + candidate.numAlignedSegments)
          iter->begin = segment;
        else
          memIntervals.insert(iter, addedMemoryInterval);
        continue;
      }
      auto prevIter = std::prev(iter);

      // Check if we fit exact.
      if (iter != memIntervals.end() &&
          addedMemoryInterval.begin == prevIter->end + 1 &&
          addedMemoryInterval.end == iter->begin) {
        prevIter->end = iter->end;
        memIntervals.erase(iter);
      }
      // Check if we must merge us with the left neighbour
      else if (addedMemoryInterval.begin == prevIter->end + 1) {
        prevIter->end = addedMemoryInterval.end;
      }
      // Check if we must merge us with the right neighbour
      else if (iter != memIntervals.end() &&
               addedMemoryInterval.end + 1 == iter->begin) {
        iter->begin = addedMemoryInterval.begin;
      }
      // We must insert the interval.
      else {
        memIntervals.insert(iter, addedMemoryInterval);
      }
    }
  }

  /// Compute the beginning of the memory segment to occupie.
  size_t computeSegment(llvm::Optional<size_t> &intervalOffset,
                        MemoryIntervalVector &userangeMemoryDist) {
    size_t segment = 0;
    if (intervalOffset.hasValue())
      segment = userangeMemoryDist[intervalOffset.getValue()].end + 1;
    return segment;
  }

  /// Compute the packed allocs memory intervals offset and update the map.
  void updateOccupiedMemoryIntervals(PackingInfo const &candidate,
                                     size_t segment) {
    for (auto entry : candidate.occupiedMemoryIntervals) {
      MemoryInterval currentAllocInterval = entry.second;
      Value currentAlloc = entry.first;
      currentAllocInterval.begin += segment;
      currentAllocInterval.end += segment;
      occupiedMemoryIntervals.insert(
          std::make_pair(currentAlloc, currentAllocInterval));
    }
  }

  /// Compute a distribution of contiguos memory intervals over the userange of
  /// the candidate. If we have find a segment with enough contigous memory we
  /// pack the candidate.
  void tryPackCandidate(PackingInfo &candidate,
                        DenseSet<Value> &alreadyUnited) {

    llvm::Optional<size_t> intervalOffset = llvm::None;
    std::vector<size_t> &userangeToCheck = candidate.bufferUserangeIDs;

    // Initialize the memory distribution over the userange with a dummy
    // interval.
    MemoryIntervalVector userangeOccupiedMemoryDist{
        MemoryInterval(numAlignedSegments, numAlignedSegments)};

    // Iteration over all userange IDs of the candidate to create the
    // memeorydistribution and find contigous memory segment to pack the
    // candidate if possible.
    for (size_t userangeID : userangeToCheck) {
      auto iterUserange = memoryDistribution.find(userangeID);

      // No memory used at the userangeID
      if (iterUserange == memoryDistribution.end())
        continue;
      MemoryIntervalVector &addIntervals = iterUserange->second;

      // Check if we occupy the first segments of memory.
      if (!intervalOffset.hasValue())
        intervalOffset = updateFirstSegmentUsage(
            userangeOccupiedMemoryDist, *addIntervals.begin(), candidate);

      auto addIntervalIter = findRelevantIntervalsToAdd(
          userangeOccupiedMemoryDist, addIntervals, intervalOffset);

      userangeOccupiedMemoryDist.insert(userangeOccupiedMemoryDist.end(),
                                        addIntervalIter, addIntervals.end());

      std::sort(userangeOccupiedMemoryDist.begin(),
                userangeOccupiedMemoryDist.end(),
                [&](MemoryInterval a, MemoryInterval b) {
                  if (a.begin == b.begin)
                    return a.end > b.end;

                  return a.begin < b.begin;
                });

      // Check contiguous memory.
      if (!updateContigousMemoryOffset(intervalOffset,
                                       userangeOccupiedMemoryDist, candidate))
        return;
    }
    size_t segment = computeSegment(intervalOffset, userangeOccupiedMemoryDist);
    packCandidate(candidate, segment, alreadyUnited);
  }

}; // namespace

/// Reuses already allocated buffer to save allocation operations.
class BufferPacking : BufferPlacementTransformationBase {
public:
  using AllocList = std::vector<Value>;

public:
  BufferPacking(Operation *op) : BufferPlacementTransformationBase(op) {
    std::vector<PackingInfo> packInfos;
    UserangeAnalysis userange(op, allocs, aliases);
    for (auto allocEntry : allocs) {
      Value v = std::get<0>(allocEntry);
      PackingInfo info(v, userange);

      auto iterPos = std::find_if(
          packInfos.begin(), packInfos.end(), [&](PackingInfo pInfo) {
            return pInfo.numAlignedSegments >= info.numAlignedSegments;
          });
      packInfos.insert(iterPos, info);
    }

    for (auto &infos : packInfos)
      infos.computePotentialReuser(packInfos);

    DenseSet<Value> alreadUnitedBuffer;
    for (auto &infos : packInfos) {
      if (alreadUnitedBuffer.contains(infos.alloc))
        continue;
      infos.packPotentialReuser(alreadUnitedBuffer);
    }

    for (Value unitedAlloc : alreadUnitedBuffer) {
      for (auto iter = packInfos.begin(), e = packInfos.end(); iter != e;
           ++iter) {
        if (iter->alloc == unitedAlloc) {
          packInfos.erase(iter);
          break;
        }
      }
    }

    DenseMap<Value, size_t> occupiedMemoryIntervals;
    size_t totalBufferSize = 0;

    for (auto infosIter = packInfos.begin(), e = packInfos.end();
         infosIter != e; ++infosIter) {
      for (auto &entry : infosIter->occupiedMemoryIntervals) {
        occupiedMemoryIntervals.insert(std::make_pair(
            entry.first, entry.second.begin * 64 + totalBufferSize));
      }
      totalBufferSize += infosIter->numAlignedSegments * 64;
    }

    // Create the packed AllocOp.
    Value firstAlloc = std::get<0>(*allocs.begin());

    Operation *packDefOp = firstAlloc.getDefiningOp();
    mlir::OpBuilder packBuilder(packDefOp);
    auto memrefType =
        MemRefType::get({totalBufferSize}, packBuilder.getIntegerType(8));
    Value packedAlloc =
        packBuilder.create<memref::AllocOp>(packDefOp->getLoc(), memrefType);

    // Iterate over all allocs and create a ConstantOp with aligned offset.
    for (auto &entry : occupiedMemoryIntervals) {
      Value currentAlloc = entry.first;
      size_t offset = entry.second;

      // Initialize a OpBuilder for the ConstantOp and ViewOp.
      Operation *viewDefOp = currentAlloc.getDefiningOp();
      Location loc = viewDefOp->getLoc();
      mlir::OpBuilder viewBuilder(viewDefOp);

      // Create a ConstantOp with the aligned offset.
      Value constantOp = viewBuilder.create<mlir::ConstantOp>(
          loc, viewBuilder.getIndexType(),
          viewBuilder.getIntegerAttr(viewBuilder.getIndexType(), offset));

      // Store the operands for the ViewOp.
      SmallVector<Value, 4> newOperands{packedAlloc};
      newOperands.emplace_back(constantOp);

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
};

struct BufferPackingPass : public BufferPackingBase<BufferPackingPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    context->getOrLoadDialect("std");
    BufferPacking packing(getFunction());
  }
};

} // namespace

std::unique_ptr<FunctionPass> createBufferPackingPass() {
  return std::make_unique<BufferPackingPass>();
}

} // end namespace mlir
