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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

namespace {

/// The PackingInfo contains all important information of a value for the buffer
/// packing.
struct PackingInfo {
  using PackingOffsetPair = std::pair<std::pair<size_t, size_t>, Value>;

public:
  PackingInfo(Value value, UserangeAnalysis analysis)
      : alloc(std::move(value)), userange(std::move(analysis)) {
    // Compute the byte size and the aligned byte size of the alloc.
    lastUsedByte = computeByteSize(alloc);
    numAlignedSegments = computeAlignedSegments(alloc);

    // Create an UseInterval that contains the first and last use of the value.
    // Gaps are not considered because the memory must be allocated during this
    // time.
    totalUseInterval = UserangeAnalysis::UseInterval(
        userange.getFirstUseIndex(alloc).getValue(),
        userange.getLastUseIndex(alloc).getValue());

    std::vector<size_t> userangeIDs;
    std::vector<PackingOffsetPair> alignedBufferSegment{std::make_pair(
        std::make_pair((size_t)0, numAlignedSegments - 1), alloc)};
    // Distribute the aligned buffer segments for the whole useInterval. Also
    // save the userange IDs for later back mapping.
    for (size_t i = totalUseInterval.first, e = totalUseInterval.second; i <= e;
         ++i) {
      memoryDistribution.insert(std::make_pair(i, alignedBufferSegment));
      userangeIDs.emplace_back(i);
    }

    allocToUserangeIDs.insert(std::make_pair(alloc, userangeIDs));
    valueOffset.insert(std::make_pair(
        alloc, std::make_pair((size_t)0, numAlignedSegments - 1)));
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
      if (userange.getLastUseIndex(alloc) >= info.totalUseInterval.first &&
          userange.getFirstUseIndex(alloc) <= info.totalUseInterval.second)
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
      // Create a copy of the current memory distribution.
      DenseMap<size_t, std::vector<PackingOffsetPair>> possibleMemeoryDist(
          memoryDistribution);
      DenseMap<Value, std::pair<size_t, size_t>> possibleOffsetToAdd;
      size_t possibleLastUsedByte = candidate.lastUsedByte;

      // Check if the candidate is already packed into an other buffer.
      if (alreadyUnited.contains(candidate.alloc))
        continue;
      // Flag if we can unite the buffers.
      bool canUnite = true;

      // Iterate over allocs to check if we can allocate the same buffer
      // segments for all userangeIDs.
      for (auto &cInterval : candidate.allocToUserangeIDs) {
        // The possible aligned segments that can be used.
        std::vector<size_t> possibleOffsets;
        for (size_t i = 0,
                    e = (numAlignedSegments - candidate.numAlignedSegments);
             i <= e; ++i)
          possibleOffsets.emplace_back(i);
        Value candidateAlloc = cInterval.first;
        size_t candidateSegments = computeAlignedSegments(candidateAlloc);
        std::vector<size_t> userangeIDs = cInterval.second;
        for (size_t userangeID : userangeIDs) {
          std::vector<PackingOffsetPair> &occupiedIntervals =
              possibleMemeoryDist[userangeID];
          // Check if any interval is occupied
          if (occupiedIntervals.empty())
            continue;

          // Delete occupied interval from candidate list.
          // If we find a interval (a,b) we must delete all candidate
          // (a - segemnts needed  + 1,b)
          for (auto &entry : occupiedIntervals) {
            std::pair<size_t, size_t> interval = entry.first;
            size_t start = candidateSegments > interval.first
                               ? 0
                               : interval.first - candidateSegments + 1;

            // Iterate over the expanded interval to erase possible
            // offsets.
            for (size_t i = start, e = interval.second; i <= e; ++i) {
              auto iter =
                  std::find(possibleOffsets.begin(), possibleOffsets.end(), i);
              // Check if we found an interference and delete it.
              if (iter != possibleOffsets.end())
                possibleOffsets.erase(iter);
            }
          }
        }

        // Check if we have possible offset if not we must not unite the
        // buffers.
        if (possibleOffsets.empty()) {
          canUnite = false;
          break;
        }
        auto minOffsetIter =
            std::min(possibleOffsets.begin(), possibleOffsets.end());
        std::pair<size_t, size_t> interval = std::make_pair(
            *minOffsetIter, (*minOffsetIter) + candidateSegments - 1);
        possibleOffsetToAdd.insert(std::make_pair(candidateAlloc, interval));
        for (size_t time : userangeIDs) {
          std::vector<PackingOffsetPair> otherIntervals =
              possibleMemeoryDist[time];
          otherIntervals.emplace_back(std::make_pair(interval, candidateAlloc));
          std::sort(otherIntervals.begin(), otherIntervals.end(),
                    [&](PackingOffsetPair a, PackingOffsetPair b) {
                      return a.first.first - b.first.first;
                    });
          possibleMemeoryDist[time] = otherIntervals;
        }
      }
      if (canUnite) {
        // Update the memory distribution.
        memoryDistribution = possibleMemeoryDist;

        // Update the alloc to userangeIDs mapping.
        allocToUserangeIDs.insert(candidate.allocToUserangeIDs.begin(),
                                  candidate.allocToUserangeIDs.end());
        // Add the alloc candidate to the united buffers.
        alreadyUnited.insert(candidate.alloc);
        valueOffset.insert(possibleOffsetToAdd.begin(),
                           possibleOffsetToAdd.end());
      }
    }
  }

  Value alloc;
  size_t numAlignedSegments;
  UserangeAnalysis::UseInterval totalUseInterval;
  std::vector<PackingInfo> potentialReuser;
  DenseMap<size_t, std::vector<PackingOffsetPair>> memoryDistribution;
  DenseMap<Value, std::vector<size_t>> allocToUserangeIDs;
  DenseMap<Value, std::pair<size_t, size_t>> valueOffset;
  size_t lastUsedByte;

private:
  UserangeAnalysis userange;
};

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

      auto iter = packInfos.begin();
      while (iter != packInfos.end() &&
             iter->numAlignedSegments >= info.numAlignedSegments)
        ++iter;
      packInfos.insert(iter, info);
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

    DenseMap<Value, size_t> valueOffsets;
    size_t totalBufferSize = 0;

    for (auto infosIter = packInfos.begin(), e = packInfos.end();
         infosIter != e; ++infosIter) {
      for (auto &entry : infosIter->valueOffset) {
        valueOffsets.insert(std::make_pair(
            entry.first, entry.second.first * 64 + totalBufferSize));
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
    for (auto &entry : valueOffsets) {
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
  void runOnFunction() override { BufferPacking packing(getFunction()); }
};

} // end namespace

std::unique_ptr<FunctionPass> createBufferPackingPass() {
  return std::make_unique<BufferPackingPass>();
}

} // end namespace mlir
