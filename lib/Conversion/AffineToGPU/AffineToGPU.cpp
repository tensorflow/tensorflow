//===- AffineToGPU.cpp - Convert an affine loop nest to a GPU kernel ------===//
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
// This implements a straightforward conversion of an affine loop nest into a
// GPU kernel.  The caller is expected to guarantee that the conversion is
// correct or to further transform the kernel to ensure correctness.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToGPU/AffineToGPU.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LowerAffine.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-to-gpu"

using namespace mlir;

// Extract an indexed value from KernelDim3.
static Value *getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

LogicalResult mlir::convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                                     unsigned numBlockDims,
                                                     unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  OpBuilder builder(forOp.getOperation());

  if (numBlockDims > 3) {
    forOp.getContext()->emitError(builder.getUnknownLoc(),
                                  "cannot map to more than 3 block dimensions");
    return failure();
  }
  if (numThreadDims > 3) {
    forOp.getContext()->emitError(
        builder.getUnknownLoc(), "cannot map to more than 3 thread dimensions");
    return failure();
  }

  // Check the structure of the loop nest:
  //   - there is enough loops to map to numBlockDims + numThreadDims;
  //   - the loops are perfectly nested;
  //   - the loop bounds can be computed above the outermost loop.
  // This roughly corresponds to the "matcher" part of the pattern-based
  // rewriting infrastructure.
  AffineForOp currentLoop = forOp;
  Region &limit = forOp.getRegion();
  for (unsigned i = 0, e = numBlockDims + numThreadDims; i < e; ++i) {
    Operation *nested = &currentLoop.getBody()->front();
    if (currentLoop.getStep() <= 0)
      return currentLoop.emitError("only positive loop steps are supported");
    if (!areValuesDefinedAbove(currentLoop.getLowerBoundOperands(), limit) ||
        !areValuesDefinedAbove(currentLoop.getUpperBoundOperands(), limit))
      return currentLoop.emitError(
          "loops with bounds depending on other mapped loops "
          "are not supported");

    // The innermost loop can have an arbitrary body, skip the perfect nesting
    // check for it.
    if (i == e - 1)
      break;

    auto begin = currentLoop.getBody()->begin(),
         end = currentLoop.getBody()->end();
    if (currentLoop.getBody()->empty() || std::next(begin, 2) != end)
      return currentLoop.emitError(
          "expected perfectly nested loops in the body");

    if (!(currentLoop = dyn_cast<AffineForOp>(nested)))
      return nested->emitError("expected a nested loop");
  }

  // Compute the ranges of the loops and collect lower bounds and induction
  // variables.
  SmallVector<Value *, 6> dims;
  SmallVector<Value *, 6> lbs;
  SmallVector<Value *, 6> ivs;
  SmallVector<int64_t, 6> steps;
  dims.reserve(numBlockDims + numThreadDims);
  lbs.reserve(numBlockDims + numThreadDims);
  ivs.reserve(numBlockDims + numThreadDims);
  steps.reserve(numBlockDims + numThreadDims);
  currentLoop = forOp;
  for (unsigned i = 0, e = numBlockDims + numThreadDims; i < e; ++i) {
    Value *lowerBound = lowerAffineLowerBound(currentLoop, builder);
    Value *upperBound = lowerAffineUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound)
      return failure();

    Value *range =
        builder.create<SubIOp>(currentLoop.getLoc(), upperBound, lowerBound);
    int64_t step = currentLoop.getStep();
    if (step > 1) {
      auto divExpr =
          getAffineSymbolExpr(0, currentLoop.getContext()).floorDiv(step);
      range = expandAffineExpr(builder, currentLoop.getLoc(), divExpr,
                               llvm::None, range);
    }
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != e - 1)
      currentLoop = cast<AffineForOp>(&currentLoop.getBody()->front());
  }
  // At this point, currentLoop points to the innermost loop we are mapping.

  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value *constOne = (numBlockDims < 3 || numThreadDims < 3)
                        ? builder.create<ConstantIndexOp>(forOp.getLoc(), 1)
                        : nullptr;
  Value *gridSizeX = dims[0];
  Value *gridSizeY = numBlockDims > 1 ? dims[1] : constOne;
  Value *gridSizeZ = numBlockDims > 2 ? dims[2] : constOne;
  Value *blockSizeX = dims[numBlockDims];
  Value *blockSizeY = numThreadDims > 1 ? dims[numBlockDims + 1] : constOne;
  Value *blockSizeZ = numThreadDims > 2 ? dims[numBlockDims + 2] : constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.  Pass the values defined outside the outermost loop and used
  // inside the innermost loop and loop lower bounds as kernel data arguments.
  // Still assuming perfect nesting so there are no values other than induction
  // variables that are defined in one loop and used in deeper loops.
  llvm::SetVector<Value *> valuesToForwardSet;
  getUsedValuesDefinedAbove(forOp.getRegion(), forOp.getRegion(),
                            valuesToForwardSet);
  auto valuesToForward = valuesToForwardSet.takeVector();
  auto originallyForwardedValues = valuesToForward.size();
  valuesToForward.insert(valuesToForward.end(), lbs.begin(), lbs.end());
  auto launchOp = builder.create<gpu::LaunchOp>(
      forOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY,
      blockSizeZ, valuesToForward);
  valuesToForward.resize(originallyForwardedValues);

  // Replace the affine terminator (loops contain only a single block) with the
  // gpu return and move the operations from the loop body block to the gpu
  // launch body block.  Do not move the entire block because of the difference
  // in block arguments.
  Operation &terminator = currentLoop.getBody()->back();
  Location terminatorLoc = terminator.getLoc();
  terminator.erase();
  builder.setInsertionPointToEnd(currentLoop.getBody());
  builder.create<gpu::Return>(terminatorLoc);
  launchOp.getBody().front().getOperations().splice(
      launchOp.getBody().front().begin(),
      currentLoop.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPointToStart(&launchOp.getBody().front());
  auto lbArgumentIt = std::next(launchOp.getKernelArguments().begin(),
                                originallyForwardedValues);
  for (auto en : llvm::enumerate(ivs)) {
    Value *id =
        en.index() < numBlockDims
            ? getDim3Value(launchOp.getBlockIds(), en.index())
            : getDim3Value(launchOp.getThreadIds(), en.index() - numBlockDims);
    if (steps[en.index()] > 1) {
      Value *factor =
          builder.create<ConstantIndexOp>(forOp.getLoc(), steps[en.index()]);
      id = builder.create<MulIOp>(forOp.getLoc(), factor, id);
    }
    Value *ivReplacement =
        builder.create<AddIOp>(forOp.getLoc(), *lbArgumentIt, id);
    en.value()->replaceAllUsesWith(ivReplacement);
    std::advance(lbArgumentIt, 1);
  }

  // Remap the values defined outside the body to use kernel arguments instead.
  // The list of kernel arguments also contains the lower bounds for loops at
  // trailing positions, make sure we don't touch those.
  for (const auto &pair :
       llvm::zip_first(valuesToForward, launchOp.getKernelArguments())) {
    Value *from = std::get<0>(pair);
    Value *to = std::get<1>(pair);
    replaceAllUsesInRegionWith(from, to, launchOp.getBody());
  }

  // We are done and can erase the original outermost loop.
  forOp.erase();

  return success();
}
