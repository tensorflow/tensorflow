//===- LoopsToGPU.cpp - Convert an affine loop nest to a GPU kernel -------===//
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
// This implements a straightforward conversion of an loop nest into a GPU
// kernel.  The caller is expected to guarantee that the conversion is correct
// or to further transform the kernel to ensure correctness.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LowerAffine.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loops-to-gpu"

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

// Get the lower bound-related operands of a loop operation.
static Operation::operand_range getLowerBoundOperands(AffineForOp forOp) {
  return forOp.getLowerBoundOperands();
}
static SmallVector<Value *, 1> getLowerBoundOperands(linalg::ForOp forOp) {
  SmallVector<Value *, 1> bounds(1, forOp.getLowerBound());
  return bounds;
}

// Get the upper bound-related operands of a loop operation.
static Operation::operand_range getUpperBoundOperands(AffineForOp forOp) {
  return forOp.getUpperBoundOperands();
}
static SmallVector<Value *, 1> getUpperBoundOperands(linalg::ForOp forOp) {
  SmallVector<Value *, 1> bounds(1, forOp.getUpperBound());
  return bounds;
}

// Get a Value that corresponds to the loop step.  If the step is an attribute,
// materialize a corresponding constant using builder.
static Value *getOrCreateStep(AffineForOp forOp, OpBuilder &builder) {
  return builder.create<ConstantIndexOp>(forOp.getLoc(), forOp.getStep());
}
static Value *getOrCreateStep(linalg::ForOp forOp, OpBuilder &) {
  return forOp.getStep();
}

// Get a Value for the loop lower bound.  If the value requires computation,
// materialize the instructions using builder.
static Value *getOrEmitLowerBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineLowerBound(forOp, builder);
}
static Value *getOrEmitLowerBound(linalg::ForOp forOp, OpBuilder &) {
  return forOp.getLowerBound();
}

// Get a Value for the loop upper bound.  If the value requires computation,
// materialize the instructions using builder.
static Value *getOrEmitUpperBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineUpperBound(forOp, builder);
}
static Value *getOrEmitUpperBound(linalg::ForOp forOp, OpBuilder &) {
  return forOp.getUpperBound();
}

// Check the structure of the loop nest:
//   - there are enough loops to map to numBlockDims + numThreadDims;
//   - the loops are perfectly nested;
//   - the loop bounds can be computed above the outermost loop.
// This roughly corresponds to the "matcher" part of the pattern-based
// rewriting infrastructure.
template <typename OpTy>
LogicalResult checkLoopNestMappable(OpTy forOp, unsigned numBlockDims,
                                    unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  OpBuilder builder(forOp.getOperation());
  if (numBlockDims > 3) {
    return emitError(builder.getUnknownLoc(),
                     "cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return emitError(builder.getUnknownLoc(),
                     "cannot map to more than 3 thread dimensions");
  }

  OpTy currentLoop = forOp;
  Region &limit = forOp.getRegion();
  for (unsigned i = 0, e = numBlockDims + numThreadDims; i < e; ++i) {
    Operation *nested = &currentLoop.getBody()->front();
    if (!areValuesDefinedAbove(getLowerBoundOperands(currentLoop), limit) ||
        !areValuesDefinedAbove(getUpperBoundOperands(currentLoop), limit))
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

    if (!(currentLoop = dyn_cast<OpTy>(nested)))
      return nested->emitError("expected a nested loop");
  }

  return success();
}

namespace {
// Helper structure that holds common state of the loop to GPU kernel
// conversion.
struct LoopToGpuConverter {
  template <typename OpTy>
  Optional<OpTy> collectBounds(OpTy forOp, unsigned numLoops);

  template <typename OpTy>
  void createLaunch(OpTy rootForOp, OpTy innermostForOp, unsigned numBlockDims,
                    unsigned numThreadDims);

  // Ranges of the loops mapped to blocks or threads.
  SmallVector<Value *, 6> dims;
  // Lower bounds of the loops mapped to blocks or threads.
  SmallVector<Value *, 6> lbs;
  // Induction variables of the loops mapped to blocks or threads.
  SmallVector<Value *, 6> ivs;
  // Steps of the loops mapped to blocks or threads.
  SmallVector<Value *, 6> steps;
};
} // namespace

// Return true if the value is obviously a constant "one".
static bool isConstantOne(Value *value) {
  if (auto def = dyn_cast_or_null<ConstantIndexOp>(value->getDefiningOp()))
    return def.getValue() == 1;
  return false;
}

// Collect ranges, bounds, steps and induction variables in preparation for
// mapping a loop nest of depth "numLoops" rooted at "forOp" to a GPU kernel.
// This may fail if the IR for computing loop bounds cannot be constructed, for
// example if an affine loop uses semi-affine maps. Return the last loop to be
// mapped on success, llvm::None on failure.
template <typename OpTy>
Optional<OpTy> LoopToGpuConverter::collectBounds(OpTy forOp,
                                                 unsigned numLoops) {
  OpBuilder builder(forOp.getOperation());
  dims.reserve(numLoops);
  lbs.reserve(numLoops);
  ivs.reserve(numLoops);
  steps.reserve(numLoops);
  OpTy currentLoop = forOp;
  for (unsigned i = 0; i < numLoops; ++i) {
    Value *lowerBound = getOrEmitLowerBound(currentLoop, builder);
    Value *upperBound = getOrEmitUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound) {
      return llvm::None;
    }

    Value *range =
        builder.create<SubIOp>(currentLoop.getLoc(), upperBound, lowerBound);
    Value *step = getOrCreateStep(currentLoop, builder);
    if (!isConstantOne(step))
      range = builder.create<DivISOp>(currentLoop.getLoc(), range, step);
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != numLoops - 1)
      currentLoop = cast<OpTy>(&currentLoop.getBody()->front());
  }
  return currentLoop;
}

// Replace the rooted at "rootForOp" with a GPU launch operation.  This expects
// "innermostForOp" to point to the last loop to be transformed to the kernel,
// and to have (numBlockDims + numThreadDims) perfectly nested loops between
// "rootForOp" and "innermostForOp".
template <typename OpTy>
void LoopToGpuConverter::createLaunch(OpTy rootForOp, OpTy innermostForOp,
                                      unsigned numBlockDims,
                                      unsigned numThreadDims) {
  OpBuilder builder(rootForOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value *constOne = (numBlockDims < 3 || numThreadDims < 3)
                        ? builder.create<ConstantIndexOp>(rootForOp.getLoc(), 1)
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
  getUsedValuesDefinedAbove(innermostForOp.getRegion(), rootForOp.getRegion(),
                            valuesToForwardSet);
  auto valuesToForward = valuesToForwardSet.takeVector();
  auto originallyForwardedValues = valuesToForward.size();
  valuesToForward.insert(valuesToForward.end(), lbs.begin(), lbs.end());
  valuesToForward.insert(valuesToForward.end(), steps.begin(), steps.end());
  auto launchOp = builder.create<gpu::LaunchOp>(
      rootForOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX,
      blockSizeY, blockSizeZ, valuesToForward);
  valuesToForward.resize(originallyForwardedValues);

  // Replace the loop terminator (loops contain only a single block) with the
  // gpu return and move the operations from the loop body block to the gpu
  // launch body block.  Do not move the entire block because of the difference
  // in block arguments.
  Operation &terminator = innermostForOp.getBody()->back();
  Location terminatorLoc = terminator.getLoc();
  terminator.erase();
  builder.setInsertionPointToEnd(innermostForOp.getBody());
  builder.create<gpu::Return>(terminatorLoc);
  launchOp.getBody().front().getOperations().splice(
      launchOp.getBody().front().begin(),
      innermostForOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPointToStart(&launchOp.getBody().front());
  auto lbArgumentIt = std::next(launchOp.getKernelArguments().begin(),
                                originallyForwardedValues);
  auto stepArgumentIt = std::next(lbArgumentIt, lbs.size());
  for (auto en : llvm::enumerate(ivs)) {
    Value *id =
        en.index() < numBlockDims
            ? getDim3Value(launchOp.getBlockIds(), en.index())
            : getDim3Value(launchOp.getThreadIds(), en.index() - numBlockDims);
    Value *step = steps[en.index()];
    if (!isConstantOne(step))
      id = builder.create<MulIOp>(rootForOp.getLoc(), step, id);

    Value *ivReplacement =
        builder.create<AddIOp>(rootForOp.getLoc(), *lbArgumentIt, id);
    en.value()->replaceAllUsesWith(ivReplacement);
    replaceAllUsesInRegionWith(steps[en.index()], *stepArgumentIt,
                               launchOp.getBody());
    std::advance(lbArgumentIt, 1);
    std::advance(stepArgumentIt, 1);
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
  rootForOp.erase();
}

// Generic loop to GPU kernel conversion function.
template <typename OpTy>
static LogicalResult convertLoopNestToGPULaunch(OpTy forOp,
                                                unsigned numBlockDims,
                                                unsigned numThreadDims) {
  if (failed(checkLoopNestMappable(forOp, numBlockDims, numThreadDims)))
    return failure();

  LoopToGpuConverter converter;
  auto maybeInnerLoop =
      converter.collectBounds(forOp, numBlockDims + numThreadDims);
  if (!maybeInnerLoop)
    return failure();
  converter.createLaunch(forOp, *maybeInnerLoop, numBlockDims, numThreadDims);

  return success();
}

LogicalResult mlir::convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                                     unsigned numBlockDims,
                                                     unsigned numThreadDims) {
  return convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims);
}

LogicalResult mlir::convertLinalgLoopNestToGPULaunch(linalg::ForOp forOp,
                                                     unsigned numBlockDims,
                                                     unsigned numThreadDims) {
  return convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims);
}
