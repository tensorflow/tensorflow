//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <limits>

namespace mlir {

class AffineForOp;
class FunctionPassBase;
class ModulePassBase;

/// Creates a constant folding pass. Note that this pass solely provides simple
/// top-down constant folding functionality; it is intended to be used for
/// testing purpose. Use Canonicalizer pass, which exploits more simplification
/// opportunties exposed by constant folding, for the general cases.
FunctionPassBase *createTestConstantFoldPass();

/// Creates an instance of the Canonicalizer pass.
FunctionPassBase *createCanonicalizerPass();

/// Creates a pass to perform common sub expression elimination.
FunctionPassBase *createCSEPass();

/// Creates a pass to vectorize loops, operations and data types using a
/// target-independent, n-D super-vector abstraction.
FunctionPassBase *
createVectorizePass(llvm::ArrayRef<int64_t> virtualVectorSize);

/// Creates a pass to allow independent testing of vectorizer functionality with
/// FileCheck.
FunctionPassBase *createVectorizerTestPass();

/// Creates a pass to lower super-vectors to target-dependent HW vectors.
FunctionPassBase *
createMaterializeVectorsPass(llvm::ArrayRef<int64_t> vectorSize);

/// Creates a loop unrolling pass with the provided parameters.
/// 'getUnrollFactor' is a function callback for clients to supply a function
/// that computes an unroll factor - the callback takes precedence over unroll
/// factors supplied through other means. If -1 is passed as the unrollFactor
/// and no callback is provided, anything passed from the command-line (if at
/// all) or the default unroll factor is used (LoopUnroll:kDefaultUnrollFactor).
FunctionPassBase *createLoopUnrollPass(
    int unrollFactor = -1, int unrollFull = -1,
    const std::function<unsigned(AffineForOp)> &getUnrollFactor = nullptr);

/// Creates a loop unroll jam pass to unroll jam by the specified factor. A
/// factor of -1 lets the pass use the default factor or the one on the command
/// line if provided.
FunctionPassBase *createLoopUnrollAndJamPass(int unrollJamFactor = -1);

/// Creates an simplification pass for affine structures.
FunctionPassBase *createSimplifyAffineStructuresPass();

/// Creates a loop fusion pass which fuses loops. Buffers of size less than or
/// equal to `localBufSizeThreshold` are promoted to memory space
/// `fastMemorySpace'.
FunctionPassBase *createLoopFusionPass(unsigned fastMemorySpace = 0,
                                       uint64_t localBufSizeThreshold = 0,
                                       bool maximalFusion = false);

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
FunctionPassBase *createLoopInvariantCodeMotionPass();

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
FunctionPassBase *createPipelineDataTransferPass();

/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
FunctionPassBase *createLowerAffinePass();

/// Creates a pass to perform tiling on loop nests.
FunctionPassBase *createLoopTilingPass(uint64_t cacheSizeBytes);

/// Creates a pass that performs parametric tiling so that the outermost loops
/// have the given fixed number of iterations.  Assumes outermost loop nests
/// are permutable.
FunctionPassBase *
createSimpleParametricTilingPass(ArrayRef<int64_t> outerLoopSizes);

/// Promotes all accessed memref regions to the specified faster memory space
/// while generating DMAs to move data.
FunctionPassBase *createDmaGenerationPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace,
    unsigned tagMemorySpace = 0, int minDmaTransferSize = 1024,
    uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max());

/// Creates a pass to lower VectorTransferReadOp and VectorTransferWriteOp.
FunctionPassBase *createLowerVectorTransfersPass();

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
FunctionPassBase *createMemRefDataFlowOptPass();

/// Creates a pass to strip debug information from a function.
FunctionPassBase *createStripDebugInfoPass();

/// Creates a pass which tests loop fusion utilities.
FunctionPassBase *createTestLoopFusionPass();

} // end namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
