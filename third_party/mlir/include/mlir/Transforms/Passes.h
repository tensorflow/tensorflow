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
class FuncOp;
class ModuleOp;
class Pass;
template <typename T> class OpPassBase;

/// Creates an instance of the Canonicalizer pass.
std::unique_ptr<Pass> createCanonicalizerPass();

/// Creates a pass to perform common sub expression elimination.
std::unique_ptr<Pass> createCSEPass();

/// Creates a pass to vectorize loops, operations and data types using a
/// target-independent, n-D super-vector abstraction.
std::unique_ptr<OpPassBase<FuncOp>>
createVectorizePass(llvm::ArrayRef<int64_t> virtualVectorSize);

/// Creates a pass to allow independent testing of vectorizer functionality with
/// FileCheck.
std::unique_ptr<OpPassBase<FuncOp>> createVectorizerTestPass();

/// Creates a pass to lower super-vectors to target-dependent HW vectors.
std::unique_ptr<OpPassBase<FuncOp>>
createMaterializeVectorsPass(llvm::ArrayRef<int64_t> vectorSize);

/// Creates a loop unrolling pass with the provided parameters.
/// 'getUnrollFactor' is a function callback for clients to supply a function
/// that computes an unroll factor - the callback takes precedence over unroll
/// factors supplied through other means. If -1 is passed as the unrollFactor
/// and no callback is provided, anything passed from the command-line (if at
/// all) or the default unroll factor is used (LoopUnroll:kDefaultUnrollFactor).
std::unique_ptr<OpPassBase<FuncOp>> createLoopUnrollPass(
    int unrollFactor = -1, int unrollFull = -1,
    const std::function<unsigned(AffineForOp)> &getUnrollFactor = nullptr);

/// Creates a loop unroll jam pass to unroll jam by the specified factor. A
/// factor of -1 lets the pass use the default factor or the one on the command
/// line if provided.
std::unique_ptr<OpPassBase<FuncOp>>
createLoopUnrollAndJamPass(int unrollJamFactor = -1);

/// Creates a simplification pass for affine structures (maps and sets). In
/// addition, this pass also normalizes memrefs to have the trivial (identity)
/// layout map.
std::unique_ptr<OpPassBase<FuncOp>> createSimplifyAffineStructuresPass();

/// Creates a loop fusion pass which fuses loops. Buffers of size less than or
/// equal to `localBufSizeThreshold` are promoted to memory space
/// `fastMemorySpace'.
std::unique_ptr<OpPassBase<FuncOp>>
createLoopFusionPass(unsigned fastMemorySpace = 0,
                     uint64_t localBufSizeThreshold = 0,
                     bool maximalFusion = false);

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of affine loop.
std::unique_ptr<OpPassBase<FuncOp>> createAffineLoopInvariantCodeMotionPass();

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
std::unique_ptr<OpPassBase<FuncOp>> createPipelineDataTransferPass();

/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
std::unique_ptr<OpPassBase<FuncOp>> createLowerAffinePass();

/// Creates a pass to perform tiling on loop nests.
std::unique_ptr<OpPassBase<FuncOp>>
createLoopTilingPass(uint64_t cacheSizeBytes);

/// Creates a pass that performs parametric tiling so that the outermost loops
/// have the given fixed number of iterations.  Assumes outermost loop nests
/// are permutable.
std::unique_ptr<OpPassBase<FuncOp>>
createSimpleParametricTilingPass(ArrayRef<int64_t> outerLoopSizes);

/// Creates a pass that transforms perfectly nested loops with independent
/// bounds into a single loop.
std::unique_ptr<OpPassBase<FuncOp>> createLoopCoalescingPass();

/// Performs packing (or explicit copying) of accessed memref regions into
/// buffers in the specified faster memory space through either pointwise copies
/// or DMA operations.
std::unique_ptr<OpPassBase<FuncOp>> createAffineDataCopyGenerationPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace,
    unsigned tagMemorySpace = 0, int minDmaTransferSize = 1024,
    uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max());

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OpPassBase<FuncOp>> createMemRefDataFlowOptPass();

/// Creates a pass to strip debug information from a function.
std::unique_ptr<OpPassBase<FuncOp>> createStripDebugInfoPass();

/// Creates a pass which tests loop fusion utilities.
std::unique_ptr<OpPassBase<FuncOp>> createTestLoopFusionPass();

/// Creates a pass which inlines calls and callable operations as defined by the
/// CallGraph.
std::unique_ptr<Pass> createInlinerPass();
} // end namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
