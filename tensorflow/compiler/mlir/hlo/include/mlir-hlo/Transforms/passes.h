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

#ifndef MLIR_HLO_TRANSFORMS_PASSES_H
#define MLIR_HLO_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates a pass that reuses buffers which are already allocated.
std::unique_ptr<OperationPass<FuncOp>> createBufferReusePass();

/// Creates a pass that tries to simplify dynamic reshapes.
std::unique_ptr<OperationPass<FuncOp>> createReshapeSimplifierPass();

/// Creates a pass that merges smaller buffer into bigger buffer to optimize
/// memory consumption.
std::unique_ptr<OperationPass<FuncOp>> createBufferPackingPass(
    unsigned window_size = 5);

/// Creates a pass that tests the useranges of the UserangeAnalysis.
std::unique_ptr<OperationPass<FuncOp>> createTestUserangePass();

/// Creates a pass that prints the analysis results of ShapeComponentsAnalysis.
std::unique_ptr<OperationPass<FuncOp>> createTestShapeComponentAnalysisPass();

/// Creates a pass that removes redundant operations that implement a
/// CopyOpInterface.
std::unique_ptr<OperationPass<FuncOp>> createCopyRemovalPass();

/// Creates a pass that computes the allocated memory.
std::unique_ptr<OperationPass<FuncOp>> createMemoryCountPass();

}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_PASSES_H
