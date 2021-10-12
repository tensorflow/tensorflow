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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_LIB_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_LIB_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {

class FunctionPass;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates a pass that reuses buffers which are already allocated.
std::unique_ptr<FunctionPass> createBufferReusePass();

/// Creates a pass that tries to simplify dynamic reshapes.
std::unique_ptr<FunctionPass> createReshapeSimplifierPass();

/// Creates a pass that tests the useranges of the UserangeAnalysis.
std::unique_ptr<FunctionPass> createTestUserangePass();

/// Creates a pass that prints the analysis results of ShapeComponentsAnalysis.
std::unique_ptr<FunctionPass> createTestShapeComponentAnalysisPass();

/// Creates a pass that removes redundant operations that implement a
/// CopyOpInterface.
std::unique_ptr<FunctionPass> createCopyRemovalPass();

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_LIB_TRANSFORMS_PASSES_H_
