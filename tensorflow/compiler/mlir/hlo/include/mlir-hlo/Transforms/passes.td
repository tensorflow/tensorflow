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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES
#define TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def BufferReuse : FunctionPass<"buffer-reuse"> {
  let summary = "Reuses already allocated buffers to save allocation "
                "operations if it is provably safe.";
  let description = [{
    This pass tries to reuse already allocated buffers if they have the same
    size, are in the right appearance order and have no interference among
    themselves.
  }];
  let constructor = "createBufferReusePass()";
}

def ReshapeSimplifier : FunctionPass<"reshape-simplifier"> {
  let summary = "";
  let description = [{
    This pass tries to simplify dynamic reshapes.
  }];
  let constructor = "createReshapeSimplifierPass()";
}

def CopyRemoval : FunctionPass<"copy-removal"> {
  let summary = "Removes redundant operations that implement a "
                "CopyOpInterface, if the intersection of the useranges from"
                "copy source and target only contains the CopyOp.";
  let constructor = "createCopyRemovalPass()";
}

def TestUserange : FunctionPass<"test-print-userange"> {
  let summary = "Test pass for checking userange intervals.";
  let constructor = "createTestUserangePass()";
}

def TestShapeComponentAnalysis : FunctionPass<"test-print-shape-components"> {
  let summary = "Test pass for analyzing shape components.";
  let constructor = "createTestShapeComponentAnalysisPass()";
}

#endif // TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES
