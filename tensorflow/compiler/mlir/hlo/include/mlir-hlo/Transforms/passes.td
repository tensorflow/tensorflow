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

def BufferReuse : Pass<"buffer-reuse", "func::FuncOp"> {
  let summary = "Reuses already allocated buffers to save allocation "
                "operations if it is provably safe.";
  let description = [{
    This pass tries to reuse already allocated buffers if they have the same
    size, are in the right appearance order and have no interference among
    themselves.
  }];
  let constructor = "createBufferReusePass()";
}

def SymbolicShapeOptimization : Pass<"symbolic-shape-optimization", "func::FuncOp"> {
  let summary = "Analyzes shapes and performs shape-related optimizations";
  let constructor = "createSymbolicShapeOptimizationPass()";
}

def CopyRemoval : Pass<"copy-removal", "func::FuncOp"> {
  let summary = "Removes redundant operations that implement a "
                "CopyOpInterface, if the intersection of the useranges from"
                "copy source and target only contains the CopyOp.";
  let constructor = "createCopyRemovalPass()";
}

def BufferPacking : Pass<"buffer-packing", "func::FuncOp"> {
  let summary = "Pass to pack allocated buffer to reduce memory consumption.";
  let description = [{The pass tries to pack smaller buffers into larger buffers.
  To do this, it sorts all allocated buffers by multiple criteria depends on the
  selected window-size.
  After this sorting, the buffers are checked whether subsequent buffers can be
  packed into them.}];
  let dependentDialects = ["func::FuncDialect","memref::MemRefDialect",
    "arith::ArithmeticDialect"];
  let constructor = "createBufferPackingPass()";
  let options = [
   Option<"window_size_", "window-size", "unsigned",
           /*default=*/"5", "The window size blurs the start position of an"
           "allocated buffer. Buffers allocated in the same sliding window area"
           "are treated equally in terms of starting position, withing the"
           "sliding window area they are sorted by memory size."
           "A window size of zero sorts the buffers only by memory size.">,
  ];
}

def MemoryCount : Pass<"memory-count", "func::FuncOp"> {
  let summary = "Test pass to count the allocated memory of a module.";
  let description = [{A test pass that prints the size of allocated memory of a
  module.}];
  let constructor = "createMemoryCountPass()";
}

def TestUserange : Pass<"test-print-userange", "func::FuncOp"> {
  let summary = "Test pass for checking userange intervals.";
  let constructor = "createTestUserangePass()";
}

def TestShapeComponentAnalysis : Pass<"test-print-shape-components",
                                      "func::FuncOp"> {
  let summary = "Test pass for analyzing shape components.";
  let constructor = "createTestShapeComponentAnalysisPass()";
}

def LowerIndexCastPass
    : Pass<"lower-index-cast", "mlir::func::FuncOp"> {
  let summary = "Lower index cast on tensors to tensor dialect";
  let constructor = "CreateLowerIndexCastPass()";
}

def ShapeSimplification
    : Pass<"shape-simplification", "mlir::func::FuncOp"> {
  let summary = "Simplify shape ops";
  let constructor = "CreateShapeSimplification()";
}

def OneShotBufferize : Pass<"hlo-one-shot-bufferize", "ModuleOp"> {
  let summary = "One shot bufferization pass.";
  let constructor = "hlo::CreateOneShotBufferizePass()";
}

def ComputeOpAndFuncBufferizePass : Pass<"computeop-and-func-bufferize", "ModuleOp"> {
  let summary = "Pass to transform compute operations (hlo and linalg) on "
                "values to buffer based ones.";
  let constructor = "CreateComputeOpAndFuncBufferizePass()";
}

def FinalBufferizePass : Pass<"final-bufferize", "ModuleOp"> {
  let summary = "Pass to transform late operations on values to buffer based "
                "ones.";
  let constructor = "CreateFinalBufferizePass()";
  let options = [
      Option<"alignment_", "alignment", "uint64_t",
             /*default=*/"64", "Memory alignment">,
  ];
}

#endif // TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES
