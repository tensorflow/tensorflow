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

def CollapseParallelLoopsTo1DPass : Pass<"collapse-parallel-loops-to-1d", "func::FuncOp"> {
  let summary = "Collapses multidimensional loops.";
  let description = [{ The pass converts a multidimensional `scf.parallel` loop
  into a 1D `scf.parallel` and index computation from a 1D to multidimensional
  index. }];
  let constructor = "createCollapseParallelLoopsTo1DPass()";
}

def TileLoopsPass : Pass<"tile-loops", "func::FuncOp"> {
  let summary = "Tiles parallel loops.";
  let description = [{ The pass converts an `scf.parallel` loop into a nested,
  "tiled", `scf.parallel` loop with 2 to 3 levels of nesting. The 3rd level of
  nesting represents operation unrolling within a tile and is only applied on
  simple memory access patterns (ones resulting from same shape, scalar, and/or
  constant operands).}];
  let constructor = "createTileLoopsPass()";
  let options = [
    ListOption<"tile_sizes_", "tile-sizes", "int64_t", "The size of the tile "
               "in each dimension, expressed as the number of "
               "`unroll_factors_` in that dimension.", "llvm::cl::ZeroOrMore">,
    ListOption<"unroll_factors_", "unroll-factors", "int64_t", "The unroll "
               "factor in each dimension, expressed as the number of elements "
               "in that dimension.", "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = ["AffineDialect"];
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
  let constructor = "createLowerIndexCastPass()";
}

def ShapeSimplification
    : Pass<"shape-simplification", "mlir::func::FuncOp"> {
  let summary = "Simplify shape ops";
  let constructor = "createShapeSimplification()";
}

def OneShotBufferize : Pass<"hlo-one-shot-bufferize", "ModuleOp"> {
  let summary = "One shot bufferization pass.";
  let constructor = "hlo::createOneShotBufferizePass()";
}

def ComputeOpAndFuncBufferizePass : Pass<"computeop-and-func-bufferize", "ModuleOp"> {
  let summary = "Pass to transform compute operations (hlo and linalg) on "
                "values to buffer based ones.";
  let constructor = "createComputeOpAndFuncBufferizePass()";
}

def FinalBufferizePass : Pass<"final-bufferize", "ModuleOp"> {
  let summary = "Pass to transform late operations on values to buffer based "
                "ones.";
  let constructor = "createFinalBufferizePass()";
  let options = [
      Option<"alignment_", "alignment", "uint64_t",
             /*default=*/"64", "Memory alignment">,
  ];
}

def PropagateStaticShapesToKernelPass : Pass<"propagate-static-shapes", "ModuleOp"> {
  let summary = "Pass to rewrite statically shaped kernel arguments to a pointer.";
  let constructor = "createPropagateStaticShapesToKernelPass()";
  let options = [
      Option<"ptr_type_opt", "convert_pointer_args", "std::string",
             /*default=*/"", "Pointer type to convert pointer arguments to">,
  ];
}

def GenericHostToLLVMPass : Pass<"generic-host-to-llvm", "ModuleOp"> {
  let summary = "Pass to lower common dialects resulting from HLO to LLVM.";
  let constructor = "hlo::createGenericHostToLLVMPass()";
  let options = [];
}

#endif // TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES
