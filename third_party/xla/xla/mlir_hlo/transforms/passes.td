/* Copyright 2021 The OpenXLA Authors.

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

#ifndef MLIR_HLO_TRANSFORMS_PASSES
#define MLIR_HLO_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CollapseParallelLoopsTo1DPass : Pass<"collapse-parallel-loops-to-1d"> {
  let summary = "Collapses multidimensional loops.";
  let description = [{ The pass converts a multidimensional `scf.parallel` loop
  into a 1D `scf.parallel` and index computation from a 1D to multidimensional
  index. }];
  let constructor = "createCollapseParallelLoopsTo1DPass()";
}

def DetensorizeScfOpsPass : Pass<"detensorize-scf-ops", "func::FuncOp"> {
  let summary = "Detensorize arguments of SCF ops where possible.";
  let constructor = "createDetensorizeScfOpsPass()";
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
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
  let dependentDialects = ["affine::AffineDialect"];
}

def LowerIndexCastPass
    : Pass<"lower-index-cast", "mlir::func::FuncOp"> {
  let summary = "Lower index cast on tensors to tensor dialect";
  let dependentDialects = [
    "tensor::TensorDialect"
  ];
  let constructor = "createLowerIndexCastPass()";
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

def GenericHostToLLVMPass : Pass<"generic-host-to-llvm", "ModuleOp"> {
  let summary = "Pass to lower common dialects resulting from HLO to LLVM.";
  let constructor = "hlo::createGenericHostToLLVMPass()";
  let options = [
      Option<"enableAvx2", "enable-avx2", "bool",
             /*default=*/"false", "Whether to enable avx2">,
  ];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::complex::ComplexDialect",
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::x86vector::X86VectorDialect",
  ];
}

def UnbufferizePass : Pass<"unbufferize", "mlir::func::FuncOp"> {
  let summary = "Unbufferize partially bufferized functions.";
  let description = [{
    Removes bufferization.to_tensor and bufferization.materialize_in_destination
    ops that are the result of XLA bufferizing during HLO to MHLO
    transformation.
  }];
  let constructor = "hlo::createUnbufferizePass()";
}

def AllocToArgPass : Pass<"alloc-to-arg", "mlir::func::FuncOp"> {
  let summary = "Hoist memref allocations to function arguments.";
  let description = [{
    Replaces memref.alloc uses with a new argument of the parent function.
  }];
  let constructor = "hlo::createAllocToArgPass()";
}

def NaiveCopyRemovalPass : Pass<"naive-copy-removal", "mlir::func::FuncOp"> {
  let summary = "Pass to remove redundant `memref.copy` ops.";
  let constructor = "createNaiveCopyRemovalPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def VectorizeCopyPass : Pass<"vectorize-copy", "mlir::func::FuncOp"> {
  let summary = "Pass to vectorize `memref.copy`.";
  let constructor = "createVectorizeCopyPass()";
  let dependentDialects = ["scf::SCFDialect", "vector::VectorDialect"];
}

#endif // TENSORFLOW_COMPILER_MLIR_HLO_TRANSFORMS_PASSES
