/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def LhloFuseLinalgPass : FunctionPass<"lhlo-fuse-linalg"> {
  let summary = "Greedily fuse linalg ops obtained after LHLO lowering.";
  let constructor = "createLhloFuseLinalgPass()";
  let options = [
    Option<"use_parallel_loops_", "use-parallel-loops", "bool",
           /*default=*/"false", "Tiles GenericOp consumer to parallel loops before linalg fusion">,
    ListOption<"tile_sizes_", "tile-sizes", "unsigned",
           "Faster memory space number to promote fusion buffers to",
           "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def LhloLegalizeToAffinePass : FunctionPass<"lhlo-legalize-to-affine"> {
  let summary = "Legalize from LHLO dialect to affine dialect.";
  let constructor = "createLhloLegalizeToAffinePass()";
}


def LhloLegalizeToGpuPass : FunctionPass<"lhlo-legalize-to-gpu"> {
  let summary = "Legalize from LHLO dialect to GPU dialect.";
  let constructor = "createLegalizeToGpuPass()";
}


def LhloLegalizeToParallelLoopsPass : FunctionPass<"lhlo-legalize-to-parallel-loops"> {
  let summary = "Legalize from LHLO dialect to parallel loops.";
  let constructor = "createLegalizeLhloToParallelLoopsPass()";
}

def LegalizeToTensorOpPass : FunctionPass<"lhlo-legalize-to-tensor-op"> {
  let summary = "Legalize bufferization.to_tensor ops inserted during mhlo to lmhlo conversion.";
  let constructor = "createLegalizeToTensorOpPass()";
}

def LhloFusionPass : FunctionPass<"lhlo-fusion"> {
  let summary = "Fuse lmhlo ops to kLoop/kInput fusion patterns.";
  let constructor = "createLhloFusionPass()";
  let options = [
    Option<"max_num_arguments_per_kernel_", "max-num-arguments-per-kernel", "int",
           /*default=*/"64", "Maximum allowed number of arguments per fused kernel.">,
  ];
}

def LhloFusionInlinerPass : FunctionPass<"lhlo-fusion-inliner"> {
  let summary = "Inline the contents of its body to the parent region "
    "after its body has been lowered";
  let constructor = "createLhloFusionInlinerPass()";
}

def LhloLegalizeRootsToParallelLoopsPass : FunctionPass<"lhlo-legalize-roots-to-parallel-loops"> {
  let summary = "Legalize the roots of lmhlo.fusion to parallel loops.";
  let constructor = "createLhloLegalizeRootsToParallelLoopsPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}

def InputInlineFusionPass : FunctionPass<"input-inline-fusion"> {
  let summary = "Fuse the rest of the lmhlo nodes into the parallel loops after the roots "
    "has been expanded into loops in LhloLegalizeRootsToParallelLoops";
  let constructor = "createInputInlineFusionPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect"
  ];
}
