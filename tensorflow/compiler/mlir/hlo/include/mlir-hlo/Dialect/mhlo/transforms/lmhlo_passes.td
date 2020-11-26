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

def LhloLegalizeToLinalgPass : Pass<"lhlo-legalize-to-linalg", "FuncOp"> {
  let summary = "Legalize from LHLO dialect to Linalg dialect.";
  let constructor = "createLegalizeLhloToLinalgPass()";
}


def LhloFuseLinalgPass : Pass<"lhlo-fuse-linalg", "FuncOp"> {
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


def LhloLegalizeToAffinePass : Pass<"lhlo-legalize-to-affine", "FuncOp"> {
  let summary = "Legalize from LHLO dialect to affine dialect.";
  let constructor = "createLhloLegalizeToAffinePass()";
}


def LhloLegalizeToGpuPass : Pass<"lhlo-legalize-to-gpu", "FuncOp"> {
  let summary = "Legalize from LHLO dialect to GPU dialect.";
  let constructor = "createLegalizeToGpuPass()";
}


def LhloLegalizeToParallelLoopsPass : Pass<"lhlo-legalize-to-parallel-loops", "FuncOp"> {
  let summary = "Legalize from LHLO dialect to parallel loops.";
  let constructor = "createLegalizeLhloToParallelLoopsPass()";
}

