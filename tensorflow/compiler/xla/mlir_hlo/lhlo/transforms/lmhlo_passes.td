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

def LhloLegalizeToAffinePass : Pass<"lhlo-legalize-to-affine", "func::FuncOp"> {
  let summary = "Legalize from LHLO dialect to affine dialect.";
  let constructor = "createLhloLegalizeToAffinePass()";
}


def LhloLegalizeToGpuPass : Pass<"lhlo-legalize-to-gpu", "func::FuncOp"> {
  let summary = "Legalize from LHLO dialect to GPU dialect.";
  let constructor = "createLegalizeToGpuPass()";
}


def LhloLegalizeToParallelLoopsPass : Pass<"lhlo-legalize-to-parallel-loops", "func::FuncOp"> {
  let summary = "Legalize from LHLO dialect to parallel loops.";
  let constructor = "createLegalizeLhloToParallelLoopsPass()";
}

def LegalizeToTensorOpPass : Pass<"lhlo-legalize-to-tensor-op", "func::FuncOp"> {
  let summary = "Legalize bufferization.to_tensor ops inserted during mhlo to lmhlo conversion.";
  let constructor = "createLegalizeToTensorOpPass()";
}
