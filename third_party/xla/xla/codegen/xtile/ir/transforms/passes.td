/* Copyright 2024 The OpenXLA Authors.

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

def VerifyLegalXTileOpsPass : Pass<"xtile-verify-legal-ops", "mlir::ModuleOp"> {
  let summary = "Verify that all ops in the module are legal";

  let description = [{
    This pass verifies that all ops in the module are legal, which is defined as
    the ops that are supported by all of the XTile backends.
  }];

}

def ConvertElementwise0DTensorToScalarPass
    : Pass<"convert-elementwise-0d-tensor-to-scalar"> {
  let summary = "Lowers 0D tensors of elementwise ops to scalars.";

  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
}

def StablehloLowerToXtilePass
    : Pass<"stablehlo-lower-to-xtile", "mlir::ModuleOp"> {
  let summary = "Lowers stablehlo ops to Xtile.";
  let description = [{
    This pass lowers stablehlo ops to Xtile supported ops.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}
