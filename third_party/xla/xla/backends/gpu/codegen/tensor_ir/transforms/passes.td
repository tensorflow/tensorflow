/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def LegalizeStablehloToTensorIrPass
    : Pass<"legalize-stablehlo-to-tensor-ir", "mlir::ModuleOp"> {
  let summary = "Legalize StableHLO ops to the nv_tensor_ir dialect.";
  let description = [{
    Converts func.func functions to nv_tensor_ir.graph operations and legalizes
    contained StableHLO operations to nv_tensor_ir operations.
  }];
  let dependentDialects = [
    "::mlir::nv_tensor_ir::TensorIRDialect",
    "::mlir::stablehlo::StablehloDialect",
    // The HLO importer emits mhlo ops (erf, bitcast) for constructs StableHLO
    // has no equivalent for, so the pass matches on those too.
    "::mlir::mhlo::MhloDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_TD_
