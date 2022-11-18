/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Auxiliary passes for lowering to XLA Cpu runtime.
//===----------------------------------------------------------------------===//

def ConvertLmhloToCpuRuntimePass :
    Pass<"xla-lmhlo-to-cpu-runtime", "mlir::ModuleOp"> {
  let summary = "Converts lmhlo operations to XLA Cpu runtime custom calls";

  let description = [{
      Converts lmhlo dialect operations to XLA Cpu runtime custom calls.
  }];

  let constructor = "createConvertLmhloToCpuRuntimePass()";
}

def LegalizeXlaAbiPass :
    Pass<"xla-legalize-abi", "mlir::ModuleOp"> {
  let summary = "Convers layouts and data formats at ABI boundaries";

  let description = [{
      Converts layouts defined in entry_computation_layout to the default
      layout assumed by the MLIR pipeline. The same is done for custom calls.

      This pass should run immediately after expand-hlo-tuples.
  }];

  let dependentDialects = [
      "mlir::func::FuncDialect", "mlir::mhlo::MhloDialect"
  ];

  let constructor = "createXlaAbiLegalizationPass()";
}

def LegalizeCollectiveOpsPass :
   Pass<"xla-legalize-collective-ops", "mlir::func::FuncOp"> {
  let summary = "Legalizes collective ops to AllToAll and regular ops.";

  let description = [{
      Lowers collective ops to xla_cpu ops.
  }];

  let dependentDialects = [
    "mlir::mhlo::MhloDialect", "mlir::xla_cpu::XlaCpuDialect"
  ];

  let constructor = "createLegalizeCollectiveOpsPass()";
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
