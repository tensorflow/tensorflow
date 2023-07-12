/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// Declare bridge passes that perform legalization and/or lowering.

include "mlir/Pass/PassBase.td"

def ConvertMHLOQuantToInt : Pass<"convert-mhlo-quant-to-int", "mlir::func::FuncOp"> {
  let summary = "Convert from MHLO quantized ops to MHLO primitive ops.";

  let description = [{
    Convert from MHLO quantized ops with MHLO quant types to MHLO primitive ops
    like int ops.
  }];
  let options = [
    Option<"legalize_chlo_", "legalize-chlo", "bool", /*default=*/"true",
        "Legalizes intermediate chlo ops to hlo">
    ];

  let constructor = "mlir::stablehlo::createConvertMHLOQuantToIntPass()";
  let dependentDialects = ["chlo::ChloDialect", "mhlo::MhloDialect",
                           "quant::QuantizationDialect",
                           "shape::ShapeDialect",
                           "func::FuncDialect"];
}
