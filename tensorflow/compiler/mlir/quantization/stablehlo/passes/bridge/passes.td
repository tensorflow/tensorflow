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

  let constructor = "mlir::quant::stablehlo::createConvertMHLOQuantToIntPass()";
  let dependentDialects = ["chlo::ChloDialect", "mhlo::MhloDialect",
                           "quant::QuantizationDialect",
                           "shape::ShapeDialect",
                           "func::FuncDialect"];
}

def ConvertTFQuantOpsToMHLO : Pass<"quant-convert-tf-quant-ops-to-mhlo", "mlir::func::FuncOp"> {
  let summary = "Convert TF Quant ops to MHLO quantizated ops.";

  let description = [{
    Convert TF Quant ops to MHLO quant ops.
  }];

  let constructor = "mlir::quant::stablehlo::CreateConvertTFQuantOpsToMHLOPass()";
  let dependentDialects = ["TF::TensorFlowDialect", "chlo::ChloDialect",
                           "mhlo::MhloDialect", "tf_type::TFTypeDialect",
                           "quant::QuantizationDialect"];
}

def ConvertTFQuantTypes : Pass<"convert-tf-quant-types", "mlir::func::FuncOp"> {
  let summary = "Replace TensorFlow qint types with int types.";

  let description = [{
    Converts TF ops with qint types to int types. Some UniformQuantized ops
    argument/result allow qint type only. For such cases, add qint <-> int
    tf.Cast around the ops so that they are still valid.
  }];

  let constructor = "mlir::quant::stablehlo::CreateConvertTFQuantTypesPass()";
  let dependentDialects = ["TF::TensorFlowDialect", "tf_type::TFTypeDialect"];
}

def VerifyQuantLegalization : Pass<"verify-quant-legalization", "mlir::func::FuncOp"> {
  let summary = "Verifies that all TF quant ops and types have been legalized.";

  let description = [{
    Ensures that all TF quant ops and types have been legalized to HLO
    and reports an error about which op failed to legalize. This pass
    does not transform any ops and is checking.}];

  let constructor = "mlir::quant::stablehlo::CreateVerifyQuantLegalizationPass()";
  let dependentDialects =  ["tf_type::TFTypeDialect",
                            "quant::QuantizationDialect"];

}
