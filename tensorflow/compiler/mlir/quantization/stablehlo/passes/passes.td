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

include "mlir/Pass/PassBase.td"

def QuantizeWeightPass : Pass<"stablehlo-quantize-weight", "mlir::func::FuncOp"> {
  let summary = "Quantizes the weight component of StableHLO graph.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let constructor = "mlir::quant::stablehlo::CreateQuantizeWeightPass()";
}

def UnfuseMhloBatchNormPass : Pass<"stablehlo-unfuse-mhlo-batch-norm", "mlir::func::FuncOp"> {
  let summary = "Unfuses batch normalization into arithmetic ops.";
}

def LiftQuantizableSpotsAsFunctionsPass : Pass<"stablehlo-lift-quantizable-spots-as-functions", "mlir::ModuleOp"> {
  let summary = "Replace quantization candidates with composite functions into the module.";
  let description = [{
    Mark frequent fusible patterns as functions for quantization targets.
    In addition to brining performance benefits by reducing q/dq op overhead in non-full quantization,
    this brings higher accuracy by keeping a smaller range when quantizing ops
    that disperse values. (ex: convolution, dot_general)
  }];
  let dependentDialects = [
      "mlir::stablehlo::StablehloDialect",
      "TF::TensorFlowDialect",
  ];
}

def ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass : Pass<"stablehlo-replace-stablehlo-ops-in-main-function-with-xla-call-module-ops", "mlir::ModuleOp"> {
  let summary = "Replaces the StableHLO ops with a separate XlaCallModuleOps.";
  let description = [{
     Replaces the StableHLO ops in the main function block with
     tf.XlaCallModuleOps as separate subgraphs. Wires them back to the main
     function block to be compatible with SavedModel structure.
  }];
}

def RestoreFunctionNamePass : Pass<"stablehlo-restore-function-name", "ModuleOp"> {
  let summary = "Restores function name from XlaCallModule op.";
}

def QuantizeCompositeFunctionsPass : Pass<"stablehlo-quantize-composite-functions", "ModuleOp"> {
  let summary = "Quantize composite functions with QDQ input / outputs.";
  let options = [
    Option<"enable_per_channel_quantized_weight_",
        "enable-per-channel-quantized-weight",
        "bool", /*default=*/"true",
        "Whether to enable per-channel quantized weights.">,
    Option<"mlir_dump_file_name_", "mlir-dump-file-name",
        "std::optional<std::string>", /*default=*/"std::nullopt",
        "MLIR dump file name.">
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect",
    "TF::TensorFlowDialect",
  ];
}

def PrepareQuantizePass : Pass<"stablehlo-prepare-quantize", "mlir::func::FuncOp"> {
  let summary = "Prepare StableHLO dialect for static range quantization by converting quantfork.stats into quantfork.qcast and dcast ops.";
  let options = [
    Option<"enable_per_channel_quantized_weight_",
        "enable-per-channel-quantized-weight",
        "bool", /*default=*/"true",
        "Whether to enable per-channel quantized weights.">,
    Option<"bit_width_", "bit-width", "int", /*default=*/"8",
        "Bitwidth of quantized integer">
    ];
  let dependentDialects = [
      "mlir::stablehlo::StablehloDialect",
      "mlir::quant::QuantizationDialect",
      "mlir::quantfork::QuantizationForkDialect",
      "mlir::arith::ArithDialect",
  ];
}

def QuantizePass : Pass<"stablehlo-quantize", "mlir::ModuleOp"> {
  let summary = "Applies static-range quantization on ops by converting quantfork.qcast, quantfork.dcast, and float op into uniform quantized ops .";
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect",
  ];
}

def PostQuantizePass : Pass<"stablehlo-post-quantize", "mlir::func::FuncOp"> {
  let summary = "Apply clean-up after quantization.";
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::quantfork::QuantizationForkDialect",
  ];
}

def UnwrapXlaCallModuleOpPass : Pass<"stablehlo-unwrap-xla-call-module-op", "ModuleOp"> {
  let summary = "Unwrap XlaCallModuleOps into inline functions if not used for quantizing fused patterns.";
  let dependentDialects = ["TF::TensorFlowDialect"];
}

def ConvertFuncToBfloat16Pass : Pass<"stablehlo-convert-func-to-bfloat16", "mlir::func::FuncOp"> {
  let summary = "Convert a StableHLO function to bfloat16";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def ConvertXlaCallModuleOpToBfloat16Pass : Pass<"stablehlo-convert-xla-call-module-op-to-bfloat16", "mlir::func::FuncOp"> {
  let summary = "Convert serialized XlaCallModuleOp to bfloat16";
  let dependentDialects = [
    "TF::TensorFlowDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def PopulateShapePass : Pass<"populate-shape", "ModuleOp"> {
  let summary = "Populate output shape with known information for CustomAggregatorOp and XlaCallModuleOp.";
  let dependentDialects = ["TF::TensorFlowDialect"];
}

def OptimizeGraphPass : Pass<"optimize-graph", "ModuleOp"> {
  let summary = "Optimize the sub-optimal patterns after quantization.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect",];
}
