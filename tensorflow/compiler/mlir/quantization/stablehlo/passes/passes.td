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
      "mlir::func::FuncDialect",
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
        "MLIR dump file name.">,
    Option<"merge_fusion_with_dequantize_",
        "merge-fusion-with-dequantize",
        "bool", /*default=*/"false",
        "Whether to merge quantized conv/dot_general fusion with subsequent dequantize.">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::quant::QuantDialect",
    "mlir::quantfork::QuantizationForkDialect",
    "TF::TensorFlowDialect",
  ];
}

def PrepareQuantizePass : Pass<"stablehlo-prepare-quantize", "mlir::ModuleOp"> {
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
      "mlir::quant::QuantDialect",
      "mlir::quantfork::QuantizationForkDialect",
      "mlir::arith::ArithDialect",
  ];
}

def QuantizePass : Pass<"stablehlo-quantize", "mlir::ModuleOp"> {
  let summary = "Applies static-range quantization on ops by converting quantfork.qcast, quantfork.dcast, and float op into uniform quantized ops .";
  let options = [
    Option<"enable_per_channel_quantized_weight_",
        "enable-per-channel-quantized-weight",
        "bool", /*default=*/"true",
        "Whether to enable per-channel quantized weights.">,
  ];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::quant::QuantDialect",
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

def XlaCallModuleToCallPass : Pass<"stablehlo-xla-call-module-to-call", "ModuleOp"> {
  let summary = "Convert XlaCallModuleOp to func.call op";
  let dependentDialects = [
    "TF::TensorFlowDialect",
  ];
}

def MergeFusionWithDequantizePass : Pass<"stablehlo-merge-fusion-with-dequantize", "mlir::ModuleOp"> {
  let summary = "Merge quantized conv/dot_general fusion with subsequent dequantize.";
  let dependentDialects = [
    "chlo::ChloDialect",
    "mlir::stablehlo::StablehloDialect",
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
    "mlir::quant::QuantDialect",
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def OptimizeGraphPass : Pass<"optimize-graph", "ModuleOp"> {
  let summary = "Optimize the sub-optimal patterns after quantization.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect",];
}

def NchwConvolutionToNhwcPass : Pass<"stablehlo-nchw-convolution-to-nhwc", "mlir::func::FuncOp"> {
  let summary = "Converts stablehlo.convolution op of NCHW format to -> NHWC.";
  let description = [{
    Matches `ConvolutionOp`s with NCHW format and converts it to NHWC
    format by inserting `TransposeOp`s to input, filter, and output tensors.
    In terms of dimension numbers, this matches
    `[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]` format and converts it to
    `[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]` format.

    This pass is useful to convert models that conventionally use the NCHW
    format to target hardwares that are more NHWC-friendly.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def DeferActivationTransposePass : Pass<"stablehlo-defer-activation-transpose", "mlir::func::FuncOp"> {
  let summary = "Merges stablehlo.transpose for activations.";
  let description = [{
    Defers activation transposes (e.g. LHS of `stablehlo.add`) to the output and
    optionally inserts `stablehlo.transpose`s to match the shape of operands.
    This is useful when recursively pushing down the extra `stablehlo.transpose`
    inserted to activation tensors after running `NchwConvolutionToNhwcPass`.

    Currently only converts limited cases that appear in NCHW->NHWC 2D
    convolution conversion, to avoid introducing unwanted pessimizations.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def InsertWeightParamPass : Pass<"stablehlo-insert-weight-param", "mlir::func::FuncOp"> {
  let summary = "Insert quantization parameters of weights for weight-only quantization and dynamic range quantization.";
  let dependentDialects = [
      "mlir::stablehlo::StablehloDialect",
      "TF::TensorFlowDialect",
      "mlir::quant::QuantDialect",
      "mlir::quantfork::QuantizationForkDialect",
  ];
}

def FoldConstantTransposePass : Pass<"stablehlo-fold-constant-transpose", "mlir::func::FuncOp"> {
  let summary = "Folds stablehlo.constant -> stablehlo.transpose patterns.";
  let description = [{
    Finds patterns where a `stablehlo.constant` is directly followed by a
    `stablehlo.transpose` and folds them into a single `stablehlo.constant`.
    This is considered an aggressive optimization, but it is useful to eliminate
    `stablehlo.constant`->`stablehlo.transpose` patterns which are often
    by-products of other shape conversion optimizations, such as NCHW->NHWC
    convolution conversion.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def RemoveShardingCustomCallPass : Pass<"stablehlo-remove-sharding-custom-call", "mlir::func::FuncOp"> {
  let summary = "Removes `stablehlo.custom_call @Sharding`";
  let description = [{
    Finds `stablehlo.custom_call @Sharding` and removes all instances of them,
    replacing the usages by its operand. This is used where sharding doesn't
    make much sense or sharding custom calls are incompatible, e.g. on-device
    targets.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def InsertCalibrationStatisticsSaverPass : Pass<"stablehlo-insert-calibration-statistics-saver", "ModuleOp"> {
  let summary = "Inserts `CalibrationStatisticsSaver` op to collect and save calibration statistics.";
  let description = [{
    Finds all `CustomAggregator` ops in the each function and add a single
    `CalibrationStatisticsSaver` op at the end of the function to collect their
    statistics.
  }];
  let options = [
    ListOption<"aggregator_ops_to_ignore_", "aggregator-ops-to-ignore", "std::string",
               "Ops to ignore when inserting CalibrationStatisticsSaver.">,
    Option<"calibration_data_dir_", "calibration-data-dir",
        "std::string", /*default=*/"",
        "The directory to save calibration data.">,
  ];
  let dependentDialects = ["TF::TensorFlowDialect"];
}
