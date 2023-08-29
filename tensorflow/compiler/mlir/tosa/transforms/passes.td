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

def TosaLegalizeTFPass : Pass<"tosa-legalize-tf", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow to TOSA";
  let constructor = "createLegalizeTFPass()";
  let dependentDialects = ["TosaDialect",
    "quant::QuantizationDialect","quantfork::QuantizationForkDialect"
  ];
}

def TosaLegalizeTFLPass : Pass<"tosa-legalize-tfl", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow Lite to TOSA. Options allow selective disabling/enabling of rewrites";
  let constructor = "createLegalizeTFLPass()";

  let dependentDialects = ["TosaDialect"];
  let options = [
    ListOption<"disabled_patterns_", "disable-patterns", "std::string",
               "Labels of patterns that should be filtered out during application">,
    ListOption<"enabled_patterns_", "enable-patterns", "std::string",
               "Labels of patterns that should be used during application, all other patterns are filtered out">,
  ];
}

def TosaLegalizeTFTFLPass : Pass<"tosa-legalize-tf-tfl", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow / TensorFlow Lite to TOSA";
  let constructor = "createLegalizeTFTFLPass()";
  let dependentDialects = ["TosaDialect"];
}

def TosaFusebiasTFPass : Pass<"tosa-fuse-bias-tf", "mlir::func::FuncOp"> {
  let summary = "Fuse tf.Op + tf.BiasAdd and legalized to TOSA";
  let constructor = "createFuseBiasTFPass()";
  let dependentDialects = ["TosaDialect"];
}

def TosaConvertTFLUint8Pass : Pass<"tosa-convert-tfl-uint8", "mlir::func::FuncOp"> {
  let summary = "Convert uint8 graph to int8 graph";
  let constructor = "createConvertTFLUint8Pass()";
  let dependentDialects = ["TosaDialect"];
}

def TosaStripQuantTypesPass : Pass<"tosa-strip-quant-types", "mlir::func::FuncOp"> {
  let summary = "Convert all quant types to their storage type.";
  let constructor = "createStripQuantTypesPass()";
  let dependentDialects = ["TosaDialect"];
}

def TosaLowerComplexTypesPass : Pass<"tosa-lower-complex-types", "mlir::func::FuncOp"> {
  let summary = "Convert all complex types to their storage type.";
  let description = [{
    Complex tensors are not supported in TOSA and are instead represented by two floating
    point tensors corresponding to the "real" and "imag" parts of the complex tensor. This
    pass replaces all complex tensors of shape [x, ..., y] with floating point tensors of
    shape [x, ..., y, 2] where each inner pair of values represents a complex value. In
    doing so, a 1:1 mapping between TFL and TOSA input/output tensors is maintained. The
    pass will fail if not all complex tensors are converted.
  }];
  let constructor = "createLowerComplexTypesPass()";
  let dependentDialects = ["TosaDialect"];
}

def TosaDequantizeTFLSoftmaxPass : Pass<"tosa-dequantize-tfl-softmax", "mlir::func::FuncOp"> {
  let summary = "Dequantize TFLite Softmax ops.";
  let description = [{
    This pass rewrites quantized TFLite Softmax ops as: Dequantize, (float) Softmax, Quantize.
    It is a work around for current performance issues with quantized Softmax codegen.
    For instance it is a 20% end-to-end speedup on certain Softmax-heavy BERTs.
    The compromise here is that this breaks bit-for-bit agreement with the quantized
    Softmax lowering. But as Softmax isn't currently a TOSA op, this isn't a TOSA
    spec violation - just breaking an invariant in the TFL-to-TOSA import pipeline.
    See https://github.com/google/iree/issues/8974.
  }];
  let constructor = "createDequantizeTFLSoftmaxPass()";
  let dependentDialects = ["mlir::TFL::TFLDialect", "quantfork::QuantizationForkDialect"];
}

def LowerGlobalTensors :
    Pass<"tflite-lower-global-tensors", "mlir::ModuleOp"> {
  let summary = "Lowers TFLite global tensors to MLProgram dialect variables.";
  let constructor = "createLowerGlobalTensorsPass()";
}

def RetainCallOnceFuncs :
    Pass<"tflite-retain-call-once-funcs", "mlir::ModuleOp"> {
  let summary = "Guarantees that functions used by tfl.call_once are retained.";
  let constructor = "createRetainCallOnceFuncsPass()";
}

def StripFunctionMetadata :
    Pass<"tosa-tflite-strip-function-metadata", "mlir::func::FuncOp"> {
  let summary = "Strip all unneeded TF/TFLite specific metadata.";
  let constructor = "createStripFunctionMetadataPass()";
}

def StripModuleMetadata :
    Pass<"tosa-tflite-strip-module-metadata", "mlir::ModuleOp"> {
  let summary = "Strip all unneeded TF/TFLite specific metadata.";
  let constructor = "createStripModuleMetadataPass()";
}

def VerifyFullyConverted :
    Pass<"tosa-tflite-verify-fully-converted", "mlir::func::FuncOp"> {
  let summary = "Verifies that all TFLite frontend ops were converted and none remain.";
  let constructor = "createVerifyFullyConvertedPass()";
}

def ConvertFunctionMetadata :
    Pass<"tosa-tflite-convert-function-metadata", "mlir::func::FuncOp"> {
  let summary = "Converts TFLite input attributes to MLProgram arg attributes on functions.";
  let constructor = "createConvertFunctionMetadataPass()";
}

