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
  let dependentDialects = ["TosaDialect", "quant::QuantizationDialect"];
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
