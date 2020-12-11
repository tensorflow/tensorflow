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

def TosaLegalizeTFPass : Pass<"tosa-legalize-tf", "FuncOp"> {
  let summary = "Legalize from TensorFlow to TOSA";
  let constructor = "createLegalizeTFPass()";
}

def TosaLegalizeTFLPass : Pass<"tosa-legalize-tfl", "FuncOp"> {
  let summary = "Legalize from TensorFlow Lite to TOSA";
  let constructor = "createLegalizeTFLPass()";
}

def TosaFusebiasTFPass : Pass<"tosa-fuse-bias-tf", "FuncOp"> {
  let summary = "Fuse tf.Op + tf.BiasAdd and legalized to TOSA";
  let constructor = "createFuseBiasTFPass()";
}

def TosaConvertTFLUint8Pass : Pass<"tosa-convert-tfl-uint8", "FuncOp"> {
  let summary = "Convert uint8 graph to int8 graph";
  let constructor = "createConvertTFLUint8Pass()";
}
