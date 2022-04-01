/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the quantization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

// Both Quantize and Dequantize ops have side effects, so we have to define
// patterns to remove dead ones after the quantization rewrite.
def : Pat<(TFL_QuantizeOp:$op $in, $qt), (replaceWithValue $in), [(HasNoUseOf:$op)]>;
def : Pat<(TFL_DequantizeOp:$op $in), (replaceWithValue $in), [(HasNoUseOf:$op)]>;
