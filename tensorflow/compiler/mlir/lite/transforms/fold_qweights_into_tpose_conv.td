/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Xnnpack supports quantized weights for any datatype input. Tflite does not.
// Remove dequant around weight input and pass the quantized weights directly.
// This needs its own file for the following reason;
// when including patterns generated with tablegen in C++, there is no
// way to select specific patterns within a `td` file to populate
// a `RewritePatternSet` with.
// This pattern blindly assumes runtme support and is currently flag protected.
// TODO : b/339546184 - If/when reference runtime supports this computation,
// remove the guarding flag and put this back into `optimize_patterns.td`.
def FoldQuantWeightsIntoTposeConv : Pat<
  (TFL_TransposeConvOp
    $output_shape,
    (TFL_DequantizeOp $quant_weights),
    $quant_input,
    $bias, $padding, $stride_h, $stride_w, $faf),
  (TFL_TransposeConvOp $output_shape, $quant_weights,
    $quant_input, $bias, $padding, $stride_h, $stride_w, $faf)>;

