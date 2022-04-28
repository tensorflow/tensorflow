/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Basically x - y => x + y * -1
def SubToAdd : Pat<(TFL_SubOp $lhs, $rhs, $act),
  (TFL_AddOp $lhs, (TFL_MulOp $rhs,
                  (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>,
                   "-1.0f">), TFL_AF_None), $act)>;

// Squash tfl.dequantize and tfl.quantize pairs.
// TODO(b/185915462): Compare the scale of input and output. This can also be
// squashed to a requantize op if the scales are different.
def FoldQuantizeDequantize: Pat<(TFL_DequantizeOp (TFL_QuantizeOp $arg, $qt)), (replaceWithValue $arg)>;
