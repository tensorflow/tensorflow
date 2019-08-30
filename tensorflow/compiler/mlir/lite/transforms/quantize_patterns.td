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
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

// Quantize attribute $0 by using quantization parameter from %1.
def QuantizeByQuantizedType : NativeCodeCall<"Quantize($0, $1.getValue())">;

// Squash tfl.dequantize and tfl.quantize pairs.
// TODO(fengliuai): Compare the scale of input and output. This can also be
// squashed to a requantize op if the scales are different.
def : Pat<(TFL_QuantizeOp (TFL_DequantizeOp $in), $qt), (replaceWithValue $in)>;

// Quantize the value of a constant op if the quantization parameters have been
// propagated to the output.
def : Pat<(TFL_QuantizeOp
              (ConstantOp ElementsAttr:$value),
              $qtype),
          (TFL_QConstOp
              $qtype,
              (QuantizeByQuantizedType $value, $qtype))>;
