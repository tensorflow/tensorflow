/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Converts tf.Const to arith.constant for statically shaped, non-opaque constants.
// Needed for QuantizationDriver to recognize constants.
def ConvertTfConstToArithConst : Pat<
  (TF_ConstOp:$res DenseElementsAttr:$value),
  (Arith_ConstantOp $value),
  [(AnyStaticShapeTensor $res)], (addBenefit 10)>;
