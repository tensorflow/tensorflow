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
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.td"


// Patterns to legalize mhlo to tfl.

// TODO(pulkitb): Both TFL_ConstOp and ConstantOp seem to work. Decide which
// one is more apropos.
def LegalizeConst : Pat<(MHLO_ConstantOp $value),
                        (Arith_ConstantOp $value)>;

def LegalizeMul : Pat<(MHLO_MulOp $lhs, $rhs),
                      (TFL_MulOp  $lhs, $rhs, TFL_AF_None)>;

def LegalizeDiv : Pat<(MHLO_DivOp $lhs, $rhs),
                      (TFL_DivOp  $lhs, $rhs, TFL_AF_None)>;

def LegalizeAdd : Pat<(MHLO_AddOp $lhs, $rhs),
                      (TFL_AddOp $lhs, $rhs, TFL_AF_None)>;

def LegalizeSub : Pat<(MHLO_SubtractOp $lhs, $rhs),
                      (TFL_SubOp $lhs, $rhs, TFL_AF_None)>;

def LegalizeSqrt : Pat<(MHLO_SqrtOp $arg), (TFL_SqrtOp $arg)>;

def LegalizeSelect : Pat<(MHLO_SelectOp $cond, $x, $y),
                         (TFL_SelectOp $cond, $x, $y)>;
