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

// TFLite legalization patterns

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"

// Unary ops patterns.
def : Pat<(TFL_AbsOp $arg), (Tosa_AbsOp $arg)>;
def : Pat<(TFL_CeilOp $arg), (Tosa_CeilOp $arg)>;
def : Pat<(TFL_FloorOp $arg), (Tosa_FloorOp $arg)>;
def : Pat<(TFL_ExpOp $arg), (Tosa_ExpOp $arg)>;
def : Pat<(TFL_LogOp $arg), (Tosa_LogOp $arg)>;
def : Pat<(TFL_RsqrtOp $arg), (Tosa_RsqrtOp $arg)>;
def : Pat<(TFL_LogicalNotOp $arg), (Tosa_LogicalNotOp $arg)>;
def : Pat<(TFL_SqrtOp $arg), (Tosa_ReciprocalOp(Tosa_RsqrtOp $arg))>;
def : Pat<(TFL_CastOp $in), (Tosa_CastOp $in)>;

//===----------------------------------------------------------------------===//
// Binary ops patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TFL_LogicalAndOp $l, $r), (Tosa_LogicalAndOp $l, $r)>;
def : Pat<(TFL_LogicalOrOp $l, $r), (Tosa_LogicalOrOp $l, $r)>;
def : Pat<(TFL_PowOp $l, $r), (Tosa_PowOp $l, $r)>;

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//

