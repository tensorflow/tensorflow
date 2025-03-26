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
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"

//===----------------------------------------------------------------------===//
// Unary ops patterns.
//===----------------------------------------------------------------------===//

def ConvertTFLAbsOp : Pat<(TFL_AbsOp $arg), (Tosa_AbsOp $arg)>;
def ConvertTFLCeilOp : Pat<(TFL_CeilOp $arg), (Tosa_CeilOp $arg)>;
def ConvertTFLFloorOp : Pat<(TFL_FloorOp $arg), (Tosa_FloorOp $arg)>;
def ConvertTFLLogOp : Pat<(TFL_LogOp $arg), (Tosa_LogOp $arg)>;
def ConvertTFLLogicalNotOp : Pat<(TFL_LogicalNotOp $arg), (Tosa_LogicalNotOp $arg)>;

// Removing the quant.stats op for unquantized models.
def ConvertQuantStatOp : Pat<(quantfork_StatisticsOp $value, $layer_stats, $axis_stats, $axis),
                         (replaceWithValue $value)>;

//===----------------------------------------------------------------------===//
// Binary ops patterns.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//
