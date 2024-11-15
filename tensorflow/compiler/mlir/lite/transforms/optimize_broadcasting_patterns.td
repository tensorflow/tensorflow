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

// Patterns to optimize explicit broadcasting-like patterns, because TFLite Ops
// support implicit broadcasting.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"

// Pattern for skipping FillOp if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseFillOpBroadcastIntoFollowingBinary<Op BinaryOp> {
  def FoldFillOpIntoBinaryOpRHS#BinaryOp : Pat<
    (BinaryOp:$result $operand,
      (TFL_FillOp:$fill_output $fill_dims,
        (Arith_ConstantOp:$fill_value $val)), $act_func),
    (BinaryOp $operand, $fill_value, $act_func),
  [(OperandsBroadcastToOutputType $operand, $fill_value, $result),
   (HasRankAtMost<4> $operand),
   (HasRankAtMost<4> $fill_output),
   (IsRankLessThanEqualTo $fill_output, $operand)]>;

  def FoldFillOpIntoBinaryOpLHS#BinaryOp : Pat<
    (BinaryOp:$result
      (TFL_FillOp:$fill_output $fill_dims, (Arith_ConstantOp:$fill_value $val)),
      $operand, $act_func),
    (BinaryOp $fill_value, $operand, $act_func),
  [(OperandsBroadcastToOutputType $operand, $fill_value, $result),
   (HasRankAtMost<4> $operand),
   (HasRankAtMost<4> $fill_output),
   (IsRankLessThanEqualTo $fill_output, $operand)]>;
}

foreach BinaryOp = [TFL_AddOp, TFL_SubOp, TFL_DivOp, TFL_MulOp] in {
  defm : FuseFillOpBroadcastIntoFollowingBinary<BinaryOp>;
}