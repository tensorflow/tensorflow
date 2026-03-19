/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"
include "stablehlo/dialect/ChloOps.td"

def IsDefinedByConvOrDotGeneralOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<mhlo::ConvolutionOp>()">,
  CPred<"$0.getDefiningOp<mhlo::DotGeneralOp>()">]>>;

def IsNull : Constraint<CPred<"!$0">>;

// This pattern optimizes:
//   conv/dot_general + a + b -> conv/dot_general + (a + b)
//   conv/dot_general - a - b -> conv/dot_general - (a + b)
//   conv/dot_general + a - b -> conv/dot_general + (a - b)
//   conv/dot_general - a + b -> conv/dot_general - (a - b)
foreach OpsTuple = [
    [CHLO_BroadcastAddOp, CHLO_BroadcastAddOp, CHLO_BroadcastAddOp],
    [CHLO_BroadcastSubOp, CHLO_BroadcastSubOp, CHLO_BroadcastAddOp],
    [CHLO_BroadcastAddOp, CHLO_BroadcastSubOp, CHLO_BroadcastSubOp],
    [CHLO_BroadcastSubOp, CHLO_BroadcastAddOp, CHLO_BroadcastSubOp]] in {
  def optimizeConsecutiveConv#OpsTuple[0]#OpsTuple[1] : Pat<
    (OpsTuple[1]
      (OpsTuple[0] $input, $zp_offset, $broadcast_dims_1),
      $bias, $broadcast_dims_2),
    (OpsTuple[0]
      $input,
      (OpsTuple[2] $zp_offset, $bias, $broadcast_dims_2), $broadcast_dims_1),
    [
      (IsNull $broadcast_dims_1),
      (IsNull $broadcast_dims_2),
      (TensorOf<[AnyInteger]> $input),
      (IsDefinedByConvOrDotGeneralOp $input)]>;
}
