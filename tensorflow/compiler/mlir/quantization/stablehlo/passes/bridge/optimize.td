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

include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"

// This pattern optimizes:
//   conv/dot_general + a + b -> conv/dot_general + (a + b)
//   conv/dot_general - a - b -> conv/dot_general - (a + b)
//   conv/dot_general + a - b -> conv/dot_general + (a - b)
//   conv/dot_general - a + b -> conv/dot_general - (a - b)
foreach OpsTuple = [
    [MHLO_AddOp, MHLO_AddOp, MHLO_AddOp],
    [MHLO_SubtractOp, MHLO_SubtractOp, MHLO_AddOp],
    [MHLO_AddOp, MHLO_SubtractOp, MHLO_SubtractOp],
    [MHLO_SubtractOp, MHLO_AddOp, MHLO_SubtractOp]] in {
  def optimizeConsecutiveConv#OpsTuple[0]#OpsTuple[1] : Pat<
    (OpsTuple[1]
      (OpsTuple[0]
        (MHLO_ConvolutionOp:$conv $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
        $zp_offset),
      $bias),
    (OpsTuple[0]
      (MHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding, $lhs_dilation,
        $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config,
        (returnType $conv)),
      (OpsTuple[2] $zp_offset, $bias)),
    [(TensorOf<[AnyInteger]> $conv)]>;

  def optimizeConsecutiveDotGeneral#OpsTuple[0]#OpsTuple[1] : Pat<
    (OpsTuple[1]
      (OpsTuple[0]
        (MHLO_DotGeneralOp:$dot $lhs, $rhs, $dimension_numbers,
          $precision_config),
        $zp_offset),
      $bias),
    (OpsTuple[0]
      (MHLO_DotGeneralOp $lhs, $rhs,$dimension_numbers, $precision_config,
        (returnType $dot)),
      (OpsTuple[2] $zp_offset, $bias)),
    [(TensorOf<[AnyInteger]> $dot)]>;
}
