// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: odml-to-stablehlo-opt %s --uniform-quantized-stablehlo-to-tfl \
// RUN:     --simplify-hlo --prepare-hlo --unfold-splat-constant-pass \
// RUN:     --tfl-legalize-hlo | FileCheck %s --dump-input=fail

// Runs the HLO -> TFLite legalization passes in the order used by
// `AddPipelinePasses` in python/stablehlo_tfl_pipeline.cc (the PyTorch/JAX
// StableHLO path). The IR there is StableHLO, so
// `uniform-quantized-stablehlo-to-tfl` lowers the uniform-quantized ops to TFL
// up front (it used to see MHLO there and do nothing), and the remaining float
// ops are left to `tfl-legalize-hlo`.

// CHECK-LABEL: func @quantized_dot_general_then_float_add
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x2x3x4xf32>, %[[ARG_1:.+]]: tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>)
func.func @quantized_dot_general_then_float_add(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5xf32> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = "stablehlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
  %3 = stablehlo.uniform_dequantize %2 : (tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>) -> tensor<1x2x3x5xf32>
  %4 = stablehlo.constant dense<1.000000e+00> : tensor<1x2x3x5xf32>
  %5 = stablehlo.add %3, %4 : tensor<1x2x3x5xf32>
  return %5 : tensor<1x2x3x5xf32>
}
// CHECK: %[[QUANT:.+]] = "tfl.quantize"(%[[ARG_0]]) <{qtype = tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>}>
// CHECK: %[[BMM:.+]] = "tfl.batch_matmul"(%[[QUANT]], %[[ARG_1]]) <{adj_x = false, adj_y = false}> : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
// CHECK: %[[DEQUANT:.+]] = "tfl.dequantize"(%[[BMM]]) : (tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>) -> tensor<1x2x3x5xf32>
// CHECK: %[[ADD:.+]] = tfl.add
// CHECK-SAME: %[[DEQUANT]]
// CHECK-NOT: stablehlo.
// CHECK: return %[[ADD]]
