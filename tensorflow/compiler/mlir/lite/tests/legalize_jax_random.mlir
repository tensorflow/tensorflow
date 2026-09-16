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
// RUN: litert-opt -tfl-legalize-random %s | FileCheck %s


// CHECK-LABEL:   func @tfl_wrapped_jax_random_normal(
// CHECK-SAME:                                        %[[RNG:.*]]: tensor<2xui32>) -> tuple<tensor<3x4xf32>> {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<[3, 4]> : tensor<2xi32>
// CHECK:           %[[VAL_1:.*]] = "tfl.custom"(%[[VAL_0]]) <{custom_code = "RandomStandardNormal", custom_option = #tfl<const_bytes : "0x">}> : (tensor<2xi32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.tuple %[[VAL_1]] : tuple<tensor<3x4xf32>>
// CHECK:           return %[[VAL_2]] : tuple<tensor<3x4xf32>>
// CHECK:         }
func.func @tfl_wrapped_jax_random_normal(%arg0: tensor<2xui32>) -> tuple<tensor<3x4xf32>> {
  // This is a fake jax random normal body.
  %0 = stablehlo.constant dense<0.0> : tensor<12xf32>
  %1 = "stablehlo.reshape"(%0) : (tensor<12xf32>) -> tensor<3x4xf32>
  %2 = "stablehlo.tuple"(%1) : (tensor<3x4xf32>) -> tuple<tensor<3x4xf32>>
  func.return %2 : tuple<tensor<3x4xf32>>
}


// CHECK-LABEL:   func @tfl_wrapped_jax_random_uniform(
// CHECK-SAME:                                         %[[RNG:.*]]: tensor<2xui32>) -> tuple<tensor<1x2xf32>> {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:           %[[VAL_1:.*]] = "tfl.custom"(%[[VAL_0]]) <{custom_code = "RandomUniform", custom_option = #tfl<const_bytes : "0x">}> : (tensor<2xi32>) -> tensor<1x2xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.tuple %[[VAL_1]] : tuple<tensor<1x2xf32>>
// CHECK:           return %[[VAL_2]] : tuple<tensor<1x2xf32>>
// CHECK:         }
func.func @tfl_wrapped_jax_random_uniform(%arg0: tensor<2xui32>) -> tuple<tensor<1x2xf32>> {
  // This is a fake jax random uniform body.
  %0 = stablehlo.constant dense<0.0> : tensor<2xf32>
  %1 = "stablehlo.reshape"(%0) : (tensor<2xf32>) -> tensor<1x2xf32>
  %2 = "stablehlo.tuple"(%1) : (tensor<1x2xf32>) -> tuple<tensor<1x2xf32>>
  func.return %2 : tuple<tensor<1x2xf32>>
}
