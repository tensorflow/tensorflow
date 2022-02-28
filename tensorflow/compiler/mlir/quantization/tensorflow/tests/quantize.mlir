// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions -quant-quantize -verify-each=false | FileCheck %s

// CHECK-LABEL: add
func @add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %cst_0 = "tf.Const"() {value = dense<-3.5> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<3.5> : tensor<f32>} : () -> tensor<f32>
  %q_x = "quant.qcast"(%arg0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.0:-10>>
  %dq_x = "quant.dcast"(%q_x) : (tensor<8x!quant.uniform<i8:f32, 1.0:-10>>) -> tensor<8xf32>
  %q_y = "quant.qcast"(%cst_0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 2.0>>
  %dq_y = "quant.dcast"(%q_y) : (tensor<!quant.uniform<i8:f32, 2.0>>) -> tensor<f32>
  %add_quant = "tf.AddV2"(%dq_x, %dq_y) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  %q_out = "quant.qcast"(%add_quant) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 3.0:7>>
  %dq_out = "quant.dcast"(%q_out) : (tensor<8x!quant.uniform<i8:f32, 3.0:7>>) -> tensor<8xf32>
  %add_float = "tf.AddV2"(%arg1, %cst_1) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  return %dq_out, %add_float : tensor<8xf32>, tensor<8xf32>
}

// CHECK: %[[quant_add:.*]] = "tf.PartitionedCall"
// CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: f = @fused_add_fn_2
// CHECK-SAME: (tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-10>>, tensor<!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<8x!quant.uniform<i8:f32, 3.000000e+00:7>>
// CHECK: %[[dequantize:.*]] = "quant.dcast"(%[[quant_add]])
// CHECK: %[[float_add:.*]] = "tf.PartitionedCall"
// CHECK-SAME: f = @fused_add_fn_1} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: return %[[dequantize]], %[[float_add]] : tensor<8xf32>, tensor<8xf32>
