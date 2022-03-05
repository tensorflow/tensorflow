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

// RUN: tf-quant-opt %s -quant-convert-fake-quant-to-qdq -quant-lift-quantizable-spots-as-functions -quant-insert-quantized-functions -quant-quantize-composite-functions -symbol-dce | FileCheck %s

func @fake_quant_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -0.1 : f32, max = 0.2 : f32, num_bits = 8} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = "tf.FakeQuantWithMinMaxArgs"(%arg1) {min = -0.3 : f32, max = 0.4 : f32, num_bits = 8} : (tensor<8xf32>) -> tensor<8xf32>
  %2 = "tf.AddV2"(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %3 = "tf.FakeQuantWithMinMaxArgs"(%2) {min = -0.4 : f32, max = 0.6 : f32, num_bits = 8} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// TODO(b/213253905): Enable again once the pipeline is ready to emit quantized composite op
// CHECK: func @fake_quant_add
// NO-CHECK-NEXT:  [[lhs_scale:%.*]] = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
// NO-CHECK-NEXT:  [[lhs_zp:%.*]] = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
// NO-CHECK-NEXT:  [[rhs_scale:%.*]] = "tf.Const"() {value = dense<0.0027450982> : tensor<f32>} : () -> tensor<f32>
// NO-CHECK-NEXT:  [[rhs_zp:%.*]] = "tf.Const"() {value = dense<-19> : tensor<i32>} : () -> tensor<i32>
// NO-CHECK-NEXT:  [[out_scale:%.*]] = "tf.Const"() {value = dense<0.00392156886> : tensor<f32>} : () -> tensor<f32>
// NO-CHECK-NEXT:  [[out_zp:%.*]] = "tf.Const"() {value = dense<-26> : tensor<i32>} : () -> tensor<i32>
// NO-CHECK-NEXT: [[lhs_i8:%.*]] = "tf.StatefulPartitionedCall"(%arg0, [[lhs_zp]], [[lhs_scale]]) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<8xf32>, tensor<i32>, tensor<f32>) -> tensor<8xi8>
// NO-CHECK-NEXT: [[rhs_i8:%.*]] = "tf.StatefulPartitionedCall"(%arg1, [[rhs_zp]], [[rhs_scale]]) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<8xf32>, tensor<i32>, tensor<f32>) -> tensor<8xi8>
// NO-CHECK-NEXT: [[add_i8:%.*]] = "tf.StatefulPartitionedCall"([[lhs_i8]], [[rhs_i8]], [[lhs_zp]], [[lhs_scale]], [[rhs_zp]], [[rhs_scale]], [[out_zp]], [[out_scale]]) {config = "", config_proto = "", executor_type = "", f = @quantized_add_fn} : (tensor<8xi8>, tensor<8xi8>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>) -> tensor<8xi8>
// NO-CHECK-NEXT: [[add:%.*]] = "tf.StatefulPartitionedCall"([[add_i8]], [[out_zp]], [[out_scale]]) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<8xi8>, tensor<i32>, tensor<f32>) -> tensor<8xf32>
// NO-CHECK-NEXT: return [[add:%.*]] : tensor<8xf32>

// CHECK-NOT: func private @quantized_conv2d_relu6_fn
// CHECK: func private @quantize_i8
// CHECK: func private @dequantize_i8
// CHECK: func private @quantized_add_fn_0
