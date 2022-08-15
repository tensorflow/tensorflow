// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -split-input-file -inline -quant-replace-cast-hacks-with-tf-xla-ops | FileCheck %s

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @conv_with_filter_larger_than_1GB(%arg0: tensor<1x2240x2240x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<1x2240x1120x512xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {value = dense<2> : tensor<960x960x3x512xi8>} : () -> tensor<960x960x3x512xi8>
    %cst_0 = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_2 = "tf.Const"() {value = dense<0.0027450982> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {value = dense<-19> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<0.01> : tensor<512xf32>} : () -> tensor<512xf32>
    %cst_5 = "tf.Const"() {value = dense<0> : tensor<512xi32>} : () -> tensor<512xi32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_0, %cst_1) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x2240x2240x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2240x2240x3xi8>
    %1 = "tf.PartitionedCall"(%0, %cst, %cst_0, %cst_1, %cst_4, %cst_5, %cst_2, %cst_3) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_relu_fn_0} : (tensor<1x2240x2240x3xi8>, tensor<960x960x3x512xi8>, tensor<f32>, tensor<i32>, tensor<512xf32>, tensor<512xi32>, tensor<f32>, tensor<i32>) -> tensor<1x2240x1120x512xi8>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x2240x1120x512xi8>) -> tensor<1x2240x1120x512xi8>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x2240x1120x512xi8>) -> tensor<1x2240x1120x512xi8>
    %4 = "tf.PartitionedCall"(%3, %cst_2, %cst_3) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<1x2240x1120x512xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2240x1120x512xf32>
    return %4 : tensor<1x2240x1120x512xf32>
  }
  func.func private @quantize_i8(%arg0: tensor<1x2240x2240x3xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x2240x2240x3xi8> {
    %0 = "tf.Div"(%arg0, %arg1) : (tensor<1x2240x2240x3xf32>, tensor<f32>) -> tensor<1x2240x2240x3xf32>
    %1 = "tf.Round"(%0) : (tensor<1x2240x2240x3xf32>) -> tensor<1x2240x2240x3xf32>
    %2 = "tf.Cast"(%1) : (tensor<1x2240x2240x3xf32>) -> tensor<1x2240x2240x3xi32>
    %3 = "tf.AddV2"(%2, %arg2) : (tensor<1x2240x2240x3xi32>, tensor<i32>) -> tensor<1x2240x2240x3xi32>
    %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x2240x2240x3xi32>) -> tensor<1x2240x2240x3xi8>
    return %4 : tensor<1x2240x2240x3xi8>
  }
  func.func private @dequantize_i8(%arg0: tensor<1x2240x1120x512xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x2240x1120x512xf32> {
    %0 = "tf.Cast"(%arg0) : (tensor<1x2240x1120x512xi8>) -> tensor<1x2240x1120x512xi32>
    %1 = "tf.Sub"(%0, %arg2) : (tensor<1x2240x1120x512xi32>, tensor<i32>) -> tensor<1x2240x1120x512xi32>
    %2 = "tf.Cast"(%1) : (tensor<1x2240x1120x512xi32>) -> tensor<1x2240x1120x512xf32>
    %3 = "tf.Mul"(%2, %arg1) : (tensor<1x2240x1120x512xf32>, tensor<f32>) -> tensor<1x2240x1120x512xf32>
    return %3 : tensor<1x2240x1120x512xf32>
  }
  func.func private @quantized_conv2d_with_relu_fn_0(%arg0: tensor<1x2240x2240x3xi8>, %arg1: tensor<960x960x3x512xi8>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<512xf32>, %arg5: tensor<512xi32>, %arg6: tensor<f32>, %arg7: tensor<i32>) -> tensor<1x2240x1120x512xi8> {
    %cst = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x2240x2240x3xi8>) -> tensor<1x2240x2240x3xi32>
    %1 = "tf.Sub"(%0, %arg3) : (tensor<1x2240x2240x3xi32>, tensor<i32>) -> tensor<1x2240x2240x3xi32>
    %2 = "tf.Identity"(%arg1) : (tensor<960x960x3x512xi8>) -> tensor<960x960x3x512xi8>
    %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<960x960x3x512xi8>) -> tensor<960x960x3x512xi32>
    %4 = "tf.Sub"(%3, %arg5) : (tensor<960x960x3x512xi32>, tensor<512xi32>) -> tensor<960x960x3x512xi32>
    %5 = "tf.Conv2D"(%1, %4) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2240x2240x3xi32>, tensor<960x960x3x512xi32>) -> tensor<1x2240x1120x512xi32>
    %6 = "tf.Mul"(%arg2, %arg4) : (tensor<f32>, tensor<512xf32>) -> tensor<512xf32>
    %7 = "tf.Div"(%6, %arg6) : (tensor<512xf32>, tensor<f32>) -> tensor<512xf32>
    %8 = "tf.Cast"(%5) {Truncate = false} : (tensor<1x2240x1120x512xi32>) -> tensor<1x2240x1120x512xf32>
    %9 = "tf.Mul"(%7, %8) : (tensor<512xf32>, tensor<1x2240x1120x512xf32>) -> tensor<1x2240x1120x512xf32>
    %10 = "tf.Round"(%9) : (tensor<1x2240x1120x512xf32>) -> tensor<1x2240x1120x512xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<1x2240x1120x512xf32>) -> tensor<1x2240x1120x512xi32>
    %12 = "tf.AddV2"(%11, %arg7) : (tensor<1x2240x1120x512xi32>, tensor<i32>) -> tensor<1x2240x1120x512xi32>
    %13 = "tf.Maximum"(%cst_0, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = "tf.ClipByValue"(%12, %13, %cst) : (tensor<1x2240x1120x512xi32>, tensor<i32>, tensor<i32>) -> tensor<1x2240x1120x512xi32>
    %15 = "tf.Cast"(%14) {Truncate = false} : (tensor<1x2240x1120x512xi32>) -> tensor<1x2240x1120x512xi8>
    return %15 : tensor<1x2240x1120x512xi8>
  }

// CHECK-LABEL: func @conv_with_filter_larger_than_1GB
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<-237772800> : tensor<512xi32>} : () -> tensor<512xi32>
// CHECK: %[[PADV2_0:.*]] = "tf.PadV2"
// CHECK: %[[XLACONVV2_0:.*]] = "tf.XlaConvV2"(%[[PADV2_0]]
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[XLACONVV2_0]], %[[CONST]])
}
