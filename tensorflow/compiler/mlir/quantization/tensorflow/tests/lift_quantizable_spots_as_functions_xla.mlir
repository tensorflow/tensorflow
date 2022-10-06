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

// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions='target-opset=XLA' | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}, tf_saved_model.semantics} {
  func.func @depthwise_conv(%arg0: tensor<1x3x4x3xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x2x2x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[7.72826624, 8.8264122, 3.64885974]> : tensor<3xf32>} : () -> tensor<3xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.16961098], [2.44217539], [-1.24544525]], [[5.70717144], [5.59951639], [-4.54814768]], [[-4.47071505], [6.03744364], [9.16278743]]], [[[7.51865291], [-2.84365463], [0.0199025106]], [[3.66925859], [4.25404072], [-2.59498501]], [[1.22392368], [0.0616633072], [-9.7246313]]]]> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<1x2x2x3xf32>
    %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<1x2x2x3xf32>, tensor<3xf32>) -> tensor<1x2x2x3xf32>
    %2 = "tf.Relu6"(%1) {device = ""} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    return %3 : tensor<1x2x2x3xf32>
  }
}

// CHECK-LABEL: func @depthwise_conv
// CHECK: "tf.PartitionedCall"
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: f = @composite_depthwise_conv2d_with_bias_and_relu6_fn_1

// CHECK-LABEL: private @composite_depthwise_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[DEPTHWISECONV2D_0:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %arg1)
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map
// CHECK-SAME: {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]}

// -----

func.func @conv_with_non_constant_filter(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-LABEL: func @conv_with_non_constant_filter
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]])
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_1

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map
// CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]

// -----

func.func @conv_with_dynamic_channel_dim(%arg0: tensor<1x3x4x?xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.16961098], [2.44217539], [-1.24544525]], [[5.70717144], [5.59951639], [-4.54814768]], [[-4.47071505], [6.03744364], [9.16278743]]], [[[7.51865291], [-2.84365463], [0.0199025106]], [[3.66925859], [4.25404072], [-2.59498501]], [[1.22392368], [0.0616633072], [-9.7246313]]]]> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %0 = "tf.Conv2D"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x?xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-LABEL: func @conv_with_dynamic_channel_dim
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {{.*}} : () -> tensor<2x3x3x1xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]], %[[CONST_0]])
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_1

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map
// CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
