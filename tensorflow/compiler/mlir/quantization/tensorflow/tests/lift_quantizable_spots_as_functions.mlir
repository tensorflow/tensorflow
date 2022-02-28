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

// RUN: tf-quant-opt %s -split-input-file -tf-fused-kernel-matcher -quant-lift-quantizable-spots-as-functions | FileCheck %s

func @float_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %res = "tf.AddV2"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %res : tensor<8xf32>
}

// CHECK: func @float_add
// CHECK-NEXT: [[res:%.*]] = "tf.PartitionedCall"(%arg0, %arg1)
// CHECK-SAME: f = @fused_add_fn
// CHECK-SAME: (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: return [[res]] : tensor<8xf32>

// CHECK: func private @fused_add_fn
// CHECK-SAME: attributes {tf_quant.fused_function}
// CHECK-NEXT: "tf.AddV2"(%arg0, %arg1)

// -----

// CHECK that _tfl_quant_trait = "fully_quantizable" is added to quantizable
// composite function.

func @multiple_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %cst_0 = "tf.Const"() {value = dense<-3.0> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
  %q1 = "quant.qcast"(%arg0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.0>>
  %dq1 = "quant.dcast"(%q1) : (tensor<8x!quant.uniform<i8:f32, 1.0>>) -> tensor<8xf32>
  %q2 = "quant.qcast"(%cst_0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 2.0>>
  %dq2 = "quant.dcast"(%q2) : (tensor<!quant.uniform<i8:f32, 2.0>>) -> tensor<f32>
  %add_0 = "tf.AddV2"(%dq1, %dq2) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  %q3 = "quant.qcast"(%add_0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 3.0>>
  %dq3 = "quant.dcast"(%q3) : (tensor<8x!quant.uniform<i8:f32, 3.0>>) -> tensor<8xf32>
  %add_1 = "tf.AddV2"(%arg1, %cst_1) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  %res = "tf.AddV2"(%dq3, %add_1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %res : tensor<8xf32>
}

// CHECK-LABEL: func @multiple_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
// CHECK: %[[CONST_1:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[DQ_1:.*]] = "quant.dcast"(%{{.*}}) : (tensor<8x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<8xf32>
// CHECK: %[[DQ_2:.*]] = "quant.dcast"(%{{.*}}) : (tensor<!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<f32>

// CHECK: %[[ADD_QUANT:.*]] = "tf.PartitionedCall"(%[[DQ_1]], %[[DQ_2]])
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: f = @fused_add_fn_3} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>

// CHECK: %[[ADD_Q:.*]] = "quant.qcast"(%[[ADD_QUANT]]) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 3.000000e+00>>
// CHECK: %[[ADD_DQ:.*]] = "quant.dcast"(%[[ADD_Q]]) : (tensor<8x!quant.uniform<i8:f32, 3.000000e+00>>) -> tensor<8xf32>

// CHECK: %[[ADD_FLOAT:.*]] = "tf.PartitionedCall"(%arg1, %[[CONST_1]])
// CHECK-SAME: f = @fused_add_fn_2} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>

// CHECK: %[[RESULT:.*]] = "tf.PartitionedCall"(%[[ADD_DQ]], %[[ADD_FLOAT]])
// CHECK-SAME: f = @fused_add_fn_1} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

// CHECK: return %[[RESULT]] : tensor<8xf32>


// Composite function definitions
// CHECK-LABEL: func private @fused_add_fn_3(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<8xf32> attributes {tf_quant.fused_function} {
// CHECK: %[[ADDV2_0:.*]] = "tf.AddV2"(%arg0, %arg1) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: return %[[ADDV2_0]] : tensor<8xf32>

// CHECK-LABEL: func private @fused_add_fn_2(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<8xf32> attributes {tf_quant.fused_function} {
// CHECK: %[[ADDV2_1:.*]] = "tf.AddV2"(%arg0, %arg1) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: return %[[ADDV2_1]] : tensor<8xf32>

// CHECK-LABEL: func private @fused_add_fn_1(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> attributes {tf_quant.fused_function} {
// CHECK: %[[ADDV2_2:.*]] = "tf.AddV2"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
// CHECK: return %[[ADDV2_2]] : tensor<8xf32>


// -----

// CHECK-LABEL: float_conv
func @float_conv(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>


  %3 = "tf.Conv2D"(%arg0, %arg1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %5 = "tf.Relu"(%4) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>


  %6 = "tf.Conv2D"(%arg0, %arg1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %7 = "tf.BiasAdd"(%6, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %2, %5, %7 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]])
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @fused_conv2d_relu6_fn_1}
// CHECK: %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]])
// CHECK-SAME: f = @fused_conv2d_relu_fn_1}
// CHECK: %[[PARTITIONEDCALL_2:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]])
// CHECK-SAME: f = @fused_conv2d_fn_1}
// CHECK: return %[[PARTITIONEDCALL_0]], %[[PARTITIONEDCALL_1]], %[[PARTITIONEDCALL_2]]
// CHECK: }

// CHECK-LABEL: private @fused_conv2d_relu6_fn_1
// CHECK-NEXT: %[[conv:.*]] = "tf._FusedConv2D"(%arg0, %arg1, %arg2)
// CHECK-SAME: {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations"
// CHECK-SAME: data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu6"], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
// CHECK-NEXT: return %[[conv]]

// CHECK-LABEL: private @fused_conv2d_relu_fn_1
// CHECK-NEXT: tf._FusedConv2D"(%arg0, %arg1, %arg2)
// CHECK-SAME: fused_ops = ["BiasAdd", "Relu"]

// CHECK-LABEL: private @fused_conv2d_fn_1
// CHECK-NEXT: tf._FusedConv2D"(%arg0, %arg1, %arg2)
// CHECK-SAME: fused_ops = ["BiasAdd"]
}


// -----

func @float_conv_strides_equals_to_dilations(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// CHECK-LABEL: func @float_conv_strides_equals_to_dilations(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> {
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @fused_conv2d_relu6_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK: return %[[PARTITIONEDCALL_0]] : tensor<*xf32>
// CHECK: }
// CHECK-LABEL: func private @fused_conv2d_relu6_fn_1(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.fused_function} {
// CHECK-NEXT: %[[CONV2D_0:.*]] = "tf._FusedConv2D"(%arg0, %arg1, %arg2)
// CHECK-SAME: {attr_map = "0:dilations,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations"
// CHECK-SAME: data_format = "NHWC", device = "", dilations = [1, 1, 2, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu6"], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
// CHECK-NEXT: return %[[CONV2D_0]] : tensor<*xf32>
