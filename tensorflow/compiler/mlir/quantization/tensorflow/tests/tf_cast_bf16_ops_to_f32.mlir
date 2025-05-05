// RUN: tf-quant-opt %s -tf-quant-cast-bf16-ops-to-f32 | FileCheck %s

func.func @cast_bf16_conv_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %3 : tensor<1x3x2x2xf32>
}

// CHECK: func @cast_bf16_conv_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>}> {device = ""} : () -> tensor<2x3x3x2xbf16>
// CHECK: %[[cast:.*]] = "tf.Cast"(%[[cst]]) <{Truncate = false}> : (tensor<2x3x3x2xbf16>) -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cast]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[conv]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

func.func @cast_bf16_conv_with_bias_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<2xbf16>} : () -> tensor<2xbf16>
  %cst_0 = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst_0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %2 = "tf.BiasAdd"(%1, %cst) {data_format = "NHWC", device = ""} : (tensor<1x3x2x2xbf16>, tensor<2xbf16>) -> tensor<1x3x2x2xbf16>
  %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %4 = "tf.IdentityN"(%3) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %4 : tensor<1x3x2x2xf32>
}

// CHECK: func @cast_bf16_conv_with_bias_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK: %[[bias_add:.*]] = "tf.BiasAdd"(%[[conv]], %[[cst_0]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[bias_add]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

func.func @cast_bf16_avg_pool_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %2 = "tf.AvgPool"(%1) {data_format = "NHWC", device = "", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %4 = "tf.IdentityN"(%3) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %4 : tensor<1x3x2x2xf32>
}

// CHECK: func @cast_bf16_avg_pool_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK: %[[avg_pool:.*]] = "tf.AvgPool"(%[[conv]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[avg_pool]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

func.func @cast_bf16_matmul_to_fp32(%arg0: tensor<1x10xf32>) -> (tensor<1x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<10x2xbf16>} : () -> tensor<10x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x10xf32>) -> tensor<1x10xbf16>
  %1 = "tf.MatMul"(%0, %cst) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x10xbf16>, tensor<10x2xbf16>) -> tensor<1x2xbf16>
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x2xbf16>) -> tensor<1x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  return %3 : tensor<1x2xf32>
}

// CHECK: func @cast_bf16_matmul_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<10x2xf32>}> : () -> tensor<10x2xf32>
// CHECK: %[[matmul:.*]] = "tf.MatMul"(%arg0, %[[cst]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[matmul]])
// CHECK: return %[[identity]] : tensor<1x2xf32>

func.func @cast_bf16_depthwise_conv_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x2x2x6xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.DepthwiseConv2dNative"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x2x2x6xbf16>
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x2x2x6xbf16>) -> tensor<1x2x2x6xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x2x2x6xf32>) -> tensor<1x2x2x6xf32>
  return %3 : tensor<1x2x2x6xf32>
}

// CHECK: func @cast_bf16_depthwise_conv_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK: %[[depthwise_conv:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[cst]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[depthwise_conv]]) {device = ""} : (tensor<1x2x2x6xf32>) -> tensor<1x2x2x6xf32>
// CHECK: return %[[identity]] : tensor<1x2x2x6xf32>

func.func @cast_bf16_batch_matmul_v2_to_fp32(%arg0: tensor<1x1x10xf32>) -> (tensor<1x1x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<10x2xbf16>} : () -> tensor<10x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x1x10xf32>) -> tensor<1x1x10xbf16>
  %1 = "tf.BatchMatMulV2"(%0, %cst) {adj_x = false, adj_y = false, device = ""} : (tensor<1x1x10xbf16>, tensor<10x2xbf16>) -> tensor<1x1x2xbf16>
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x1x2xbf16>) -> tensor<1x1x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  return %3 : tensor<1x1x2xf32>
}

// CHECK: func @cast_bf16_batch_matmul_v2_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<10x2xf32>}> : () -> tensor<10x2xf32>
// CHECK: %[[batch_matmul:.*]] = "tf.BatchMatMulV2"(%arg0, %[[cst]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[batch_matmul]]) {device = ""} : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK: return %[[identity]] : tensor<1x1x2xf32>

// Tests that an AddV2 op accepting two bf16 operands is transformed into
// an AddV2 op that accepts two fp32 operands.
func.func @cast_bf16_add_v2_to_fp32(%arg0: tensor<2xbf16>, %arg1: tensor<2xbf16>) -> tensor<2xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<2xbf16>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}
// The signature of the function is not changed.
// CHECK: func @cast_bf16_add_v2_to_fp32(%[[ARG_0:.*]]: tensor<2xbf16>, %[[ARG_1:.*]]: tensor<2xbf16>) -> tensor<2xf32>

// bfloat16 operands are cast to f32 operands.
// CHECK-DAG: %[[CAST_0:.*]] = "tf.Cast"(%[[ARG_0]]) <{Truncate = false}> : (tensor<2xbf16>) -> tensor<2xf32>
// CHECK-DAG: %[[CAST_1:.*]] = "tf.Cast"(%[[ARG_1]]) <{Truncate = false}> : (tensor<2xbf16>) -> tensor<2xf32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[CAST_0]], %[[CAST_1]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK: return %[[ADD]] : tensor<2xf32>
