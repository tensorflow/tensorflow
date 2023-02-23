// RUN: tf-quant-opt %s -quant-cast-bf16-ops-to-f32 | FileCheck %s

func.func @cast_bf16_conv_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %6 = "tf.IdentityN"(%5) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %6 : tensor<1x3x2x2xf32>
}

// CHECK: func @cast_bf16_conv_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
// CHECK: %[[cast:.*]] = "tf.Cast"(%[[cst]]) {Truncate = false} : (tensor<2x3x3x2xbf16>) -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cast]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[conv]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

func.func @cast_bf16_avg_pool_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %2 = "tf.AvgPool"(%1) {data_format = "NHWC", device = "", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %5 = "tf.Cast"(%4) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %6 = "tf.Identity"(%5) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %7 = "tf.IdentityN"(%6) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %7 : tensor<1x3x2x2xf32>
}

// CHECK: func @cast_bf16_avg_pool_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %[[avg_pool:.*]] = "tf.AvgPool"(%[[conv]]) {data_format = "NHWC", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[avg_pool]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

func.func @cast_bf16_matmul_to_fp32(%arg0: tensor<1x10xf32>) -> (tensor<1x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<10x2xbf16>} : () -> tensor<10x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x10xf32>) -> tensor<1x10xbf16>
  %1 = "tf.MatMul"(%0, %cst) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x10xbf16>, tensor<10x2xbf16>) -> tensor<1x2xbf16>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x2xbf16>) -> tensor<1x2xbf16>
  %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x2xbf16>) -> tensor<1x2xbf16>
  %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x2xbf16>) -> tensor<1x2xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %6 = "tf.IdentityN"(%5) {device = ""} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  return %6 : tensor<1x2xf32>
}

// CHECK: func @cast_bf16_matmul_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<10x2xf32>} : () -> tensor<10x2xf32>
// CHECK: %[[matmul:.*]] = "tf.MatMul"(%arg0, %[[cst]]) {transpose_a = false, transpose_b = false} : (tensor<1x10xf32>, tensor<10x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[matmul]])
// CHECK: return %[[identity]] : tensor<1x2xf32>

func.func @cast_bf16_depthwise_conv_to_fp32(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x2x2x6xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.DepthwiseConv2dNative"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x2x2x6xbf16>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x2x2x6xbf16>) -> tensor<1x2x2x6xbf16>
  %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x2x2x6xbf16>) -> tensor<1x2x2x6xbf16>
  %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x2x2x6xbf16>) -> tensor<1x2x2x6xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<1x2x2x6xf32>) -> tensor<1x2x2x6xf32>
  %6 = "tf.IdentityN"(%5) {device = ""} : (tensor<1x2x2x6xf32>) -> tensor<1x2x2x6xf32>
  return %6 : tensor<1x2x2x6xf32>
}

// CHECK: func @cast_bf16_depthwise_conv_to_fp32
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK: %[[depthwise_conv:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[cst]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x6xf32>
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[depthwise_conv]]) {device = ""} : (tensor<1x2x2x6xf32>) -> tensor<1x2x2x6xf32>
// CHECK: return %[[identity]] : tensor<1x2x2x6xf32>
