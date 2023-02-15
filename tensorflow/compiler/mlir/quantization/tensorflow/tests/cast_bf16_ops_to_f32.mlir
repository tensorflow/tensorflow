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
// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
// CHECK: %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<1x3x4x3xbf16>) -> tensor<1x3x4x3xf32>
// CHECK: %2 = "tf.Cast"(%[[cst]]) {Truncate = false} : (tensor<2x3x3x2xbf16>) -> tensor<2x3x3x2xf32>
// CHECK: %3 = "tf.Conv2D"(%1, %2)
// CHECK: %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xbf16>
// CHECK: %5 = "tf.Identity"(%4) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
// CHECK: %6 = "tf.Identity"(%5) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
// CHECK: %7 = "tf.Cast"(%6) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
// CHECK: %8 = "tf.Identity"(%7) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %9 = "tf.IdentityN"(%8) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %9 : tensor<1x3x2x2xf32>

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
// TODO(b/269041864): Remove redundant cast ops.
// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
// CHECK: %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<1x3x4x3xbf16>) -> tensor<1x3x4x3xf32>
// CHECK: %2 = "tf.Conv2D"(%1, %[[cst]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xbf16>
// CHECK: %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
// CHECK: %5 = "tf.AvgPool"(%4) {data_format = "NHWC", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %6 = "tf.Cast"(%5) {Truncate = false} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xbf16>
// CHECK: %7 = "tf.Identity"(%6) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
// CHECK: %8 = "tf.Identity"(%7) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
// CHECK: %9 = "tf.Cast"(%8) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
// CHECK: %10 = "tf.Identity"(%9) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: %11 = "tf.IdentityN"(%10) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %11 : tensor<1x3x2x2xf32>
